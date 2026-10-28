"""Compatibility checks for explicit historical batch imports; not the inference pipeline."""
import hashlib
import json
import sys

from run_storage import ensure_run_writable, write_json
from validation_cache import check_cached_table

# Changes to scheduling/reporting do not change inference equivalence.
ORCHESTRATION_FILES = {'run_size_validation.py', 'batch_progress.py',
                       'prepare_size_validation.py', 'select_full_validation.py',
                       'run_storage.py', 'validation_cache.py', 'validation_reuse.py',
                       'validation_reporting.py'}


# Exact historical GT-only exceptions; unknown inference changes still fail.
GT_V71_EQUIVALENT_HASHES = {
    'real_scene.py': ('98fbeb2e8a7dc6a49e4fd798718b12addeb4c3516e17d4088e2b9ded48fc46e4',
                      'e485b62afb81bd094d07ba7f544fbc847f2e19164a41edbb1dea692ca71b86cf'),
    'v4_policy.py': ('4df86aabc388bfcb4cd1807cbb680a8ee5a1aa190c2b102989f062210b876e2f',
                     'b5f5b9272cf746f1a0f6497530c0609043b084d3a0253eba22d003e6ec72a9b6'),
}


# Explicit mixed-version import, not a claim that v7.1 and v7.3 are equivalent.
V73_LEGACY_GT_HASHES = {
    'active_diagnostic.py': '9de7357ac6409c151b45c9568d90df4aa792850e529095865ab1bac848ced11e',
    'v4_policy.py': 'b5f5b9272cf746f1a0f6497530c0609043b084d3a0253eba22d003e6ec72a9b6',
    'response_schemas.py': 'c338333bf539d3114311f710370588e8efa7ea741405e5b0e27fa03614caf1db',
}


def comparable_settings(settings, reuse=False):
    value = {**settings, 'code_sha256': {k: v for k, v in settings['code_sha256'].items()
                                       if k not in ORCHESTRATION_FILES}}
    if (settings.get('object_table') == 'gt' and
            settings.get('v4', {}).get('version') == 'relative-size-v7.1'):
        for name, (current, baseline) in GT_V71_EQUIVALENT_HASHES.items():
            if value['code_sha256'].get(name) == current:
                value['code_sha256'][name] = baseline
    for key in ('workers', 'api_concurrency', 'api_slot_dir'):
        value.pop(key, None)
    value['api_config'] = {k: v for k, v in settings.get('api_config', {}).items()
                           if k not in ('api_concurrency', 'api_slot_dir')}
    if reuse:
        for key in ('selection', 'input_sha256', 'scene_root'):
            value.pop(key, None)
    return value


def reuse_results(source, output, rows, settings, allow_previous_version=False):
    """Snapshot evaluations; keep images and traces in the original run directory."""
    ensure_run_writable(output)
    source = source.resolve()
    if source == output:
        raise ValueError('--reuse-from must differ from --output')
    previous = json.loads((source / 'manifest.json').read_text())
    previous_compare = comparable_settings(previous, reuse=True)
    current_compare = comparable_settings(settings, reuse=True)
    legacy = (allow_previous_version and
              previous.get('v4', {}).get('version') != settings.get('v4', {}).get('version'))
    if legacy:
        if (previous.get('object_table') != 'gt' or settings.get('object_table') != 'gt' or
                previous.get('v4', {}).get('version') != 'relative-size-v7.1' or
                settings.get('v4', {}).get('version') != 'relative-size-v7.3'):
            raise ValueError('Previous-version reuse supports only the verified v7.1 GT baseline into v7.3 GT')
        for name, expected in V73_LEGACY_GT_HASHES.items():
            if previous_compare['code_sha256'].get(name) != expected:
                raise ValueError(f'Unverified legacy inference file: {name}')
            previous_compare['code_sha256'].pop(name)
            current_compare['code_sha256'].pop(name, None)
        for name, (current, baseline) in GT_V71_EQUIVALENT_HASHES.items():
            if current_compare['code_sha256'].get(name) == current:
                current_compare['code_sha256'][name] = baseline
        for value in (previous_compare, current_compare):
            value['v4'] = {k: v for k, v in value['v4'].items() if k != 'version'}
            value['query_syntax'] = {k: v for k, v in value.get('query_syntax', {}).items()
                                     if k != 'implementation_version'}
    if (previous['selection']['csv_sha256'] != settings['selection']['csv_sha256'] or
            previous_compare != current_compare):
        raise ValueError('Reuse source has different inference code/settings or source CSV')
    source_rows = previous['selection']['rows']
    if any(not (source / f"q{r['csv_row']:05d}/evaluation.json").exists() for r in source_rows):
        raise ValueError('Reuse source is still incomplete; wait for the 300-case run to finish')
    annotations = {r['csv_row']: r for r in source_rows}
    reused = 0
    already_cached = 0
    for row in rows:
        original = annotations.get(row['csv_row'])
        if original is None:
            continue
        if any(row.get(k) != v for k, v in original.items()):
            raise ValueError(f'Reuse annotation mismatch at CSV row {row["csv_row"]}')
        scene = row['scene']
        if previous.get('input_sha256', {}).get(scene) != settings.get('input_sha256', {}).get(scene):
            raise ValueError(f'Reuse scene input mismatch: {scene}')
        cached = source / f"q{row['csv_row']:05d}/evaluation.json"
        saved = json.loads(cached.read_text())
        check_cached_table(saved, settings['object_table'])
        if any(saved.get(k) != v for k, v in original.items()):
            raise ValueError(f'Reuse evaluation mismatch: {cached}')
        destination = output / f"q{row['csv_row']:05d}/evaluation.json"
        if destination.exists():
            already_cached += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, {**saved, **row, 'reused_from': str(cached),
                                'source_implementation_version': previous['v4']['version'],
                                'reused_evaluation_sha256': hashlib.sha256(cached.read_bytes()).hexdigest()})
        reused += 1
    write_json(output / 'reuse_verification.json', {
        'source': str(source), 'object_table': settings['object_table'],
        'source_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
        'source_code_sha256': previous['code_sha256'],
        'current_code_sha256': settings['code_sha256'],
        'compatible_settings': not legacy, 'reuse_mode': 'previous_version' if legacy else 'equivalent',
        'source_implementation_version': previous['v4']['version'],
        'current_implementation_version': settings['v4']['version'],
        'matching_rows': len(set(annotations) & {row['csv_row'] for row in rows})})
    print(f'Reuse verified: {reused} newly imported, {already_cached} already cached '
          f'from {source}', file=sys.stderr, flush=True)
    if legacy:
        print('Legacy GT results retain their v7.1 provenance; new inference uses v7.3. '
              'See results.json groups.implementation_version for separate scores.', file=sys.stderr, flush=True)
