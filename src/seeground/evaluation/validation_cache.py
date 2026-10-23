"""Persistent historical-success cache, separate from inference and batch scheduling."""
import fcntl
import hashlib
import json
from pathlib import Path

from seeground.evaluation.run_storage import ensure_run_writable, write_json

def check_cached_table(saved, object_table):
    # Historical v7.1 evaluations were always GT and did not record this field.
    if saved.get('object_table', 'gt') != object_table:
        raise ValueError('Cached prediction has a different object-table source')


def correct_cache_path(scene_root, object_table):
    return Path(scene_root) / f'.correct_results_{object_table}.json'


def save_correct_cache(cache, scene_root):
    path = correct_cache_path(scene_root, cache['object_table'])
    # Merge under a lock so overlapping runs cannot lose each other's successes.
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['csv_sha256'] != cache['csv_sha256']:
                raise ValueError('Correct-result cache belongs to another dataset')
            cache['results'].update(saved['results'])
        write_json(path, cache)


def _correct_result(row, evaluation):
    return (evaluation.get('correct') is True and
            evaluation.get('predicted_id') == row.get('target_id') and
            evaluation.get('predicted_id') is not None and
            same_sample(row, evaluation))


def same_sample(row, other):
    # Selection-specific groups (size_form, easy, stage...) do not identify a query.
    return all(row.get(key) == other.get(key) for key in
               ('csv_row', 'scene', 'query', 'target_id'))


def load_correct_cache(scene_root, selection, object_table):
    """Find successful rows across earlier runs, even when inference code changed."""
    csv_hash = selection['csv_sha256']
    rows = {row['csv_row']: row for row in selection['rows']}
    cache_file = correct_cache_path(scene_root, object_table)
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        if cache.get('csv_sha256') != csv_hash or cache.get('object_table') != object_table:
            raise ValueError(f'Correct-result cache does not match this dataset/table: {cache_file}')
    else:
        cache = {'schema_version': 1, 'csv_sha256': csv_hash,
                 'object_table': object_table, 'results': {}}

    # Seed the persistent index from existing run folders, including historical versions.
    for source in sorted(Path(scene_root).iterdir()):
        manifest_path = source / 'manifest.json'
        if not source.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        source_selection = manifest.get('selection')
        if not isinstance(source_selection, dict) or not isinstance(source_selection.get('rows'), list):
            continue
        if (manifest.get('object_table', 'gt') != object_table or
                source_selection.get('csv_sha256') != csv_hash):
            continue
        source_rows = {row['csv_row']: row for row in source_selection['rows']}
        for path in source.glob('q*/**/*evaluation.json'):
            try:
                csv_row = int(path.relative_to(source).parts[0][1:])
                row = rows.get(csv_row)
                if (row is None or not same_sample(row, source_rows.get(csv_row, {}))):
                    continue
                if str(csv_row) in cache['results']:
                    continue
                evaluation = json.loads(path.read_text())
            except (OSError, ValueError, KeyError):
                continue
            if (evaluation.get('object_table', object_table) != object_table or
                    not _correct_result(row, evaluation)):
                continue
            key = str(csv_row)
            source_record = {'run': str(source),
                'evaluation_path': str(path),
                'implementation_version': (evaluation.get('implementation_version') or
                    evaluation.get('source_implementation_version') or
                    manifest.get('v4', {}).get('version', 'unknown')),
                'evaluation_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            cache['results'][key] = {'row': row, 'evaluation': evaluation,
                                     'source': source_record}
    save_correct_cache(cache, scene_root)
    return cache


def remember_correct_result(cache, scene_root, row, evaluation, source_run):
    if not _correct_result(row, evaluation):
        return False
    key = str(row['csv_row'])
    entry = cache['results'].get(key)
    if entry is not None:
        if not same_sample(row, entry['row']):
            raise ValueError(f'Correct-cache sample mismatch for CSV row {row["csv_row"]}')
        return True
    source_record = {'run': str(source_run),
        'implementation_version': (evaluation.get('implementation_version') or
                                    evaluation.get('source_implementation_version', 'unknown')),
        'evaluation_sha256': hashlib.sha256(json.dumps(
            evaluation, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    cache['results'][key] = {'row': row, 'evaluation': evaluation,
                             'source': source_record}
    save_correct_cache(cache, scene_root)
    return True


def apply_correct_cache(cache, output, rows):
    """Restore past correct answers and archive any superseded local attempt."""
    ensure_run_writable(output)
    imported, replaced = [], []
    for row in rows:
        entry = cache['results'].get(str(row['csv_row']))
        if entry is None or not _correct_result(row, entry['evaluation']):
            continue
        destination = output / f"q{row['csv_row']:05d}/evaluation.json"
        saved = None
        if destination.exists():
            saved = json.loads(destination.read_text())
            if not same_sample(row, saved):
                raise ValueError(f'Correct-cache annotation mismatch: {destination}')
            if _correct_result(row, saved):
                continue
        provenance = entry['source']
        if saved is not None:
            archive = destination.parent / 'superseded_by_correct_cache'
            archive.mkdir(exist_ok=True)
            archived = archive / 'evaluation.json'
            if archived.exists():
                archived = archive / f"evaluation_{hashlib.sha256(destination.read_bytes()).hexdigest()[:12]}.json"
            write_json(archived, saved)
            replaced.append(row['csv_row'])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            imported.append(row['csv_row'])
        evaluation = {**entry['evaluation'], **row,
            'implementation_version': (entry['evaluation'].get('implementation_version') or
                                       provenance.get('implementation_version', 'unknown')),
            'correct_cache_imported': True,
            'correct_cache_source_run': provenance['run'],
            'correct_cache_source_sha256': provenance['evaluation_sha256']}
        write_json(destination, evaluation)
    return {'imported_rows': imported, 'replaced_rows': replaced,
            'cached_correct_rows': len(cache['results'])}
