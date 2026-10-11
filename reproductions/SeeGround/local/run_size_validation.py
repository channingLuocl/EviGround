"""Run a fixed NR3D manifest in isolated processes; answers are evaluation-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from batch_nr3d import summarize
from run import ROOT


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_source(manifest, source):
    expected = manifest['csv_sha256']
    if not source.exists():
        return {'path': str(source), 'expected_sha256': expected, 'status': 'missing'}
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('NR3D CSV hash does not match the selection')
    return {'path': str(source), 'expected_sha256': expected,
            'actual_sha256': actual, 'status': 'verified'}


def evaluate(row, prediction, log):
    data = json.loads(prediction.read_text())
    if (data['query'] != row['query'] or data.get('object_table') != 'gt' or
            data.get('mode') != 'active_diagnostic'):
        raise ValueError(f'Prediction provenance mismatch: {prediction}')
    trace = data.get('active_trace', {})
    return {**row, 'predicted_id': data.get('predicted_id'),
            'prediction_source': data.get('prediction_source'),
            'correct': data.get('predicted_id') == row['target_id'],
            'accepted': data.get('accepted'), 'decision_status': data.get('decision_status'),
            'missing': trace.get('final_evidence_state', {}).get('missing', []),
            'stop_reason': trace.get('stop_reason'), 'api_calls': data.get('api_calls'),
            'pipeline_latency_s': data.get('pipeline_latency_s'),
            'prediction_path': str(prediction), 'log': str(log)}


def run_case(row, scene_root, output, review_limit, retry_errors=False):
    directory = output / f"q{row['csv_row']:05d}"
    result_path = directory / 'evaluation.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if any(saved.get(key) != value for key, value in row.items()):
            raise ValueError(f'Cached annotation mismatch: {result_path}')
        if not (retry_errors and 'error' in saved):
            return saved
    directory.mkdir(parents=True, exist_ok=True)
    prediction = directory / 'prediction.json'
    log = directory / 'console.log'
    try:
        if not prediction.exists():
            # Target/stimulus IDs are deliberately absent from the inference process arguments.
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--scene-dir', str(scene_root / row['scene']), '--query', row['query'],
                       '--csv-row', str(row['csv_row']), '--output', str(directory),
                       '--review-limit', str(review_limit)]
            with log.open('w') as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise ValueError(f'Inference process exit code {completed.returncode}; see {log}')
        result = evaluate(row, prediction, log)
    except (OSError, ValueError, KeyError) as exc:
        result = {**row, 'correct': False, 'accepted': False, 'error': str(exc),
                  'prediction_path': str(prediction), 'log': str(log)}
    write_json(result_path, result)
    return result


def report(output, results, total):
    summary = {**summarize(results), 'selected_total': total,
               'errors': sum('error' in row for row in results),
               'accepted_wrong': sum(row.get('accepted') is True and not row['correct']
                                     for row in results)}
    groups = {}
    for key in ('stage', 'size_form'):
        groups[key] = {value: summarize([r for r in results if r.get(key) == value])
                       for value in sorted({r.get(key, 'unknown') for r in results})}
    write_json(output / 'results.json', {'summary': summary, 'groups': groups,
               'results': sorted(results, key=lambda r: r['csv_row'])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'local/validation/nr3d_size_300.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--review-limit', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Run the first N; omit to resume the whole selection')
    parser.add_argument('--retry-errors', action='store_true', help='Retry cached failed API/process runs')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scene-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--query', help=argparse.SUPPRESS)
    parser.add_argument('--csv-row', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.review_limit < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers, review-limit and limit must be positive')
    if args.worker:
        from real_scene import run_query
        run_query(args.scene_dir, args.query, args.output,
                  query_source=f'Official NR3D original utterance; CSV row {args.csv_row}',
                  mode='active_diagnostic', object_table='gt', review_limit=args.review_limit)
        return
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['rows']
    selected = rows[:args.limit] if args.limit else rows
    missing = [str(args.scene_root / scene / name)
               for scene in sorted({r['scene'] for r in selected})
               for name in ('scene.npz', 'objects_gt.json', 'scene_instances.json', 'provenance.json')
               if not (args.scene_root / scene / name).exists()]
    if missing:
        raise ValueError('Prepare scenes before inference. Missing files: ' + ', '.join(missing[:10]))
    for scene in {r['scene'] for r in selected}:
        objects = json.loads((args.scene_root / scene / 'objects_gt.json').read_text())
        known = {obj['bbox_id'] for obj in objects}
        if any(not set(r['stimulus_candidate_ids']).issubset(known)
               for r in selected if r['scene'] == scene):
            raise ValueError(f'NR3D candidate IDs missing in {scene}')
    source = ROOT / 'local_outputs/downloads/nr3d_official.csv'
    from qwen_api import api_config_summary
    from query_syntax import syntax_settings
    source_check = verify_source(manifest, source)
    if source_check['status'] == 'missing':
        print(f'Original NR3D CSV not found at {source}; using the manifest hash for preflight',
              flush=True)
    else:
        print(f'Verified original NR3D CSV: {source}', flush=True)
    output = (args.output or ROOT / 'local_outputs' /
              ('nr3d_size_300_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    settings = {'selection': manifest, 'review_limit': args.review_limit,
        'query_syntax': syntax_settings(),
        'api_config': api_config_summary(),
        'scene_root': str(args.scene_root.resolve()), 'object_table': 'gt',
        'mode': 'active_diagnostic',
        'models': {key: os.getenv(key, '') for key in (
            'SEEGROUND_REACT_BACKEND', 'SEEGROUND_JEV_MODEL', 'SEEGROUND_MODEL',
            'SEEGROUND_BASE_URL')},
        'code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / 'local').glob('*.py'))
                        if not path.name.startswith('test_')}}
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / 'manifest.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Output has different code/settings/selection; choose a fresh output directory')
    write_json(output / 'csv_verification.json', source_check)
    write_json(settings_path, settings)
    results = []
    print(f'Running {len(selected)}/{len(rows)} selected cases; workers={args.workers}; {output}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, row, args.scene_root.resolve(), output, args.review_limit,
                               args.retry_errors)
                   for row in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            report(output, results, len(rows))
            print(json.dumps({key: result.get(key) for key in (
                'csv_row', 'scene', 'predicted_id', 'prediction_source', 'target_id', 'correct', 'accepted',
                'decision_status', 'missing', 'error')}, ensure_ascii=False), flush=True)
    print(json.dumps(report(output, results, len(rows)), ensure_ascii=False, indent=2))
    print(f'Results: {output / "results.json"}')


if __name__ == '__main__':
    main()
