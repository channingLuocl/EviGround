"""Frozen category hypotheses -> local grounding; evaluation never enters workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from categories_v9 import R8_VERSION, prepare_tables, sha256
from run import ROOT
from run_storage import ensure_run_writable, write_json


def save_manifest(path, settings):
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Run inputs/code/settings changed; choose a fresh --output directory')
    if not path.exists():
        write_json(path, settings)


def run_case(row, scene_root, output, top_k, review_limit):
    from validation_reporting import evaluate
    from metrics import bbox_iou_3d
    table = f'v9_top{top_k}'
    directory = output / f"q{row['csv_row']:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory/'evaluation.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        if any(result.get(k) != v for k, v in row.items()) or result['object_table'] != table:
            raise ValueError('Cached evaluation belongs to different inputs')
        return result  # Resume every outcome equally; never reuse an ever-correct cache.
    prediction, log = directory/'prediction.json', directory/'console.log'
    try:
        if not prediction.exists():
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--scene-dir', str(scene_root/row['scene']), '--query', row['query'],
                       '--output', str(directory), '--category-top-k', str(top_k),
                       '--review-limit', str(review_limit)]
            with log.open('w') as handle:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if process.returncode:
                raise ValueError(f'Worker failed ({process.returncode}); see {log}')
        result = evaluate(row, prediction, log, table)
        data = json.loads(prediction.read_text())
        result['category_candidate_recall'] = row['target_id'] in data['category_target_ids']
        result['category_candidate_count'] = len(data['category_target_ids'])
        target_binding = next((b for b in data.get('category_bindings', {}).get('targets', [])
                               if b['bbox_id'] == row['target_id']), {})
        ranks = [h['rank'] for h in target_binding.get('matched_hypotheses', [])]
        result['target_category_best_rank'] = min(ranks) if ranks else None
        # Evaluation-only geometry. This path is never supplied to the inference worker.
        boxes = {o['bbox_id']: o['bbox_3d'] for o in
                 json.loads((scene_root/row['scene']/'objects_gt.json').read_text())}
        result['iou_3d'] = bbox_iou_3d(data['bbox_3d'], boxes[row['target_id']]) if data.get('bbox_3d') else 0.
    except (ValueError, OSError, KeyError) as exc:
        from qwen_api import _redact
        run_path = directory/'run.json'
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        from v5_runtime import actual_attempts
        from v4_policy import settings as policy_settings
        result = {**row, 'object_table': table, 'implementation_version': policy_settings()['version'],
                  'api_attempts': sum(actual_attempts(c) for c in run.get('calls', [])),
                  'pipeline_latency_s': run.get('pipeline_latency_s'),
                  'call_usage': [c.get('usage') for c in run.get('calls', [])],
                  'error_stage': run.get('error_stage'),
                  'predicted_id': None, 'correct': False, 'accepted': False, 'iou_3d': 0.,
                  'error': _redact(str(exc)), 'log': str(log)}
    from visual109_policy import null_reason
    data = json.loads(prediction.read_text()) if prediction.exists() else {}
    run_path = directory / 'run.json'
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    gate = data.get('candidate_gate', run.get('candidate_gate', {}))
    effective = gate.get('effective_candidate_ids', data.get('category_target_ids', []))
    result.update(candidate_gate=gate, inference_candidate_count=len(effective),
                  effective_candidate_recall=row['target_id'] in effective,
                  null_reason=null_reason({**data, 'error': result.get('error')}),
                  inference_variant='r18-all-instances-atlas-binding-v1')
    result['visual109_resource_usage'] = data.get('active_trace', run.get('active_diagnostic', {})).get('r18_initial_resources', {})
    write_json(saved, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'local/validation/nr3d_val_full.json')
    parser.add_argument('--predictions', type=Path, default=ROOT/'local_outputs/v8_3_zero_shot_full')
    parser.add_argument('--scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--category-top-k', type=int, choices=(1, 5), default=5)
    parser.add_argument('--min-category-score', type=float,
                        help='Legacy v8.3 only (default 0.20); omit for r8')
    parser.add_argument('--max-category-gap', type=float,
                        help='Legacy v8.3 only (default 0.03); omit for r8')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--api-concurrency', type=int, default=8)
    parser.add_argument('--review-limit', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scene-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--query', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from v4_policy import v9_enabled, settings as policy_settings
    if not v9_enabled():
        parser.error('Use local/run_local_v9.sh to activate the fixed v9 policy')
    if min(args.workers, args.api_concurrency, args.review_limit, args.limit or 1) < 1 or args.limit == 0:
        parser.error('limits, workers and concurrency must be positive')
    if args.worker:
        from real_scene import run_query
        run_query(args.scene_dir, args.query, args.output, query_source='Official NR3D original utterance',
                  mode='active_diagnostic', object_table=f'v9_top{args.category_top_k}',
                  review_limit=args.review_limit)
        return
    selection = json.loads(args.manifest.read_text())
    rows = selection['rows'][:args.limit] if args.limit else selection['rows']
    if not rows or len({r['csv_row'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique query selection')
    output = args.output.resolve()
    ensure_run_writable(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['SEEGROUND_V9_CATEGORY_ROOT'] = str(output/'category_inputs')
    os.environ['SEEGROUND_API_CONCURRENCY'] = str(args.api_concurrency)
    os.environ['SEEGROUND_API_SLOT_DIR'] = str(output/'.api_slots')
    scenes = {r['scene'] for r in rows}
    for scene in scenes:
        for name in ('scene.npz', 'scene_instances.json', 'objects_gt.json'):
            if not (args.scene_root/scene/name).is_file():
                raise ValueError(f'Missing local input: {scene}/{name}')
    categories = prepare_tables(args.predictions, args.scene_root, output/'category_inputs', scenes,
                                args.category_top_k, args.min_category_score, args.max_category_gap)
    os.environ['SEEGROUND_V9_SOURCE'] = 'r8' if categories['version'] == R8_VERSION else 'v8.3'
    from qwen_api import api_config_summary
    from query_syntax import syntax_settings
    from v5_runtime import file_identity
    # Hash structural geometry and evaluation files too; author OLT is never read.
    scene_hashes = {scene: file_identity(args.scene_root/scene/name for name in
                    ('scene.npz', 'scene_instances.json', 'objects_gt.json')) for scene in sorted(scenes)}
    settings = {'version': policy_settings()['version'], 'evaluation_policy': 'fixed_predictions_no_historical_correct_reuse',
        'selection_sha256': sha256(args.manifest), 'rows': rows, 'categories': categories,
        'scene_root': str(args.scene_root.resolve()), 'scene_sha256': scene_hashes,
        'policy': policy_settings(), 'review_limit': args.review_limit, 'workers': args.workers,
        'r8_grounding_revision': os.getenv('SEEGROUND_R8_GROUNDING_REVISION', 'r1'),
        'query_syntax': syntax_settings(), 'api': api_config_summary(),
        'models': {key: os.getenv(key, '') for key in
                   ('SEEGROUND_REACT_BACKEND', 'SEEGROUND_JEV_MODEL', 'SEEGROUND_MODEL', 'SEEGROUND_BASE_URL')},
        'code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))
                        if not p.name.startswith(('test_', '._'))}}
    save_manifest(output/'manifest.json', settings)
    from batch_progress import BatchProgress
    from validation_reporting import report
    results, pending = [], []
    for row in rows:
        if (output/f"q{row['csv_row']:05d}"/'evaluation.json').exists():
            results.append(run_case(row, args.scene_root, output, args.category_top_k, args.review_limit))
        else:
            pending.append(row)
    report(output, results, len(rows))
    if args.prepare_only:
        print(f'Prepared {len(rows)} cases: {len(results)} cached, {len(pending)} pending; '
              'no inference started.', file=sys.stderr, flush=True)
        return
    progress = BatchProgress(len(rows), initial_results=results)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_case, row, args.scene_root, output, args.category_top_k, args.review_limit) for row in pending]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            if time.monotonic() - last_report >= 30:
                report(output, results, len(rows))
                last_report = time.monotonic()
            progress.update(result)
    summary = report(output, results, len(rows))
    summary['evaluation_policy'] = settings['evaluation_policy']
    summary['category_candidate_recall'] = sum(r.get('category_candidate_recall', False) for r in results)/len(rows)
    summary['category_candidate_recall_observed_queries'] = sum('category_candidate_recall' in r for r in results)
    summary['target_category_rank_groups'] = {
        name: {'queries': len(group), 'correct': sum(r['correct'] for r in group),
               'accuracy': sum(r['correct'] for r in group)/len(group) if group else None}
        for name, group in (
            ('top1_match', [r for r in results if r.get('target_category_best_rank') == 1]),
            ('top2_to_5_rescue', [r for r in results if r.get('target_category_best_rank') in (2, 3, 4, 5)]))}
    write_json(output/'results.json', {'summary': summary, 'results': sorted(results, key=lambda r: r['csv_row'])})


if __name__ == '__main__':
    main()
