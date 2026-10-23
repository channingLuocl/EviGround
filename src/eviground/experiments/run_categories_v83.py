"""v8.3 training-free classification by default; supervised heads are explicit legacy experiments."""
from eviground.paths import source_file
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from eviground.categories.category_common_v82 import ROOT, category_vocabulary, file_hash, instance_seed, load_cloud, run_lock
from eviground.classifiers.evidence_data_v83 import VERSION, check_disjoint
from eviground.classifiers.pointcloud_classifier_v83 import PointCloudClassifierV83, load_head
from eviground.classifiers.pointcloud_classifier_v82 import POINTNEXT_SHA256, load_text_features
from eviground.classifiers.pointcloud_zero_shot_v83 import VERSION as ZERO_SHOT_VERSION, ZeroShotClassifierV83
from eviground.experiments.run_categories_v82 import CategoryProgress, code_hashes, evaluate, select_jobs
from eviground.evaluation.run_storage import write_json


def summarize(output, jobs, predictions, labels, development, method):
    ordered = [predictions[(r['scene'], r['bbox_id'])] for r in jobs
               if (r['scene'], r['bbox_id']) in predictions]
    zero_shot = method != 'supervised-experimental'
    scope = 'no_task_specific_training' if zero_shot else ('development_overlap' if development else
                                                        'scene_disjoint_from_head_training_and_dev')
    summary = {'version': ZERO_SHOT_VERSION if zero_shot else VERSION,
               'method': method, 'selected': len(jobs), 'completed': len(ordered),
               'errors': sum(r['status'] == 'error' for r in ordered),
               'complete': len(ordered) == len(jobs),
               'evaluation_scope': scope,
               'pretraining_overlap': 'not_audited; no strict_zero_shot_claim'}
    write_json(output/'predictions.json', {'summary': summary, 'results': ordered})
    rows = [evaluate(r, labels[r['scene']][r['bbox_id']]) for r in ordered] if labels else []
    if labels:
        classes = defaultdict(list)
        for row in rows:
            classes[row['target']].append(row['correct'])
        # Match the baseline's normalized-class averaging.
        from eviground.categories.category_common_v82 import normalized_category
        normalized = defaultdict(list)
        for name, flags in classes.items():
            normalized[normalized_category(name)].extend(flags)
        for key in ('correct', 'exact_correct', 'top5_correct'):
            summary[key] = sum(r[key] for r in rows)
            summary[key.replace('correct', 'accuracy')] = summary[key]/len(rows) if rows else None
        summary['class_mean_accuracy'] = float(np.mean([np.mean(v) for v in normalized.values()])) if rows else None
        summary['targets_outside_vocabulary'] = sum(not r['target_in_vocabulary'] for r in rows)
        write_json(output/'results.json', {'summary': summary, 'results': rows,
                   'per_class': {k: {'count': len(v), 'correct': sum(v), 'accuracy': float(np.mean(v))}
                                 for k, v in sorted(normalized.items())}})
    return summary, rows


def run(args):
    jobs, clouds, labels = select_jobs(args)
    if not jobs:
        raise ValueError('No instances selected')
    method = args.method
    zero_shot = method != 'supervised-experimental'
    if zero_shot:
        if args.head_checkpoint:
            raise ValueError('Zero-shot inference does not use a trained head')
        if file_hash(args.checkpoint) != POINTNEXT_SHA256:
            raise ValueError('Frozen PointNeXt checkpoint differs from verified release')
        load_text_features(args.text_features)
        hashes = {**code_hashes(), **{name: file_hash(source_file(name)) for name in
                  ('run_categories_v83.py', 'pointcloud_zero_shot_v83.py',
                   'pointcloud_classifier_v83.py', 'evidence_data_v83.py')}}
        record = {'mode': method}
    else:
        from eviground.experiments.train_categories_v83 import training_code_hashes
        _, record = load_head(args.head_checkpoint)  # Never silently use random weights.
        hashes = training_code_hashes()
        if record['code_sha256'] != hashes:
            raise ValueError('v8.3 model/training code changed since checkpoint creation')
        hashes = {**hashes, 'run_categories_v83.py': file_hash(__file__)}
    selected_scenes = {r['scene'] for r in jobs}
    sources = [{'scene': scene, 'cloud_sha256': file_hash(clouds[scene])} for scene in sorted(selected_scenes)]
    overlap = False
    if not zero_shot:
        try:
            check_disjoint(record['train_sources'] + record['dev_sources'], sources)
        except ValueError:
            overlap = True
            if not args.allow_development_evaluation:
                raise ValueError('Selected scenes overlap head training/dev; use independent scenes, '
                                 'or --allow-development-evaluation for a clearly marked diagnostic only')
    version = ZERO_SHOT_VERSION if zero_shot else VERSION
    config = {'version': version, 'method': method, 'mode': record['mode'], 'seed': args.seed,
              'head_sha256': file_hash(args.head_checkpoint) if not zero_shot else None,
              'learning_regime': 'training_free_no_task_labels' if zero_shot else 'supervised_adaptation',
              'checkpoint_sha256': file_hash(args.checkpoint),
              'category_text_sha256': file_hash(args.text_features),
              'code_sha256': hashes,
              'vocabulary': list(category_vocabulary()), 'jobs': jobs, 'sources': sources,
              'development_overlap': overlap,
              'gt_sha256': {scene: file_hash(args.scene_root/scene/'objects_gt.json')
                            for scene in sorted(selected_scenes)} if labels else {},
              'input_contract': 'xyz_and_instance_membership_only; no_author_labels_or_test_GT'}
    output = args.output.resolve()
    with run_lock(output):
        path = output/'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Model, code, selection or inputs changed; choose a new --output')
        write_json(path, config)
        write_json(output/'selection.json', {'seed': args.seed, 'instances': jobs})
        predictions, pending = {}, []
        for job in jobs:
            path = output/job['scene']/f'object_{job["bbox_id"]:04d}'/'prediction.json'
            cached = json.loads(path.read_text()) if path.exists() else None
            if cached and not (args.retry_errors and cached['status'] == 'error'):
                predictions[(job['scene'], job['bbox_id'])] = cached
            else:
                pending.append(job)
        summary, rows = summarize(output, jobs, predictions, labels, overlap, method)
        print(f'v8.3 {record["mode"]}: {len(jobs)} objects; {len(predictions)} cached; '
              f'{len(pending)} pending. Test GT is evaluation-only; no author labels.', file=sys.stderr)
        if args.prepare_only:
            return 0
        classifier = None
        if pending:
            classifier = (ZeroShotClassifierV83(args.checkpoint, args.text_features,
                                               global_only=method == 'zero-shot-global') if zero_shot else
                          PointCloudClassifierV83(args.checkpoint, args.text_features, args.head_checkpoint))
            runtime = output/'runtime.json'
            if runtime.exists() and json.loads(runtime.read_text()) != classifier.metadata:
                raise ValueError('Runtime changed; choose a new --output')
            write_json(runtime, classifier.metadata)
        progress = CategoryProgress(len(jobs), initial_results=rows) if labels else None
        grouped = defaultdict(list)
        for job in pending:
            grouped[job['scene']].append(job)
        interrupted = False
        try:
            for scene, scene_jobs in grouped.items():
                xyz, _, ids = load_cloud(clouds[scene])
                for job in scene_jobs:
                    oid = job['bbox_id']
                    directory = output/scene/f'object_{oid:04d}'
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory/'prediction.json'
                    if path.exists():
                        attempt = directory/'attempts'/str(time.time_ns())
                        attempt.mkdir(parents=True)
                        for name in ('prediction.json', 'evaluation.json'):
                            if (directory/name).exists():
                                (directory/name).rename(attempt/name)
                    seed = instance_seed(args.seed, scene, oid)
                    points = xyz[ids == oid]
                    prediction = {'version': version, 'method': method, **job, 'seed': seed, 'point_count': len(points)}
                    try:
                        # Do not pass GT, object metadata or annotations to the classifier.
                        prediction.update(classifier.predict(points, None, seed))
                        lo, hi = points.min(0), points.max(0)
                        prediction['bbox_3d'] = np.r_[(lo+hi)/2, hi-lo].tolist()
                    except (ValueError, RuntimeError) as exc:
                        prediction.update(status='error', predicted_object=None, error=f'{type(exc).__name__}: {exc}')
                        if isinstance(exc, RuntimeError):
                            write_json(path, prediction)
                            predictions[(scene, oid)] = prediction
                            raise
                    write_json(path, prediction)
                    predictions[(scene, oid)] = prediction
                    if labels:
                        row = evaluate(prediction, labels[scene][oid])
                        write_json(directory/'evaluation.json', row)
                        progress.update(row)
                    else:
                        print(json.dumps({k: prediction[k] for k in
                                          ('bbox_id', 'predicted_object', 'status')}), flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print('Stopped; saved predictions can be resumed.', file=sys.stderr)
        finally:
            summary, _ = summarize(output, jobs, predictions, labels, overlap, method)
            print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        return int(interrupted or summary['errors'] or not summary['complete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cloud', type=Path)
    mode.add_argument('--manifest', type=Path, help='Evaluation scene list, v8.2 format')
    parser.add_argument('--scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--bbox-id', type=int, nargs='+')
    parser.add_argument('--method', choices=('zero-shot', 'zero-shot-global', 'supervised-experimental'),
                        default='zero-shot')
    parser.add_argument('--head-checkpoint', type=Path, help='Only for --method supervised-experimental')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/pnext_cls.pth')
    parser.add_argument('--text-features', type=Path, default=ROOT/'weights/v82_clip_text_features.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'local_outputs/v8_3_zero_shot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--allow-development-evaluation', action='store_true')
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 0) or (args.bbox_id is not None and not args.cloud):
        parser.error('Nonnegative --limit required; --bbox-id requires --cloud')
    if (args.method == 'supervised-experimental') != (args.head_checkpoint is not None):
        parser.error('--head-checkpoint is required only with --method supervised-experimental')
    try:
        return run(args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\nSee local/README_V8_3.md\n')


if __name__ == '__main__':
    sys.exit(main())
