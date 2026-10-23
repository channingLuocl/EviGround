"""v8.2: actual point-cloud inference, with an optional separate evaluation layer."""
from seeground.paths import source_file
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from seeground.evaluation.batch_progress import BatchProgress
from seeground.categories.category_common_v82 import (ROOT, VERSION, category_vocabulary, file_hash, instance_seed,
                                 load_cloud, normalized_category, run_lock)
from seeground.classifiers.pointcloud_classifier_v82 import POINTNEXT_SHA256, PointCloudClassifier, check_inference_platform
from seeground.evaluation.run_storage import write_json


class CategoryProgress(BatchProgress):
    def update(self, row):
        if self.tty:
            self.stderr.write('\r\033[2K')
        print(json.dumps({key: row[key] for key in
                          ('bbox_id', 'target', 'predicted_object', 'correct')}, ensure_ascii=False),
              file=self.stdout, flush=True)
        self.done += 1
        self.correct += row['correct'] is True
        self.draw()


def select_jobs(args):
    """Evaluation IDs define the test universe; category names never enter jobs."""
    if args.cloud:
        _, _, ids = load_cloud(args.cloud)
        available = sorted(int(i) for i in np.unique(ids) if i >= 0)
        selected = args.bbox_id if args.bbox_id is not None else available
        if not selected or len(set(selected)) != len(selected) or any(i not in available for i in selected):
            raise ValueError('Select distinct nonnegative instance IDs present in the cloud')
        if args.limit is not None and args.limit > 0:
            selected = selected[:args.limit]
        scene = args.cloud.parent.name if args.cloud.name == 'scene.npz' else args.cloud.stem
        return [{'scene': scene, 'bbox_id': i} for i in selected], {scene: args.cloud}, {}
    manifest = args.manifest or ROOT/'data/validation/nr3d_val_full.json'
    scenes = sorted({row['scene'] for row in json.loads(manifest.read_text())['rows']})
    jobs, clouds, labels = [], {}, {}
    for scene in scenes:
        if Path(scene).name != scene or scene in ('.', '..'):
            raise ValueError(f'Invalid scene name: {scene}')
        directory = args.scene_root/scene
        # IDs come from the measured instance table; GT names are loaded separately
        # for scoring, and never select or describe model inputs.
        instances = json.loads((directory/'objects.json').read_text())
        annotations = json.loads((directory/'objects_gt.json').read_text())
        ids = [o['bbox_id'] for o in instances]
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'Invalid evaluation instance IDs: {scene}')
        labels[scene] = {o['bbox_id']: o['target'] for o in annotations}
        if set(ids) != set(labels[scene]):
            raise ValueError(f'Instance and evaluation IDs differ: {scene}')
        jobs.extend({'scene': scene, 'bbox_id': i} for i in sorted(ids))
        clouds[scene] = directory/'scene.npz'
    # Same sorted-scene, sorted-ID, seed-42 ordering as the prior subset experiment.
    random.Random(args.seed).shuffle(jobs)
    limit = 100 if args.limit is None else args.limit
    jobs = jobs[:limit] if limit else jobs
    return jobs, clouds, labels


def evaluate(prediction, target):
    predicted = prediction['predicted_object']
    return {'scene': prediction['scene'], 'bbox_id': prediction['bbox_id'],
            'target': target, 'predicted_object': predicted, 'status': prediction['status'],
            'correct': predicted is not None and normalized_category(predicted) == normalized_category(target),
            'exact_correct': predicted == target,
            'top5_correct': any(normalized_category(o['category']) == normalized_category(target)
                                for o in prediction.get('top5', [])),
            'target_in_vocabulary': normalized_category(target) in
                                   {normalized_category(c) for c in category_vocabulary()},
            'top5': prediction.get('top5', []), 'error': prediction.get('error')}


def summarize(output, jobs, predictions, labels):
    ordered = [predictions[(j['scene'], j['bbox_id'])] for j in jobs
               if (j['scene'], j['bbox_id']) in predictions]
    summary = {'version': VERSION, 'unit': 'unique_scene_instance', 'selected': len(jobs),
               'completed': len(ordered), 'successful': sum(p['status'] == 'completed' for p in ordered),
               'errors': sum(p['status'] == 'error' for p in ordered),
               'complete': len(ordered) == len(jobs)}
    write_json(output/'predictions.json', {'summary': summary, 'results': ordered})
    if not labels:
        return summary, {}
    rows = [evaluate(p, labels[p['scene']][p['bbox_id']]) for p in ordered]
    by_class = defaultdict(list)
    for row in rows:
        by_class[normalized_category(row['target'])].append(row)
    classes = {name: {'count': len(items), 'correct': sum(r['correct'] for r in items),
                      'accuracy': sum(r['correct'] for r in items)/len(items)}
               for name, items in sorted(by_class.items())}
    for name in ('correct', 'exact_correct', 'top5_correct'):
        summary[name] = sum(r[name] for r in rows)
        summary[name.replace('correct', 'accuracy')] = summary[name]/len(rows) if rows else None
    summary.update(class_mean_accuracy=float(np.mean([c['accuracy'] for c in classes.values()]))
                   if classes else None,
                   targets_outside_vocabulary=sum(not r['target_in_vocabulary'] for r in rows),
                   matching='v8.1 lexical equivalence; no subtype merging; exact accuracy also reported',
                   evaluation_policy='fixed_predictions_with_explicit_error_retries')
    write_json(output/'results.json', {'summary': summary, 'per_class': classes, 'results': rows})
    return summary, {(r['scene'], r['bbox_id']): r for r in rows}


def code_hashes():
    files = [source_file(name) for name in
             ('run_categories_v82.py', 'pointcloud_classifier_v82.py', 'category_common_v82.py',
              'batch_progress.py', 'run_storage.py', 'prepare_v82_text_features.py')]
    files += [ROOT/'data/scannet200_constants.py']
    files += sorted(p for p in (ROOT/'third_party/models').rglob('*') if p.suffix in ('.py', '.yaml'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def print_summary(summary):
    if 'accuracy' in summary:
        print(f'v8.2: True {summary["correct"]}/{summary["completed"]} '
              f'= {summary["accuracy"] or 0:.2%}; '
              f'exact top-1 {summary["exact_accuracy"] or 0:.2%}; '
              f'top-5 {summary["top5_accuracy"] or 0:.2%}; '
              f'class mean {summary["class_mean_accuracy"] or 0:.2%}; '
              f'errors {summary["errors"]}.', file=sys.stderr)


def run(args, classifier_factory=None, extra_config=None):
    jobs, clouds, labels = select_jobs(args)
    if not jobs:
        raise ValueError('No instances selected')
    output = args.output.resolve()
    with run_lock(output):
        if (args.backend == 'pointnext' and args.checkpoint.exists()
                and file_hash(args.checkpoint) != POINTNEXT_SHA256):
            raise ValueError('PointNeXt checkpoint differs from verified release; preserving cached results')
        config = {'version': VERSION, 'backend': args.backend, 'seed': args.seed,
                  'mode': 'cloud_only' if args.cloud else 'evaluation',
                  'input_contract': 'instance_xyz_optional_rgb_only; labels_never_enter_encoder',
                  'vocabulary': category_vocabulary(), 'code_sha256': code_hashes(),
                  'checkpoint': str(args.checkpoint.resolve()) if args.backend == 'pointnext' else None,
                  'checkpoint_sha256': POINTNEXT_SHA256 if args.backend == 'pointnext' else None,
                  'text_features_sha256': file_hash(args.text_features) if args.text_features else None,
                  'selection_source': str((args.cloud or args.manifest or
                                          ROOT/'data/validation/nr3d_val_full.json').resolve()),
                  'scene_root': str(args.scene_root.resolve()) if not args.cloud else None}
        if extra_config is not None:
            config['extension'] = extra_config
        # JSON round-trip converts the vocabulary tuple to its on-disk representation.
        config = json.loads(json.dumps(config))
        config_path = output/'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Code, model or inputs changed; choose a new --output directory')
        grouped = defaultdict(list)
        for job in jobs:
            grouped[job['scene']].append(job)
        inputs_path = output/'inputs.json'
        inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
        for scene in grouped:
            value = {'cloud': str(clouds[scene].resolve()), 'cloud_sha256': file_hash(clouds[scene])}
            if labels:
                value['gt_sha256'] = file_hash(args.scene_root/scene/'objects_gt.json')
            if scene in inputs and inputs[scene] != value:
                raise ValueError(f'Scene inputs changed: {scene}; choose a new --output')
            inputs[scene] = value
        write_json(config_path, config)
        write_json(inputs_path, inputs)
        write_json(output/'selection.json', {'seed': args.seed, 'instances': jobs})
        predictions, pending = {}, defaultdict(list)
        for job in jobs:
            path = output/job['scene']/f'object_{job["bbox_id"]:04d}'/'prediction.json'
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved and not (args.retry_errors and saved['status'] == 'error'):
                predictions[(job['scene'], job['bbox_id'])] = saved
            else:
                pending[job['scene']].append(job)
        summary, rows = summarize(output, jobs, predictions, labels)
        print(f'v8.2 {args.backend}: {len(jobs)} unique objects; {len(predictions)} cached; '
              f'{len(jobs)-len(predictions)} pending. GT is evaluation-only; no author labels.', file=sys.stderr)
        if args.prepare_only:
            print(f'Prepared: {output}', file=sys.stderr)
            return 0
        if not pending:
            print_summary(summary)
            return 1 if summary['errors'] else 0
        # Fail before creating predictions if model/dependencies/weights cannot be loaded.
        classifier = (classifier_factory or PointCloudClassifier)(
            args.backend, args.checkpoint, args.text_features)
        runtime_path = output/'runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != classifier.metadata:
            raise ValueError('Runtime or model assets changed; choose a new --output')
        write_json(runtime_path, classifier.metadata)
        progress = CategoryProgress(len(jobs), initial_results=list(rows.values())) if labels else None
        interrupted = False
        try:
            for scene, scene_jobs in pending.items():
                xyz, rgb, ids = load_cloud(clouds[scene])
                for job in scene_jobs:
                    oid = job['bbox_id']
                    directory = output/scene/f'object_{oid:04d}'
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory/'prediction.json'
                    if path.exists():
                        attempt = directory/'attempts'/str(time.time_ns())
                        attempt.mkdir(parents=True)
                        for name in ('prediction.json', 'evaluation.json'):
                            old = directory/name
                            if old.exists():
                                old.rename(attempt/name)
                    seed = instance_seed(args.seed, scene, oid)
                    selected = ids == oid
                    points = xyz[selected]
                    prediction = {'version': VERSION, 'backend': args.backend, **job, 'seed': seed,
                                  'point_count': len(points)}
                    try:
                        answer = classifier.predict(points, rgb[selected] if rgb is not None else None, seed)
                        lo, hi = points.min(0), points.max(0)
                        prediction.update(answer, bbox_3d=np.concatenate([(lo+hi)/2, hi-lo]).tolist())
                    except (ValueError, RuntimeError) as exc:
                        prediction.update(predicted_object=None, status='error', error=f'{type(exc).__name__}: {exc}')
                        write_json(path, prediction)
                        predictions[(scene, oid)] = prediction
                        # CUDA failures should stop, preserving results; data errors can continue.
                        if isinstance(exc, RuntimeError):
                            raise
                    write_json(path, prediction)
                    predictions[(scene, oid)] = prediction
                    if labels:
                        row = evaluate(prediction, labels[scene][oid])
                        write_json(directory/'evaluation.json', row)
                        progress.update(row)
                    else:
                        print(json.dumps({key: prediction[key] for key in
                                          ('bbox_id', 'predicted_object', 'status')}, ensure_ascii=False), flush=True)
                        print(f'{len(predictions)}/{len(jobs)} instances saved', file=sys.stderr)
        except KeyboardInterrupt:
            interrupted = True
            print('\nStopped; saved predictions will be reused by the same command.', file=sys.stderr)
        finally:
            summary, _ = summarize(output, jobs, predictions, labels)
            print_summary(summary)
        return 1 if interrupted or summary['errors'] or not summary['complete'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cloud', type=Path, help='NPZ xyz/ids, optional RGB 0..255; no labels needed')
    mode.add_argument('--manifest', type=Path, help='Evaluation scene list; defaults to nr3d_val_full.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--bbox-id', type=int, nargs='+', help='Only valid with --cloud')
    parser.add_argument('--backend', choices=('pointnext', 'openshape'), default='pointnext')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/pnext_cls.pth')
    parser.add_argument('--text-features', type=Path, help='Offline PointNeXt category text features NPZ')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Evaluation default 100; cloud default all; 0 means all')
    parser.add_argument('--prepare-only', action='store_true', help='Select/hash inputs without loading models')
    parser.add_argument('--retry-errors', action='store_true', help='Preserve and retry errors, never wrong answers')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.bbox_id is not None and not args.cloud:
        parser.error('--bbox-id requires --cloud')
    if args.text_features and args.backend != 'pointnext':
        parser.error('--text-features is only supported by PointNeXt')
    if args.output is None:
        suffix = 'cloud' if args.cloud else 'subset'
        args.output = ROOT/f'local_outputs/v8_2_{args.backend}_{suffix}'
    try:
        if not args.prepare_only:
            check_inference_platform()
        return run(args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\nSetup and commands: local/README_V8_2.md\n')


if __name__ == '__main__':
    sys.exit(main())
