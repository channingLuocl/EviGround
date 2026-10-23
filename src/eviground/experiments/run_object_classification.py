"""Evaluate v8 object categories on a reproducible subset of unique instances."""
from eviground.paths import source_file
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time

import numpy as np

from eviground.evaluation.batch_progress import BatchProgress
from eviground.classifiers.object_classifier_v8 import VERSION, category_vocabulary, classify_instance, normalized_category
from eviground.api.qwen_api import api_config_summary, _redact, _settings
from eviground.rendering.run import ROOT
from eviground.evaluation.run_storage import ensure_run_writable, write_json


class ObjectProgress(BatchProgress):
    def update(self, result):
        if self.tty:
            self.stderr.write('\r\033[2K')
            self.stderr.flush()
        print(json.dumps({key: result[key] for key in
                          ('bbox_id', 'target', 'predicted_object', 'correct')}, ensure_ascii=False),
              file=self.stdout, flush=True)
        self.done += 1
        self.correct += result['correct'] is True
        self.draw()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_instances(scene_root, manifest, limit, seed):
    scenes = sorted({row['scene'] for row in json.loads(manifest.read_text())['rows']})
    instances = []
    for scene in scenes:
        # Author labels are not copied into inference jobs.
        objects = json.loads((scene_root / scene / 'objects.json').read_text())
        seen = set()
        for item in sorted(objects, key=lambda o: o['bbox_id']):
            oid = item['bbox_id']
            if type(oid) is not int or oid < 0 or oid in seen:
                raise ValueError(f'Invalid or duplicate instance ID in {scene}')
            seen.add(oid)
            instances.append({'scene': scene, 'bbox_id': oid, 'bbox_3d': item['bbox_3d']})
    random.Random(seed).shuffle(instances)
    return instances[:limit] if limit else instances


def evaluate_prediction(job, prediction, gt_label, author_label):
    predicted = prediction['predicted_object']
    row = {'scene': job['scene'], 'bbox_id': job['bbox_id'], 'target': gt_label,
            'predicted_object': predicted,
            'correct': predicted is not None and normalized_category(predicted) == normalized_category(gt_label),
            'exact_correct': predicted == gt_label,
            'author_predicted_object': author_label,
            'author_correct': normalized_category(author_label) == normalized_category(gt_label),
            'status': prediction['status'], 'error': prediction.get('error'),
            'alternatives': prediction.get('alternatives', [])}
    return row


def report(output, jobs, evaluations):
    rows = [evaluations[(job['scene'], job['bbox_id'])] for job in jobs
            if (job['scene'], job['bbox_id']) in evaluations]
    n = len(rows)
    summary = {'version': VERSION, 'method': 'independent_pointcloud_vlm', 'unit': 'unique_scene_instance', 'selected': len(jobs), 'completed': n,
               'correct': sum(r['correct'] for r in rows),
               'exact_correct': sum(r['exact_correct'] for r in rows),
               'author_correct': sum(r['author_correct'] for r in rows),
               'errors': sum(r['status'] == 'error' for r in rows),
               'unrecognised': sum(r['predicted_object'] is None and r['status'] != 'error' for r in rows),
               'evaluation_policy': 'fixed_predictions_with_explicit_error_retries',
               'matching': 'case/spacing/hyphen, explicit plurals and sofa/television/trashcan aliases; no subtype merging'}
    for field in ('correct', 'exact_correct', 'author_correct'):
        summary[field.replace('correct', 'accuracy')] = summary[field]/n if n else None
    write_json(output / 'results.json', {'summary': summary, 'results': rows})
    return summary


@contextmanager
def run_lock(output):
    ensure_run_writable(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This classification output is already running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args):
    output, scene_root = args.output.resolve(), args.scene_root.resolve()
    with run_lock(output):
        jobs = select_instances(scene_root, args.manifest, args.limit, args.seed)
        if not jobs:
            raise ValueError('No object instances selected')
        vocabulary = category_vocabulary()
        api = api_config_summary()
        sources = ['object_classifier_v8.py', 'run_object_classification.py', 'run.py', 'qwen_api.py',
                   'run_storage.py', 'response_schemas.py', 'batch_progress.py', 'v4_policy.py',
                   'mesh_renderer_v8.py']
        identity = {'version': VERSION, 'method': 'independent_pointcloud_vlm',
                    'input_contract': 'xyz_rgb_instance_ids_only', 'model': os.getenv('EVIGROUND_MODEL', ''),
                    'api_base_url': os.getenv('EVIGROUND_BASE_URL', '').rstrip('/'),
                    'provider': api['provider'],
                    'request_options': {**api['request_options'], 'response_format': {
                        'type': 'json_schema' if api['provider'] == 'openai' else 'json_object'}},
                    'output_contract': ('strict_json_schema' if api['provider'] == 'openai'
                                        else 'json_object'),
                    'seed': args.seed, 'scene_root': str(scene_root), 'vocabulary': vocabulary,
                    'vocabulary_sha256': file_hash(ROOT/'data/scannet200_constants.py'),
                    'manifest_sha256': file_hash(args.manifest),
                    'code_sha256': {name: file_hash(source_file(name)) for name in sources}}
        path = output / 'config.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Code, model or inputs changed; use a new --output directory')
        write_json(path, identity)
        grouped = defaultdict(list)
        for job in jobs:
            grouped[job['scene']].append(job)
        old_inputs = json.loads((output/'inputs.json').read_text()) if (output/'inputs.json').exists() else {}
        inputs, labels, authors, scene_counts = dict(old_inputs), {}, {}, {}
        for scene in grouped:
            directory = scene_root/scene
            hashes = {name: file_hash(directory/name) for name in
                      ('scene.npz', 'objects.json', 'objects_gt.json')}
            if scene in old_inputs and hashes != old_inputs[scene]:
                raise ValueError(f'Scene inputs changed: {scene}; choose a new --output')
            inputs[scene] = hashes
            labels[scene] = {o['bbox_id']: o['target'] for o in json.loads((directory/'objects_gt.json').read_text())}
            authors[scene] = {o['bbox_id']: o['target'] for o in json.loads((directory/'objects.json').read_text())}
            scene_counts[scene] = len(authors[scene])
            if any(job['bbox_id'] not in labels[scene] for job in grouped[scene]):
                raise ValueError(f'Missing evaluation label in {scene}')
        write_json(output/'inputs.json', inputs)
        write_json(output/'selection.json', {'seed': args.seed, 'unit': 'unique_scene_instance', 'instances': jobs})
        evaluations, pending = {}, defaultdict(list)
        for job in jobs:
            directory = output/job['scene']/f'object_{job["bbox_id"]:04d}'
            prediction_path = directory/'prediction.json'
            prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
            if prediction and not (args.retry_errors and prediction['status'] == 'error'):
                row = evaluate_prediction(job, prediction, labels[job['scene']][job['bbox_id']],
                                          authors[job['scene']][job['bbox_id']])
                evaluations[(job['scene'], job['bbox_id'])] = row
            else:
                pending[job['scene']].append(job)
        report(output, jobs, evaluations)
        print(f'v8 categories: {len(jobs)} unique objects; {len(evaluations)} cached; '
              f'{len(jobs)-len(evaluations)} pending. Independent point-cloud recognition; GT/author labels are evaluation-only.', file=sys.stderr)
        if args.prepare_only:
            return
        if not identity['model'] or not os.getenv('EVIGROUND_API_KEY'):
            raise ValueError('Configure EVIGROUND_MODEL and EVIGROUND_API_KEY in .env')
        _settings()  # Validate the endpoint before scheduling any model requests.
        stop, lock = threading.Event(), threading.Lock()
        progress = ObjectProgress(len(jobs), initial_results=list(evaluations.values()))
        def interrupt(signum, frame):
            stop.set()
            print('\nStopping new objects; active requests will save. Resume with the same command.', file=sys.stderr)
        previous_handler = signal.signal(signal.SIGINT, interrupt)

        def finish(job, prediction, directory):
            row = evaluate_prediction(job, prediction, labels[job['scene']][job['bbox_id']],
                                      authors[job['scene']][job['bbox_id']])
            write_json(directory/'evaluation.json', row)
            with lock:
                evaluations[(job['scene'], job['bbox_id'])] = row
                report(output, jobs, evaluations)
                progress.update(row)
                if row['status'] == 'error':
                    print(f'{job["scene"]}/{job["bbox_id"]}: {row["error"]}', file=sys.stderr)

        def process_scene(scene, scene_jobs):
            if stop.is_set():
                return
            try:
                with np.load(scene_root/scene/'scene.npz') as cloud:
                    xyz, rgb, ids = cloud['xyz'], cloud['rgb'], cloud['ids']
            except Exception:
                stop.set()
                raise
            for job in scene_jobs:
                if stop.is_set():
                    break
                directory = output/scene/f'object_{job["bbox_id"]:04d}'
                directory.mkdir(parents=True, exist_ok=True)
                # Retry errors only when requested, preserving every previous attempt.
                if (directory/'prediction.json').exists():
                    attempt = directory/'attempts'/str(time.time_ns())
                    attempt.mkdir(parents=True)
                    for name in ('prediction.json', 'evaluation.json', 'response.json', 'request.json'):
                        if (directory/name).exists():
                            (directory/name).rename(attempt/name)
                try:
                    prediction = classify_instance(xyz, rgb, ids, job['bbox_id'], directory, vocabulary)
                except Exception as exc:
                    diagnostic = _redact(getattr(exc, 'diagnostic', {}))
                    prediction = {'bbox_id': job['bbox_id'], 'predicted_object': None, 'status': 'error',
                                  'version': VERSION, 'error': _redact(str(exc)), 'diagnostic': diagnostic}
                diagnostic = prediction.get('diagnostic', {})
                if prediction['status'] == 'error' and (
                        diagnostic.get('http_status') not in (None, 200)
                        or diagnostic.get('failure_kind') in ('timeout', 'network')):
                    stop.set()
                write_json(directory/'prediction.json', prediction)
                finish(job, prediction, directory)

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_scene, scene, scene_jobs) for scene, scene_jobs in pending.items()]
                for future in as_completed(futures):
                    future.result()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            report(output, jobs, evaluations)
        # Partial selections remain explicitly partial; record the prediction method.
        for scene, scene_jobs in grouped.items():
            table = [{'bbox_id': job['bbox_id'], 'target': evaluations[(scene, job['bbox_id'])]['predicted_object'],
                      'bbox_3d': job['bbox_3d']} for job in scene_jobs
                     if (scene, job['bbox_id']) in evaluations]
            complete = (len(table) == scene_counts[scene] and all(o['target'] is not None for o in table)
                        and all(evaluations[(scene, job['bbox_id'])]['status'] == 'completed'
                                for job in scene_jobs if (scene, job['bbox_id']) in evaluations))
            (output/scene).mkdir(parents=True, exist_ok=True)
            write_json(output/scene/'predicted_objects.json', {'method': 'independent_pointcloud_vlm', 'complete': complete, 'objects': table})
        if stop.is_set():
            print('Run stopped; completed predictions are saved. Use --retry-errors to retry API errors.', file=sys.stderr)
        return 1 if stop.is_set() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/validation/nr3d_val_full.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--output', type=Path, default=ROOT/'local_outputs/v8_1_independent_subset')
    parser.add_argument('--limit', type=int, default=100, help='Unique objects; 0 selects all instances')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--api-concurrency', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or args.api_concurrency < 1:
        parser.error('limit must be nonnegative; workers and api-concurrency must be positive')
    os.environ['EVIGROUND_API_CONCURRENCY'] = str(args.api_concurrency)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{_redact(str(exc))}\n')


if __name__ == '__main__':
    sys.exit(main())
