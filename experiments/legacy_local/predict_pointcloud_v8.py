"""Predict instance categories from an NPZ point cloud without any label tables."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from object_classifier_v8 import VERSION, classify_instance, category_vocabulary
from qwen_api import api_config_summary, _redact, _settings
from run import ROOT
from run_object_classification import file_hash, run_lock
from run_storage import write_json


def load_cloud(path):
    with np.load(path, allow_pickle=False) as cloud:
        xyz, rgb, ids = cloud['xyz'], cloud['rgb'], cloud['ids']
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all()
            or rgb.shape != xyz.shape or not np.isfinite(rgb).all()
            or np.any(rgb < 0) or np.any(rgb > 255)
            or ids.shape != (len(xyz),) or ids.dtype.kind not in 'iu'):
        raise ValueError('Expected finite xyz[N,3], RGB 0..255 [N,3], integer instance ids[N]')
    return xyz, rgb.astype(np.uint8), ids


def predict(args):
    """Only point coordinates, colours and instance IDs are loaded for inference."""
    xyz, rgb, ids = load_cloud(args.cloud)
    available = sorted(int(i) for i in np.unique(ids) if i >= 0)
    selected = args.bbox_id if args.bbox_id is not None else available
    if not selected or len(set(selected)) != len(selected) or any(i not in available for i in selected):
        raise ValueError('Select distinct nonnegative instance IDs present in the cloud')
    output = args.output.resolve()
    with run_lock(output):
        config = {'version': VERSION, 'input_contract': 'xyz_rgb_instance_ids_only',
                  'cloud': str(args.cloud.resolve()), 'cloud_sha256': file_hash(args.cloud),
                  'model': os.getenv('SEEGROUND_MODEL', ''),
                  'api_base_url': os.getenv('SEEGROUND_BASE_URL', '').rstrip('/'),
                  'api': api_config_summary(),
                  'code_sha256': {name: file_hash(Path(__file__).parent/name) for name in
                                  ('predict_pointcloud_v8.py', 'object_classifier_v8.py',
                                   'mesh_renderer_v8.py', 'run.py', 'qwen_api.py', 'response_schemas.py',
                                   'run_storage.py', 'run_object_classification.py', 'v4_policy.py')},
                  'vocabulary': category_vocabulary()}
        path = output/'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Model, code or cloud changed; choose a new --output')
        write_json(path, config)
        write_json(output/'selection.json', {'bbox_ids': selected})
        if args.prepare_only:
            return 0
        _settings()
        stop, lock = threading.Event(), threading.Lock()
        predictions = {}
        for oid in selected:
            path = output/f'object_{oid:04d}'/'prediction.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if not (args.retry_errors and saved['status'] == 'error'):
                    predictions[oid] = saved

        def save_summary():
            write_json(output/'predictions.json', {'version': VERSION,
                       'selected': len(selected), 'completed': len(predictions),
                       'errors': sum(p['status'] == 'error' for p in predictions.values()),
                       'results': [predictions[i] for i in selected if i in predictions]})

        save_summary()
        def interrupt(signum, frame):
            stop.set()
            print('Stopping new instances; active predictions will save.', file=sys.stderr)
        previous = signal.signal(signal.SIGINT, interrupt)
        def process(oid):
            if stop.is_set():
                return
            directory = output/f'object_{oid:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory/'prediction.json').exists():
                attempt = directory/'attempts'/str(time.time_ns())
                attempt.mkdir(parents=True)
                for name in ('prediction.json', 'response.json', 'request.json'):
                    if (directory/name).exists():
                        (directory/name).rename(attempt/name)
            try:
                result = classify_instance(xyz, rgb, ids, oid, directory, category_vocabulary())
            except Exception as exc:
                diagnostic = _redact(getattr(exc, 'diagnostic', {}))
                result = {'bbox_id': oid, 'predicted_object': None, 'status': 'error',
                          'version': VERSION, 'error': _redact(str(exc)), 'diagnostic': diagnostic}
                if (diagnostic.get('http_status') not in (None, 200)
                        or diagnostic.get('failure_kind') in ('timeout', 'network')):
                    stop.set()
            write_json(directory/'prediction.json', result)
            with lock:
                predictions[oid] = result
                save_summary()
                print(json.dumps({key: result[key] for key in
                                 ('bbox_id', 'predicted_object', 'status')}, ensure_ascii=False), flush=True)
                print(f'{len(predictions)}/{len(selected)} instances saved', file=sys.stderr)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(process, [i for i in selected if i not in predictions]))
        finally:
            signal.signal(signal.SIGINT, previous)
            save_summary()
        return 1 if stop.is_set() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', required=True, type=Path, help='NPZ with xyz, rgb, ids; Z up, metres')
    parser.add_argument('--bbox-id', type=int, nargs='+', help='Default: all nonnegative instance IDs')
    parser.add_argument('--output', type=Path, default=ROOT/'local_outputs/v8_1_independent_cloud')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--api-concurrency', type=int, default=8)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.api_concurrency < 1:
        parser.error('Concurrency must be positive')
    os.environ['SEEGROUND_API_CONCURRENCY'] = str(args.api_concurrency)
    try:
        return predict(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{_redact(str(exc))}\n')


if __name__ == '__main__':
    sys.exit(main())
