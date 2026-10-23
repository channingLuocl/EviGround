"""Run download, strict model preparation, smoke checks and the fixed full evaluation."""
from eviground.paths import source_file
import json
import os
from pathlib import Path
import subprocess
import sys
import time


from eviground.paths import ROOT
EXPERIMENT = Path('/root/autodl-tmp/uni3d_baseline_20261018')
ASSETS = EXPERIMENT/'assets'
RUNS = EXPERIMENT/'yup'


def status(stage, state):
    record = {'stage': stage, 'state': state, 'protocol': 'scannet_zup_to_uni3d_yup',
              'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (EXPERIMENT/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


def run(stage, script, *args):
    status(stage, 'running')
    subprocess.run([sys.executable, str(source_file(script)), *map(str, args)],
                   cwd=ROOT, check=True)


def verify_output(directory, count):
    summary = json.loads((directory/'results.json').read_text())['summary']
    if not summary['complete'] or summary['completed'] != count or summary['errors']:
        raise ValueError(f'Incomplete or failed {count}-instance evaluation: {summary}')
    reference = Path('/root/autodl-tmp/v8_2_openshape_full_3413')
    expected = json.loads((reference/'selection.json').read_text())['instances'][:count]
    actual = json.loads((directory/'selection.json').read_text())['instances']
    if actual != expected:
        raise ValueError('Uni3D instance selection differs from existing OpenShape baseline')
    inputs = json.loads((directory/'inputs.json').read_text())
    old_inputs = json.loads((reference/'inputs.json').read_text())
    if any(value != old_inputs[scene] for scene, value in inputs.items()):
        raise ValueError('Input point clouds or GT changed since the OpenShape baseline')
    prediction = next(directory.glob('scene*/object_*/prediction.json'))
    scores = json.loads(prediction.read_text())['scores']
    import numpy as np
    if len(scores) != 197 or not np.isfinite(scores).all():
        raise ValueError('Expected 197 finite cosine scores')
    print(f'Verified {count} instances, identical inputs/selection, {summary["errors"]} errors', flush=True)


def main():
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    stage = 'download'
    try:
        run(stage, 'download_uni3d_weights.py', '--asset-dir', ASSETS)
        stage = 'prepare_text'
        run(stage, 'prepare_uni3d_assets.py', '--asset-dir', ASSETS)
        common = ['--checkpoint', ASSETS/'weights/BAAI/Uni3D/modelzoo/uni3d-b/model.pt',
                  '--text-features', ASSETS/'text_features_eva02e_197.npz',
                  '--source-dir', ASSETS/'source',
                  '--scene-root', '/root/autodl-tmp/seeground-v10/scenes']
        for count, name in [(1, 'smoke_1'), (100, 'smoke_100'), (3413, 'full_3413')]:
            stage = name
            output = RUNS/name
            run(stage, 'run_uni3d_categories.py', *common, '--output', output,
                '--limit', 0 if count == 3413 else count)
            verify_output(output, count)
        status('full_3413', 'completed')
    except Exception:
        status(stage, 'failed')
        raise


if __name__ == '__main__':
    main()
