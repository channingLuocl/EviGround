"""Validate zero-shot ULIP-2, then complete the two fixed supervised adaptations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from category_common_v82 import run_lock
from ulip2_classifier import CHECKPOINT_NAME


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('/root/autodl-tmp/ulip2_baseline_20261018')
TRAINING = Path('/root/autodl-tmp/ulip2_scannet_train_20261018')


def status(stage, state):
    record = {'stage': stage, 'state': state, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (EXPERIMENT/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


def run(stage, script, *args):
    status(stage, 'running')
    subprocess.run([sys.executable, str(ROOT/'local'/script), *map(str, args)], cwd=ROOT, check=True)


def main():
    checkpoint = EXPERIMENT/'weights/datasets/SFXX/ulip'/CHECKPOINT_NAME
    common = ['--checkpoint', checkpoint, '--source-dir', EXPERIMENT/'source',
              '--text-features', EXPERIMENT/'text_features_bigg_197.npz']
    with run_lock(EXPERIMENT/'experiment_lock'):
        try:
            status('cpu_tests', 'running')
            subprocess.run([sys.executable, '-m', 'unittest', 'test_ulip2', 'test_uni3d_training'],
                           cwd=ROOT/'local', check=True)
            for count, name in [(1, 'smoke_1'), (100, 'smoke_100'), (3413, 'full_3413')]:
                run(name, 'run_ulip2_categories.py', *common, '--scene-root',
                    '/root/autodl-tmp/seeground-v10/scenes', '--output', EXPERIMENT/name,
                    '--limit', 0 if count == 3413 else count)
                summary = json.loads((EXPERIMENT/name/'results.json').read_text())['summary']
                if not summary['complete'] or summary['completed'] != count or summary['errors']:
                    raise ValueError(f'Incomplete or failed {count}-instance run: {summary}')
            run('audit_zero_shot', 'audit_ulip2.py', '--zero-shot', EXPERIMENT/'full_3413')
            train = common+['--manifests', '/root/autodl-tmp/seeground-v10/manifests',
                            '--point-bank-dir', '/root/autodl-tmp/pointbert_scannet_train_20261018']
            run('training_smoke', 'train_ulip2_scannet.py', *train,
                '--output', EXPERIMENT/'training_smoke', '--smoke-only')
            run('supervised_training', 'train_ulip2_scannet.py', *train, '--output', TRAINING)
            run('audit_training', 'audit_ulip2.py', '--training', TRAINING)
            status('all_experiments', 'completed')
        except Exception:
            previous = json.loads((EXPERIMENT/'status.json').read_text())['stage']
            status(previous, 'failed')
            raise


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    main()
