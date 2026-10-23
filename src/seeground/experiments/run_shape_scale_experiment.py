"""Sequential train/select, evaluate, and independent audit for the scale study."""
from seeground.paths import source_file
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('feature-root', 'prior', 'output', 'design'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'status.json').exists():
            raise ValueError('Existing launch; inspect it before starting another run')

        def status(stage, **details):
            temporary = args.output / 'status.tmp'
            temporary.write_text(json.dumps({'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details}, indent=2) + '\n')
            temporary.replace(args.output / 'status.json')

        common = ['--feature-root', str(args.feature_root), '--prior', str(args.prior),
                  '--output', str(args.output), '--design', str(args.design), '--device', args.device]
        try:
            for stage in ('train-select', 'evaluate'):
                status(stage)
                subprocess.run([sys.executable, '-u', str(source_file('shape_scale.py')), stage, *common], check=True)
            status('auditing')
            subprocess.run([sys.executable, '-u', str(source_file('audit_shape_scale.py')),
                            '--feature-root', str(args.feature_root), '--output', str(args.output)], check=True)
            status('completed', audit_passed=True)
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
