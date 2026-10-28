"""Freeze a new inference version and its input files; never overwrite a version."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from verify import ROOT, digest

WRAPPERS = ('run.sh', 'verify.py', 'experiment.py', 'retry_state.py', 'scanrefer_metrics.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--max-retries', type=int, required=True)
    p.add_argument('--allow-full', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', a.name) or a.max_retries < 0:
        p.error('Use a simple lowercase version name and nonnegative retry limit')
    target = ROOT / 'versions' / (a.name + '.json')
    if target.exists():
        raise ValueError('Version already exists; do not overwrite frozen versions')
    paths = list((ROOT / 'inputs').glob('*manifest.json'))
    paths += [ROOT / 'inputs/r8_predictions' / name for name in
              ('predictions.json', 'config.json', 'EXPORT_VERIFIED.json')]
    paths += [p for p in (ROOT / 'inputs/scenes').glob('*/*')
              if p.name in ('scene.npz', 'scene_instances.json', 'objects_gt.json')]
    record = {'name': a.name, 'created_at': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'allow_full': a.allow_full, 'max_retry_rounds': a.max_retries,
              'source_sha256': {p.name: digest(p) for p in (ROOT / 'source_r18/local').glob('*.py')
                                if not p.name.startswith(('test_', '._'))},
              'wrapper_sha256': {name: digest(ROOT / name) for name in WRAPPERS},
              'input_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}}
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
