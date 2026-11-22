#!/usr/bin/env python3
"""Run Nr3D Step 3 with Qwen3-VL-32B-Instruct on SiliconFlow."""
import argparse
import getpass
import os
from pathlib import Path
import warnings


ROOT = Path(__file__).resolve().parents[1]
WORKTREES = Path.home() / '.codex/worktrees'
PYTHON = WORKTREES / 'laya-qwen-fresh-20261119/EviGround/.venv/bin/python'
ASSETS = (WORKTREES / 'qwen-direct-fresh-20261120/EviGround/outputs/qwen-direct'
          / 'step2-three-level-full-nr3d-20261121-v3/asset-index.json')
DISK = Path('/Volumes/KESU')
OUTPUT = DISK / 'EviGround/experiments/step3-launcher/qwen3-vl-32b-siliconflow'
KEY_ENV = 'SILICONFLOW_API_KEY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-rows', type=int, nargs='+', help='Run specified CSV rows. Default is all 7805.')
    parser.add_argument('--output', type=Path, help='Results directory. Use the same directory to resume.')
    parser.add_argument('--prepare-only', action='store_true', help='Check all assets and worker startup without API calls.')
    args = parser.parse_args()
    if not DISK.is_mount():
        parser.error(f'Mount {DISK} before starting the batch')
    for path in (PYTHON, ASSETS):
        if not path.is_file():
            parser.error(f'Missing local dependency: {path}')

    rows = sorted(set(args.sample_rows or []))
    scope = 'rows-' + '-'.join(map(str, rows)) if rows else 'full'
    output = (args.output or OUTPUT / scope).expanduser().resolve()
    env = dict(os.environ)
    command = [str(PYTHON), '-B', '-u', str(ROOT/'scripts/qwen_direct_batch.py'), '--siliconflow',
               '--workers', '1', '--visual-levels', '3', '--assets-file', str(ASSETS), '--output', str(output)]
    if rows:
        command.extend(['--sample-rows', *map(str, rows)])
    if args.prepare_only:
        env.pop(KEY_ENV, None)
        command.append('--prepare-only')
    else:
        key = env.get(KEY_ENV, '').strip()
        if not key:
            with warnings.catch_warnings():
                warnings.simplefilter('error', getpass.GetPassWarning)
                try:
                    key = getpass.getpass('硅基流动 API Key：').strip()
                except (EOFError, getpass.GetPassWarning):
                    parser.error('Enter the key in an interactive terminal or set SILICONFLOW_API_KEY')
                except KeyboardInterrupt:
                    return 130
        if not key:
            parser.error('SiliconFlow API Key is empty')
        env[KEY_ENV] = key
    print(f'SiliconFlow Qwen/Qwen3-VL-32B-Instruct · {output}', flush=True)
    os.chdir(ROOT)
    os.execve(PYTHON, command, env)


if __name__ == '__main__':
    raise SystemExit(main())
