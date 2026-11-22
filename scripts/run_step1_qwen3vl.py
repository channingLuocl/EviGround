#!/usr/bin/env python3
"""Run the original direct-vision Nr3D workflow with Qwen3-VL on SiliconFlow."""
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
OUTPUT = DISK / 'EviGround/experiments/step1-launcher/qwen3-vl-32b-siliconflow-256k/full'
KEY_ENV = 'SILICONFLOW_API_KEY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Use the same directory to resume.')
    parser.add_argument('--prepare-only', action='store_true', help='Verify assets and worker startup offline.')
    args = parser.parse_args()
    if not DISK.is_mount():
        parser.error(f'Mount {DISK} before starting the batch')
    for path in (PYTHON, ASSETS):
        if not path.is_file():
            parser.error(f'Missing local dependency: {path}')

    output = args.output.expanduser().resolve()
    env = dict(os.environ)
    command = [str(PYTHON), '-B', '-u', str(ROOT/'scripts/qwen_direct_batch.py'),
               '--workers', '1', '--assets-file', str(ASSETS), '--output', str(output)]
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
    print(f'SiliconFlow Qwen/Qwen3-VL-32B-Instruct | direct vision | {output}', flush=True)
    os.chdir(ROOT)
    os.execve(PYTHON, command, env)


if __name__ == '__main__':
    raise SystemExit(main())
