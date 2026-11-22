#!/usr/bin/env python3
"""Run the GLM Step3 evaluation through BigModel's China API from this Mac."""

import argparse
import getpass
import os
from pathlib import Path
import sys
import warnings


WORKTREES = Path.home() / ".codex/worktrees"
WORKTREE = Path(__file__).resolve().parents[1]
PYTHON = WORKTREES / "laya-qwen-fresh-20261119/EviGround/.venv/bin/python"
ASSETS = (WORKTREES / "qwen-direct-fresh-20261120/EviGround/outputs/qwen-direct"
          / "step2-three-level-full-nr3d-20261121-v3/asset-index.json")
DISK = Path("/Volumes/KESU")
OUTPUT = DISK / "EviGround/experiments/step3-launcher/glm-4.5v-bigmodel"


def main():
    parser = argparse.ArgumentParser(description="使用智谱国内标准付费 API 运行 GLM-4.5V NR3D Step3。")
    parser.add_argument("--sample-rows", type=int, nargs="+", help="指定 CSV 行号。默认全量。")
    parser.add_argument("--output", type=Path, help="指定独立结果目录。续跑沿用同一路径。")
    parser.add_argument("--reuse-results-from", type=Path, help="全量运行复用已完成的 BigModel 子集结果。")
    parser.add_argument("--reuse-correct-from", type=Path, help="只复用旧全量运行中判对的样本，其余用新版重跑。")
    parser.add_argument("--continue-from", type=Path, help="保留旧全量运行已完成的结果，未完成样本用新版继续。")
    parser.add_argument("--resume-frozen", action="store_true", help="用结果目录中的原版代码续跑。")
    args = parser.parse_args()
    if args.sample_rows and args.reuse_results_from:
        parser.error("--reuse-results-from 仅用于全量运行")
    if args.reuse_correct_from and (args.sample_rows or args.reuse_results_from or args.resume_frozen):
        parser.error("--reuse-correct-from 仅用于独立的新版全量运行")
    if args.continue_from and (args.sample_rows or args.reuse_results_from or args.reuse_correct_from or args.resume_frozen):
        parser.error("--continue-from 仅用于独立的新版全量运行")
    runner = WORKTREE / "scripts/qwen_direct_batch.py"
    if not DISK.is_mount():
        parser.error(f"请先挂载 {DISK}")
    for path in (runner, PYTHON, ASSETS):
        if not path.is_file():
            parser.error(f"缺少本地文件 {path}")

    rows = sorted(set(args.sample_rows or []))
    scope = "rows-" + "-".join(map(str, rows)) if rows else "full"
    output = (args.output or OUTPUT / scope).expanduser().resolve()
    old_output = args.continue_from or args.reuse_correct_from
    if old_output and output == old_output.expanduser().resolve():
        parser.error("--output 必须与旧结果目录不同")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            key = getpass.getpass("智谱 BigModel 标准 API Key：").strip()
        except (EOFError, getpass.GetPassWarning):
            parser.error("请在交互式 Mac 终端输入 API Key")
        except KeyboardInterrupt:
            return 130
    if not key:
        parser.error("API Key 为空")

    command = [str(PYTHON), "-B", "-u", str(runner), "--bigmodel", "--workers", "1",
               "--visual-levels", "3", "--assets-file", str(ASSETS), "--output", str(output)]
    if rows:
        command += ["--sample-rows", *map(str, rows)]
    if args.reuse_results_from:
        command += ["--reuse-results-from", str(args.reuse_results_from.expanduser().resolve())]
    if args.reuse_correct_from:
        command += ["--reuse-correct-from", str(args.reuse_correct_from.expanduser().resolve())]
    if args.continue_from:
        command += ["--continue-from", str(args.continue_from.expanduser().resolve())]
    if args.resume_frozen:
        command += ["--resume-frozen"]
    print(f"BigModel glm-4.5v（标准 API 按 Token 计费），结果目录 {output}", flush=True)
    os.chdir(WORKTREE)
    os.execve(PYTHON, command, dict(os.environ, EVIGROUND_BIGMODEL_API_KEY=key))


if __name__ == "__main__":
    sys.exit(main())
