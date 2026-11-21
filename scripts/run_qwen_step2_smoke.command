#!/bin/bash
set -euo pipefail
export PYTHON_DOTENV_DISABLED=1
exec /Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python -u /Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/qwen_direct_batch.py --assets-file /Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/data/qwen_direct/full-assets.json --output /Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/outputs/qwen-direct/step2-smoke-20261120 --sample-rows 171 --workers 1 "$@"
