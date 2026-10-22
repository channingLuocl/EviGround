#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/outputs/r18_qwen_original_resume_20261022/runtime_original"
set -a
source /Users/luocl/.config/seeground/qwen-token-plan-20261022.env
set +a
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0 SEEGROUND_V9_SOURCE=r8
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_SYNTAX_BACKEND=spacy
export SEEGROUND_API_PROVIDER=openai SEEGROUND_REACT_BACKEND=openai
export SEEGROUND_MODEL=qwen3.7-plus SEEGROUND_JEV_MODEL=""
export SEEGROUND_BASE_URL=https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1
export SEEGROUND_ENABLE_THINKING=1 SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_OPENAI_TOKEN_PARAMETER=max_completion_tokens SEEGROUND_OPENAI_TEMPERATURE=0
export SEEGROUND_ALLOW_HTTP=1 SEEGROUND_CONTEXT_LIMIT=131072
export PYTHONPATH=local PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "$ROOT/reproductions/SeeGround/.venv/bin/python" -u local/run_grounding_v9.py \
  --manifest "$ROOT/reproductions/SeeGround/local/validation/nr3d_val_full.json" \
  --predictions "$ROOT/outputs/20261020_r8_object_predictions" \
  --scene-root /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs \
  --category-top-k 5 --workers 8 --api-concurrency 8 --review-limit 3 \
  --output /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021 "$@"
