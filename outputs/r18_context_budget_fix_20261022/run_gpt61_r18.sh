#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SEEGROUND_RUNTIME_PROJECT:-$ROOT/reproductions/SeeGround}"
set -a
source /Users/luocl/.config/seeground/gpt61-relay.env
set +a
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0 SEEGROUND_V9_SOURCE=r8
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_SYNTAX_BACKEND=spacy
export SEEGROUND_API_PROVIDER=openai SEEGROUND_REACT_BACKEND=openai
export SEEGROUND_MODEL=gpt-6.1-sol SEEGROUND_BASE_URL=http://106.14.25.135:3000/v1
export SEEGROUND_ALLOW_HTTP=1 SEEGROUND_OPENAI_TOKEN_PARAMETER=max_completion_tokens
export SEEGROUND_ENABLE_THINKING=1 SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_CONTEXT_LIMIT=131072 PYTHONPATH=local PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$ROOT/reproductions/SeeGround/.venv/bin/python" -c 'from qwen_api import _settings, request_options; base,model,key=_settings(); assert key and model=="gpt-6.1-sol"; assert request_options(1600,None,True)["max_completion_tokens"]==8192; print("Preflight passed: GPT relay, exported credential, 8192 token floor")'
exec "$ROOT/reproductions/SeeGround/.venv/bin/python" -u local/run_grounding_v9.py \
  --predictions "$ROOT/outputs/20261020_r8_object_predictions" \
  --scene-root /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs \
  --category-top-k 5 --workers 4 --api-concurrency 4 "$@"
