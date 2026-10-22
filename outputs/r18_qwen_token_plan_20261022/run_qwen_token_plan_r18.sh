#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SEEGROUND_RUNTIME_PROJECT:-$ROOT/reproductions/SeeGround}"
set -a
source /Users/luocl/.config/seeground/qwen-token-plan-20261022.env
set +a
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0 SEEGROUND_V9_SOURCE=r8
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_SYNTAX_BACKEND=spacy
export SEEGROUND_API_PROVIDER=qwen SEEGROUND_REACT_BACKEND=qwen
export SEEGROUND_MODEL=qwen3.7-plus
export SEEGROUND_BASE_URL=https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1
export SEEGROUND_ENABLE_THINKING="${SEEGROUND_ENABLE_THINKING:-0}" SEEGROUND_CONTEXT_LIMIT=131072
export PYTHONPATH=local PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$ROOT/reproductions/SeeGround/.venv/bin/python" -c 'from qwen_api import _settings, api_config_summary; base,model,key=_settings(); config=api_config_summary(); assert key and model=="qwen3.7-plus" and config["provider"]=="qwen" and config["local_schema_validation"]; print("Preflight passed: Qwen Token Plan, exported credential, local JSON schema validation")'
exec "$ROOT/reproductions/SeeGround/.venv/bin/python" -u local/run_grounding_v9.py \
  --predictions "$ROOT/outputs/20261020_r8_object_predictions" \
  --scene-root /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs \
  --category-top-k 5 --workers 4 --api-concurrency 4 "$@"
