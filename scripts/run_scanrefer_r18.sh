#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/run_scanrefer_r18.sh MANIFEST PREDICTIONS SCENE_ROOT OUTPUT [options]" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R18_ROOT="${SEEGROUND_R18_ROOT:-$PROJECT_ROOT/../seeground/outputs/r13_single_pass_20261020/source_r18}"
R18_PYTHON="${SEEGROUND_V9_PYTHON:-$PROJECT_ROOT/../seeground/reproductions/SeeGround/.venv/bin/python}"
if [[ -f "$R18_ROOT/.env" ]]; then
  set -a
  source "$R18_ROOT/.env"
  set +a
fi
API_ENV_FILE="${SEEGROUND_ENV_FILE:-$HOME/.config/seeground/qwen-token-plan-20261022.env}"
if [[ -f "$API_ENV_FILE" ]]; then
  set -a
  source "$API_ENV_FILE"
  set +a
elif [[ -n "${SEEGROUND_ENV_FILE:-}" ]]; then
  printf 'API 配置文件不存在：%s\n' "$API_ENV_FILE" >&2
  exit 2
fi
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_QUERY_SYNTAX=spacy
export SEEGROUND_REACT_BACKEND=openai SEEGROUND_API_PROVIDER=openai SEEGROUND_MODEL=qwen3.7-plus
export SEEGROUND_ENABLE_THINKING=1 SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_OPENAI_TEMPERATURE=0 SEEGROUND_CONTEXT_LIMIT=131072
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R18_ROOT/local"
"$R18_PYTHON" - <<'PY'
import sys
from qwen_api import QwenAPIError, _settings

try:
    _settings()
except QwenAPIError as exc:
    print(f"ScanRefer API 配置检查失败：{exc}；请设置 SEEGROUND_ENV_FILE 指向已有的私密配置文件。", file=sys.stderr)
    sys.exit(2)
print("ScanRefer API 配置检查通过（未发送请求）。", file=sys.stderr)
PY
MANIFEST="$1" PREDICTIONS="$2" SCENE_ROOT="$3" OUTPUT="$4"
shift 4
exec "$R18_PYTHON" -u "$R18_ROOT/local/run_grounding_v9.py" \
  --manifest "$MANIFEST" --predictions "$PREDICTIONS" \
  --scene-root "$SCENE_ROOT" --output "$OUTPUT" \
  --category-top-k 5 --workers 8 --api-concurrency 8 --review-limit 3 "$@"
