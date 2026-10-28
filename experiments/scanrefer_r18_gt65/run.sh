#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python
ACTION="${1:-preflight}"
SPLIT="${2:-84}"
VERSION="${GT65_VERSION:-r18-baseline-v1}"
RETRY="${3:-}"
if [[ $# -gt 3 || ( -n "$RETRY" && "$RETRY" != --retry-false ) ]]; then
  echo 'Usage: GT65_VERSION=version bash run.sh [preflight|run|report] [32|84|full] [--retry-false]' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$HERE/verify.py" --split "$SPLIT" --version "$VERSION"
API_ENV_FILE="${SEEGROUND_ENV_FILE:-$HOME/.config/seeground/qwen-token-plan-20261022.env}"
for name in ${!SEEGROUND_@}; do unset "$name"; done
set -a
source /Users/luocl/Desktop/seeground/outputs/r13_single_pass_20261020/source_r18/.env
source "$API_ENV_FILE"
set +a
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_QUERY_SYNTAX=spacy
export SEEGROUND_V9_SOURCE=r8
export SEEGROUND_REACT_BACKEND=openai SEEGROUND_API_PROVIDER=openai SEEGROUND_MODEL=qwen3.7-plus
export SEEGROUND_ENABLE_THINKING=1 SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_OPENAI_TEMPERATURE=0 SEEGROUND_CONTEXT_LIMIT=131072
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$HERE/source_r18/local"
/usr/bin/caffeinate -i "$PY" -u "$HERE/experiment.py" "$ACTION" "$SPLIT" --version "$VERSION" ${RETRY:+"$RETRY"}
