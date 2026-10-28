#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
ACTION="${1:-preflight}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(prepare|preflight|run|report)$ ]]; then
  echo 'Usage: bash retry_all_false_gpt61.sh [prepare|preflight|run|report]' >&2
  exit 2
fi
PY=/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
CREDENTIAL="$HOME/.config/seeground/scanrefer-gpt61-sol.env"
if [[ ! -f "$CREDENTIAL" ]]; then
  echo 'Private model credential file is missing' >&2
  exit 2
fi
for name in ${!SEEGROUND_@}; do unset "$name"; done
set -a
source /Users/luocl/Desktop/seeground/outputs/r13_single_pass_20261020/source_r18/.env
source "$CREDENTIAL"
set +a
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0
export SEEGROUND_R8_GROUNDING_REVISION=r18 SEEGROUND_QUERY_SYNTAX=spacy
export SEEGROUND_V9_SOURCE=r8
export SEEGROUND_REACT_BACKEND=openai SEEGROUND_API_PROVIDER=openai
export SEEGROUND_BASE_URL=http://106.14.25.135:3000/v1 SEEGROUND_MODEL=gpt-6.1-sol
export SEEGROUND_ALLOW_HTTP=1
export SEEGROUND_ENABLE_THINKING=1 SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_OPENAI_TEMPERATURE=0 SEEGROUND_CONTEXT_LIMIT=131072
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SEEGROUND_VISUAL109_POLICY=all_instances_atlas_binding_v1
export PYTHONPATH="$HERE/visual109_runtime:$HERE/source_r18/local:$HERE"
exec /usr/bin/caffeinate -i "$PY" -u "$HERE/retry_all_false_gpt61.py" "$ACTION"
