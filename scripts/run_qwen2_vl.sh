#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  2b|72b) model="$1"; shift ;;
  *) echo "Usage: $0 {2b|72b} {nr3d|scanrefer} [eviground arguments]" >&2; exit 2 ;;
esac
case "${1:-}" in
  nr3d) dataset="$1"; source=seeground_pointnext_clip_reproduction; shift ;;
  scanrefer) dataset="$1"; source=seeground_mask3d_scannet200; shift ;;
  *) echo "Specify the native input dataset: nr3d or scanrefer" >&2; exit 2 ;;
esac
: "${EVIGROUND_NATIVE_INPUT_ROOT:?Set EVIGROUND_NATIVE_INPUT_ROOT to the frozen native inputs}"
for arg in "$@"; do
  case "$arg" in
    --predictions|--predictions=*|--category-cache|--category-cache=*|--expected-category-source|--expected-category-source=*|--proposal-policy|--proposal-policy=*|--category-top-k|--category-top-k=*|--review-limit|--review-limit=*)
      echo "Native launcher owns the category source, cache and evidence budget: $arg" >&2; exit 2 ;;
  esac
done
cd "$PROJECT_ROOT"
export EVIGROUND_MODEL="qwen2-vl-$model" EVIGROUND_API_PROVIDER=openai
export EVIGROUND_BASE_URL="${EVIGROUND_BASE_URL:-http://127.0.0.1:18000/v1}"
export EVIGROUND_API_KEY="${EVIGROUND_API_KEY:-EMPTY}"
export EVIGROUND_ENABLE_THINKING=0 EVIGROUND_OPENAI_TOKEN_PARAMETER=max_tokens
export EVIGROUND_OPENAI_TEMPERATURE=0 EVIGROUND_ROOT="$PROJECT_ROOT"
export EVIGROUND_DAG_MONITOR_ENABLED=1 EVIGROUND_POLICY_ANNEALING_ENABLED=1
export EVIGROUND_SUCCESSFUL_TOOL_BUDGET=1
export EVIGROUND_INSPECT_ENABLED=1 EVIGROUND_MEASURE_ENABLED=1 EVIGROUND_CATEGORY_INSPECTION_REQUIRED=1
export EVIGROUND_ACTION_NO_REPEAT="${EVIGROUND_ACTION_NO_REPEAT:-1}"
export EVIGROUND_GUIDED_DECODING_BACKEND="${EVIGROUND_GUIDED_DECODING_BACKEND:-guidance}"
export EVIGROUND_DAG_REPAIR_LIMIT=6
export EVIGROUND_DECISION_MAX_TOKENS="${EVIGROUND_DECISION_MAX_TOKENS:-4000}"
export EVIGROUND_API_READ_TIMEOUT="${EVIGROUND_API_READ_TIMEOUT:-900}"
export EVIGROUND_CONTEXT_LIMIT="${EVIGROUND_CONTEXT_LIMIT:-32768}"
export EVIGROUND_PROMPT_PROFILE="${EVIGROUND_PROMPT_PROFILE:-compact}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export EVIGROUND_MONITOR_GPU=1
exec "${PYTHON:-$PROJECT_ROOT/.venv/bin/python}" -m eviground \
  --predictions "$EVIGROUND_NATIVE_INPUT_ROOT/$dataset/categories" \
  --category-cache "$EVIGROUND_NATIVE_INPUT_ROOT/shared_tables/$dataset" \
  --expected-category-source "$source" --category-top-k 5 --review-limit 10 "$@"
