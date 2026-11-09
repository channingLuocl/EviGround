#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SEEGROUND_V9_PYTHON:-$PROJECT_ROOT/../seeground/reproductions/SeeGround/.venv/bin/python}"
set -a
source "$PROJECT_ROOT/../seeground/outputs/r13_single_pass_20261020/source_r18/.env"
source "${SEEGROUND_ENV_FILE:-$HOME/.config/seeground/qwen-token-plan-20261022.env}"
set +a
for name in API_KEY BASE_URL OPENAI_TOKEN_PARAMETER; do
  previous="SEEGROUND_$name"
  if [[ -n "${!previous:-}" ]]; then
    export "EVIGROUND_$name=${!previous}"
  fi
done
export EVIGROUND_MODEL=qwen3.7-plus EVIGROUND_API_PROVIDER=openai
export EVIGROUND_ENABLE_THINKING=1 EVIGROUND_THINKING_TOKEN_FLOOR=8192
export EVIGROUND_OPENAI_TEMPERATURE=0 EVIGROUND_CONTEXT_LIMIT=131072
export EVIGROUND_ROOT="$PROJECT_ROOT" PYTHONPATH="${EVIGROUND_SOURCE_ROOT:-$PROJECT_ROOT/src}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" - <<'PY'
from eviground.cli.entrypoint import configure
from eviground.api.client import _settings
configure()
_settings()
print("EviGround API configuration validated; no request sent.", flush=True)
PY
exec "$PY" -u -m eviground "$@"
