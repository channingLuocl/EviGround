#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export SEEGROUND_V4=1 SEEGROUND_V7=1 SEEGROUND_V7_4=1 SEEGROUND_V9=1
export SEEGROUND_V7_2=0 SEEGROUND_V7_3=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=local
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec "${SEEGROUND_V9_PYTHON:-$ROOT/.venv/bin/python}" -u local/run_grounding_v9.py \
  --workers "${SEEGROUND_WORKERS:-4}" \
  --api-concurrency "${SEEGROUND_API_CONCURRENCY:-2}" "$@"
