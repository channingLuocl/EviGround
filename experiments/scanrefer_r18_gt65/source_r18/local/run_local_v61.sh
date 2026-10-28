#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

workers="${SEEGROUND_WORKERS:-4}"
api_concurrency="${SEEGROUND_API_CONCURRENCY:-2}"
run_dir="${SEEGROUND_RUN_DIR:-$ROOT/local_outputs/v6_1_local_$(date +%Y%m%d_%H%M%S)}"

exec .venv/bin/python -u local/run_size_validation.py \
  --manifest local/validation/v5_development_68.json \
  --scene-root "$ROOT/local_outputs" \
  --output "$run_dir" \
  --review-limit 3 \
  --workers "$workers" \
  --api-concurrency "$api_concurrency"
