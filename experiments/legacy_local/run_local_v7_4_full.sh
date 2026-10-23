#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export SEEGROUND_V4=1
export SEEGROUND_V7=1
export SEEGROUND_V7_2=0
export SEEGROUND_V7_3=0
export SEEGROUND_V7_4=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Historical correct answers are imported automatically; failed rows run v7.4.
run_dir="${SEEGROUND_RUN_DIR:-$ROOT/local_outputs/v7_4_nr3d_val_full_gt}"

exec .venv/bin/python -u local/run_size_validation.py \
  --manifest "$ROOT/local/validation/nr3d_val_full.json" \
  --scene-root "$ROOT/local_outputs" \
  --output "$run_dir" \
  --review-limit 3 \
  --workers "${SEEGROUND_WORKERS:-4}" \
  --api-concurrency "${SEEGROUND_API_CONCURRENCY:-2}" \
  --gt \
  "$@"
