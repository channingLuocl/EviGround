#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Independent recognition, without v7 grounding policies or correctness caches.
export SEEGROUND_V4=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

exec .venv/bin/python -u local/run_object_classification.py \
  --workers "${SEEGROUND_WORKERS:-10}" \
  --api-concurrency "${SEEGROUND_API_CONCURRENCY:-8}" \
  --output "${SEEGROUND_RUN_DIR:-$ROOT/local_outputs/v8_category_subset}" \
  "$@"
