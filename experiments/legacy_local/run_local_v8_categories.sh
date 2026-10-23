#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Category experiment, without v7 grounding policies or correctness caches.
export SEEGROUND_V4=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

ENTRYPOINT=local/run_object_classification.py
DEFAULT_OUTPUT="$ROOT/local_outputs/v8_1_independent_subset"
for argument in "$@"; do
  if [[ "$argument" == --cloud || "$argument" == --cloud=* ]]; then
    ENTRYPOINT=local/predict_pointcloud_v8.py
    DEFAULT_OUTPUT="$ROOT/local_outputs/v8_1_independent_cloud"
    break
  fi
done

exec .venv/bin/python -u "$ENTRYPOINT" \
  --workers "${SEEGROUND_WORKERS:-10}" \
  --api-concurrency "${SEEGROUND_API_CONCURRENCY:-8}" \
  --output "${SEEGROUND_RUN_DIR:-$DEFAULT_OUTPUT}" \
  "$@"
