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
export SEEGROUND_V7_2=1
export SEEGROUND_V7_3=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Parse the label source with the same contract as the Python entry point.
category_source="$(.venv/bin/python - "$@" <<'PY'
import argparse
import sys
from run_size_validation import parse_bool
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--gt', nargs='?', const=True, default=False, type=parse_bool)
args, _ = parser.parse_known_args(sys.argv[1:])
print('gt' if args.gt else 'pred')
PY
)"
run_dir="${SEEGROUND_RUN_DIR:-$ROOT/local_outputs/v7_2_size_300_$category_source}"

exec .venv/bin/python -u local/run_size_validation.py \
  --manifest local/validation/nr3d_size_300.json \
  --scene-root "$ROOT/local_outputs" \
  --output "$run_dir" \
  --review-limit 3 \
  --workers "${SEEGROUND_WORKERS:-4}" \
  --api-concurrency "${SEEGROUND_API_CONCURRENCY:-2}" \
  "$@"
