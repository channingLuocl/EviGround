#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON="${SEEGROUND_V83_PYTHON:-${SEEGROUND_V82_PYTHON:-python}}"
exec "$PYTHON" -u local/run_categories_v83.py "$@"
