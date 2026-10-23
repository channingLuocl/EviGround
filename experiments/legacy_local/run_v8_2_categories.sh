#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Deliberately do not load .env or invoke the VLM API / old grounding runner.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON="${SEEGROUND_V82_PYTHON:-python}"
exec "$PYTHON" -u local/run_categories_v82.py "$@"
