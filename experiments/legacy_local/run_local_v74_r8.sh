#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/local/run_local_v9.sh" \
  --predictions "$ROOT/../../outputs/r8_grounding_input_20261020/20261020_r8_object_predictions" \
  --category-top-k 5 "$@"
