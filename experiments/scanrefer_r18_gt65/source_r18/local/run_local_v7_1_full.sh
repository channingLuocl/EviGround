#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A fixed output directory makes the same command resume after interruption.
export SEEGROUND_RUN_DIR="${SEEGROUND_RUN_DIR:-$ROOT/local_outputs/v7_1_nr3d_val_full}"
reuse_from="${SEEGROUND_REUSE_FROM:-$ROOT/local_outputs/v7_1_size_300_20261014_112437}"

exec bash local/run_local_v7_1.sh \
  --manifest "$ROOT/local/validation/nr3d_val_full.json" \
  --reuse-from "$reuse_from" \
  "$@"
