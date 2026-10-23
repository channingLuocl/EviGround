#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SEEGROUND_R8_GROUNDING_REVISION=r2
exec bash "$ROOT/local/run_local_v74_r8.sh" "$@"
