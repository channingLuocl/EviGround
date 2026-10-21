#!/usr/bin/env bash
set -euo pipefail
export SEEGROUND_ENABLE_THINKING=1
export SEEGROUND_THINKING_TOKEN_FLOOR=8192
export SEEGROUND_OPENAI_TEMPERATURE=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SEEGROUND_R8_GROUNDING_REVISION=r18
exec bash "$ROOT/local/run_local_v74_r8.sh" "$@"
