#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/outputs/r18_qwen_top5_full7805_freeze_20261023/verify_snapshot.py"
