#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/Volumes/KESU/datasets/ScanRefer
PY=/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python
ACTION="${1:-preflight}"
COUNT="${2:-32}"
if [[ ! "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Sample count must be a positive integer (currently at most 84).' >&2
  exit 2
fi
PREPARED="$DATA/gt_instance_diagnostics/dev${COUNT}_v1"
OUT="$DATA/runs/scanrefer_gt_instances_predtop5_dev${COUNT}_v1"
PREDICTIONS=/Users/luocl/Desktop/seeground/outputs/20261020_r8_object_predictions
export EVIGROUND_SOURCE_ROOT="$DATA/experiment_sources/r23_scores_recovery_v1/src"
export PYTHONDONTWRITEBYTECODE=1
case "$ACTION" in
  preflight|run)
    "$PY" "$PROJECT_ROOT/scripts/prepare_scanrefer_gt.py" --count "$COUNT"
    set --
    if [[ "$ACTION" == preflight ]]; then set -- --prepare-only; fi
    exec /usr/bin/caffeinate -i bash "$PROJECT_ROOT/scripts/run_eviground_local.sh" \
      --manifest "$PREPARED/manifest.json" --predictions "$PREDICTIONS" \
      --scene-root "$PREPARED/scenes" --output "$OUT" \
      --category-top-k 5 --workers 4 --api-concurrency 4 --review-limit 3 "$@"
    ;;
  report)
    exec "$PY" "$PROJECT_ROOT/scripts/scanrefer.py" report \
      --manifest "$PREPARED/manifest.json" --results "$OUT/results.json" \
      --proposals gt --output "$OUT/scanrefer_metrics.json"
    ;;
  *)
    echo 'Usage: bash scripts/scanrefer_gt_ready.sh [preflight|run|report] [count=32]' >&2
    exit 2
    ;;
esac
