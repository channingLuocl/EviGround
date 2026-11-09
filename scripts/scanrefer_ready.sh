#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/Volumes/KESU/datasets/ScanRefer
PY=/Users/luocl/Desktop/seeground/reproductions/SeeGround/.venv/bin/python
MANIFEST="$DATA/scanrefer_val_manifest.json"
SCENES="$DATA/scanrefer_mask3d_prepared_scenes"
PREDICTIONS="$DATA/scanrefer_mask3d_r8_verified_export"
OUT="$DATA/runs/scanrefer_r23_top5_thinking_single_pass"
POLICY="$DATA/proposal_policies/mask_nms080_with_scores_v1.json"
export EVIGROUND_SOURCE_ROOT="$DATA/experiment_sources/r23_scores_recovery_v1/src"
ACTION="${1:-preflight}"
case "$ACTION" in
  preflight|run)
    for path in "$MANIFEST" "$SCENES/protocol.json" "$PREDICTIONS/EXPORT_VERIFIED.json" \
                "$POLICY" "$EVIGROUND_SOURCE_ROOT/eviground/__main__.py"; do
      if [[ ! -f "$path" ]]; then
        printf 'ScanRefer 尚未就绪，缺少：%s\n' "$path" >&2
        printf '请核对 ScanRefer 预测实例、r8 分类及冻结源码；不能复用 Nr3D 的 GT 实例导出。\n' >&2
        exit 2
      fi
    done
    set --
    if [[ "$ACTION" == preflight ]]; then set -- --prepare-only; fi
    exec /usr/bin/caffeinate -i bash "$PROJECT_ROOT/scripts/run_eviground_local.sh" \
      --manifest "$MANIFEST" --predictions "$PREDICTIONS" \
      --scene-root "$SCENES" --proposal-policy "$POLICY" --output "$OUT" \
      --category-top-k 5 --workers 8 --api-concurrency 8 --review-limit 3 "$@"
    ;;
  report)
    exec "$PY" "$PROJECT_ROOT/scripts/scanrefer.py" report \
      --manifest "$MANIFEST" --results "$OUT/results.json" \
      --proposals mask3d --output "$OUT/scanrefer_metrics.json"
    ;;
  *)
    echo 'Usage: bash scripts/scanrefer_ready.sh [preflight|run|report]' >&2
    exit 2
    ;;
esac
