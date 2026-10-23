# Qwen Top-5 complete single-pass result — 2026-10-23

This snapshot freezes the completed original r18 thinking run. It contains all **7,805** selected Nr3D queries and **5,437** correct object-ID predictions: **69.66% Overall**. It uses predicted Top-5 categories and `qwen3.7-plus` with thinking enabled. Resumption preserved every existing outcome equally, including incorrect predictions and failures. No historical-correct-only accumulation is used.

| Method | Model | Easy | Hard | Dep. | Indep. | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| Predicted-category Top-5 full single pass (r18, thinking enabled) | qwen3.7-plus | 74.34 | 65.28 | 62.10 | 73.78 | 69.66 |
| Correct / total | | 2,805 / 3,773 | 2,632 / 4,032 | 1,709 / 2,752 | 3,728 / 5,053 | 5,437 / 7,805 |

Values are percentages of all selected queries in each group. Group assignments and targets come from the frozen official selection. Accuracy is exact predicted-object-ID correctness, not IoU accuracy. All **176 nulls**, including **36 execution failures**, count as incorrect. The raw summary's `completed=7629` counts answered queries; all 7,805 queries have recorded outcomes. The other 140 nulls are abstentions.

## Frozen artifacts

- `results.json.gz`: lossless byte-for-byte compressed copy of the final raw results, including per-query API diagnostics.
- `original_manifest.json.gz`: lossless copy of the original run manifest with selection, source hashes, scene hashes and configuration.
- `per_query_results.csv`: all 7,805 outcomes in CSV-row order, with targets, predictions, group flags and execution-failure flags.
- `metrics.json`: independently recomputed group counts and accuracies.
- `table_row.md` and `table_row.tsv`: the requested Chinese table row. “定位零样本” describes the grounding stage; it does not assert that the upstream category classifier is untrained.
- `final_audit.json`: per-case audit of actual prediction/evaluation/run records, candidate membership, model identity, GT-input exclusion flags, source identity and the complete denominator.
- `preservation.json`: verifies that all 2,889 pre-resumption evaluation bytes and prediction metadata remain unchanged.
- `source_input_integrity.json`: verifies all 152 frozen inference files and all 140 category-input files.
- `raw_artifacts.json` and `freeze_manifest.json`: original and snapshot checksums.

The inference source is already committed at `outputs/r18_qwen_original_resume_20261022/runtime_original/local/`. These 152 files match the original run manifest. The repository's current experimental `reproductions/SeeGround/local/` code is not the frozen inference entry point.

The exact 140 input files are already committed at `outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`. They match the actual runtime export at `outputs/20261020_r8_object_predictions/`. The canonical directory can be absent from the sparse working tree; its Git blobs are verified directly. Top-5 scores are weighted standardized 3D/RGB fusion scores, not probabilities.

The original full output directory remains at `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021`. Per-case full traces, rendered images and raw ScanNet scene assets remain there; they are not duplicated in this compact Git snapshot. Their provenance is recorded in the original manifest and the final audit. No inference source, input, or original result was edited to freeze this result.

## Configuration and provenance

The preserved launcher is `outputs/r18_qwen_original_resume_20261022/resume_original_qwen.sh`: Top-5, workers 8, API concurrency 8, review limit 3, `qwen3.7-plus`, thinking enabled, thinking-token floor 8192, temperature zero, native `json_schema` output, and `max_completion_tokens`. The `openai` provider label denotes the compatibility adapter used for Qwen.

The raw result has two implementation-version labels: 7,799 rows are labeled `predicted-category-grounding-v9-r8-r18`, while six pre-existing execution-failure rows carry `predicted-category-grounding-v9-r1`. Those six rows have no predicted object, remain incorrect, and were not rerun or replaced. Consequently the raw summary reports `mixed_implementation_versions=true`; the raw file is preserved as-is. Source and per-call model checks are reported separately by the audits.

Credentials remain in the private profile outside Git. This snapshot contains no API key and verification makes no API calls.

## Verify without inference

From the repository root:

```bash
python3 outputs/r18_qwen_top5_full7805_freeze_20261023/verify_snapshot.py
```

This verifies snapshot checksums, lossless raw copies, all 152 inference-source hashes, all 140 canonical Git input blobs, exact predictions against selected targets, every metric and the final audit. It does not require the external drive or credentials.

To repeat the per-case audit while the original external output is available:

```bash
python3 outputs/r18_context_budget_fix_20261022/audit_single_pass.py \
  --run '/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021' \
  --runtime outputs/r18_qwen_original_resume_20261022/runtime_original/local \
  --freeze-manifest outputs/r18_qwen_original_resume_20261022/audit_freeze_manifest.json \
  --output /tmp/r18_qwen_top5_full7805_reaudit.json
```

The earlier GT-category cumulative score and the earlier predicted-category cumulative score are different evaluation policies. This row should be named as a complete predicted-category Top-5 single pass.
