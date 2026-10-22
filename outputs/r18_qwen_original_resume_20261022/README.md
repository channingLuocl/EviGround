# Original Qwen thinking run resumed — 2026-10-22

The user explicitly changed the active task to resume the earlier Qwen run, rather than continue the newer inference experiments. The newer Qwen full run, Qwen validation and remaining GPT experimental runs were stopped by verified process-group identity. Their outputs were preserved; see `new_experiments_stopped.json`.

The original output is `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021`. It contained **2,889** recorded outcomes before resumption: 1,972 correct, 18 recorded errors and 4,916 queries remaining out of 7,805. Its completed-fragment accuracy was 68.26%; this is not a completed full-benchmark score.

All **152 inference source files** were recovered from `outputs/r13_single_pass_20261020/source_r18/local` and matched individually against the original run's `manifest.json`. The recovered files are isolated in `runtime_original/local`; no newer candidate fallback, observation compaction, image limit, loop fix or spatial-order experiment was inserted into this runtime. Its source manifest records every hash. The current working experiment code is not used by the resume launcher.

The original API and policy configuration were restored exactly: `qwen3.7-plus`, `enable_thinking=true`, an 8,192-token minimum, temperature zero, native JSON-schema output, `max_completion_tokens`, Top-5 categories, eight workers and concurrency eight. The original `provider=openai` value names its compatibility adapter; the actual model remains Qwen. Both endpoint and model match the original manifest. The updated credential is read from the private Token Plan profile outside Git.

The original driver accepted `--prepare-only` in the existing output directory without changing the original manifest: 2,889 cached outcomes and 4,916 pending. All prior evaluation hashes were checked unchanged. An image request through the exact original API configuration then succeeded, returned `qwen3.7-plus`, and reported reasoning tokens; see `original_provider_health.json` and `resume_preflight.json`.

`resume_original_qwen.sh` resumes this same directory. Existing correct, incorrect and failed evaluations are reused equally; only queries without evaluations continue. No result from any newer run is imported. `before_resume.json` inventories the existing evaluations and prediction-file metadata, and `resume_launch.json` identifies the resumed process. The original manifest and all prior outcomes must remain unchanged during continuation.

The original full run is still incomplete. Its final Overall and any 65–66% target require completion of all 7,805 queries with failures and abstentions retained in the denominator.

## Read-only original-format audit

`audit_freeze_manifest.json` freezes the 152 original source hashes and the exact original provider configuration for the external auditor. It explicitly selects the original `category_target_ids` candidate pool, native `json_schema` output, and absent local-validation flag. The newer auditor's default checks remain unchanged. No inference source is edited by this compatibility change.

All nine auditor unit tests pass, including original-format candidate membership, counted empty-category abstentions, and fatal failures retained in the full denominator. `original_preservation_checkpoint_20261022.json` independently verifies that all 2,889 pre-resume evaluation hashes, existing prediction-file metadata, and the original manifest remain unchanged.

Run the read-only benchmark audit from the repository root:

```bash
reproductions/SeeGround/.venv/bin/python \
  outputs/r18_context_budget_fix_20261022/audit_single_pass.py \
  --run '/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021' \
  --runtime outputs/r18_qwen_original_resume_20261022/runtime_original/local \
  --freeze-manifest outputs/r18_qwen_original_resume_20261022/audit_freeze_manifest.json \
  --output outputs/r18_qwen_original_resume_20261022/original_live_audit_20261022.json
```

This recomputes correctness from real predicted object IDs and the official selected targets; saved `correct` flags are cross-checked. A live checkpoint can include cases completed while the audit is reading. It is not an atomic snapshot or a final benchmark score. The completion and 66% gates remain false until all 7,805 outcomes are present and structural checks pass.

The first compatible live audit recorded 2,979 outcomes and 2,031 correct real predictions (68.18% of the completed fragment), with 4,826 still missing. It found zero structural issues, 19 recorded fatal errors and 68 category-gate abstentions. Errors and abstentions remain in the denominator. See `original_live_audit_20261022.json` for the complete checkpoint and `original_live_checkpoint_20261022.json` for its compact summary; neither establishes a final Overall or a passed accuracy threshold.

## Input identity and later checkpoint

`original_input_integrity_latest.json` verifies all **140 committed input files** against the actual runtime export using Git blob identifiers at input snapshot `4f750d1`. Every file matches byte for byte. The SHA-256 hashes of the actual `predictions.json` and `config.json` also match the original run manifest. The canonical Git data is read directly from Git objects to accommodate the existing sparse checkout. These checks preserve the running input files.

The live audit completed at 2026-10-22 07:24 UTC recorded **3,563 outcomes**, **2,447 correct real predictions** (68.68% of the completed fragment), 19 fatal errors and 76 category-gate abstentions, with zero structural issues. It still had 4,242 missing outcomes. `original_live_audit_20261022T072033Z.json` contains that checkpoint; `original_latest_audit_checkpoint.json` records its summary and SHA-256. Failures and abstentions remain counted. The full benchmark, final Overall and accuracy threshold remain unverified until all 7,805 outcomes are present and audited.
