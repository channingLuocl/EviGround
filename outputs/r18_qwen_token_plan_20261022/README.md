# R18 Qwen Token Plan independent single pass — 2026-10-22

The user supplied a Token Plan-specific endpoint and updated credential. Text and an actual scene image both returned HTTP 200 with model `qwen3.7-plus`; see `provider_health.json`. Earlier Coding Plan authentication errors and standard-API arrears belong to different profiles/endpoints and are not evidence that this Token Plan profile is unavailable. Credentials stay in a mode-600 file outside Git.

This run uses fixed inference source `4ea57f8`, including model-selected ReAct inspect/measure, predicted Top-5 hypotheses and shared local action-schema validation. The 245-file source snapshot is reused unchanged from `outputs/r18_qwen_single_pass_20261022/runtime_4ea57f8`; all files were rechecked against the source manifest before launch.

The eight-case smoke selection is a pre-existing failure-focused mechanism test, not an Overall estimate. Every failure and abstention stays in its denominator. All Token Plan results use fresh output directories and cannot be merged with standard Qwen or GPT results. No historical correct-result accumulation is allowed.

The final gate remains a complete, independently executed 7,805-row official cohort with at least 66% Overall. The smoke set and incomplete full-run checkpoints do not satisfy that gate. Source hashes, provider/model, input identity and complete-row auditing must match the frozen manifest.
