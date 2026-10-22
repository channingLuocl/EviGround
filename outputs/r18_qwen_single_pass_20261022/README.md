# R18 Qwen independent single-pass evaluation — 2026-10-22

Goal: preserve model-selected ReAct inspect/measure evidence, use predicted Top-5 categories, and verify at least 66% Overall on all 7,805 official rows. Errors and abstentions remain in the denominator. No historical correct-result reuse.

## Provider checks

The existing external `qwen.env` profile returned HTTP 200 for text and an actual scene image using `qwen3-vl-plus`. The newly supplied credential is stored only in a separate mode-600 external profile; its matching Coding Plan endpoint returned HTTP 401 (`invalid_api_key`). No credential values are stored in this directory. `provider_health.json` records sanitized checks, which are not benchmark predictions.

Use the verified standard Qwen profile for the independent evaluation. Do not mix these predictions with existing GPT runs, which retain their original source snapshot and result directories.

## Minimal code change

Qwen previously received JSON-object mode without the local strict-schema checks used for GPT. The same schema text, bounded format retries, action-ID validation and evidence requirements now apply to both providers. Qwen keeps its own API parameters; GPT completion parameters are not sent to Qwen. The ReAct policy, candidate rules and evidence budgets are unchanged.

The Qwen API, ReAct and predicted-category grounding suites pass 44 tests. A fixed runtime is created from the committed source before evaluation. The eight-case smoke set is reused only as a mechanism/stability check and cannot estimate Overall. Development comparisons use the existing dev32 selection. The final result must come from a separate complete 7,805-row run, with all failures retained.

## Launch

`run_qwen_r18.sh` loads only the verified external Qwen profile, explicitly activates R18 and Top-5, and supports `SEEGROUND_RUNTIME_PROJECT` for a fixed source snapshot. Each fresh experiment needs a new output directory. Frozen protocol/source manifests record the exact selection, model and code before launch.
