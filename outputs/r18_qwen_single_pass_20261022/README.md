# R18 Qwen independent single-pass evaluation — 2026-10-22

Goal: preserve model-selected ReAct inspect/measure evidence, use predicted Top-5 categories, and verify at least 66% Overall on all 7,805 official rows. Errors and abstentions remain in the denominator. No historical correct-result reuse.

## Provider checks

The existing external `qwen.env` profile initially returned HTTP 200 for text and an actual scene image using `qwen3-vl-plus`. The newly supplied credential is stored only in a separate mode-600 external profile; both the China and international Coding Plan endpoints returned HTTP 401 (`invalid_api_key`). The user confirmed that the purchase was Coding Plan. No credential values are stored in this directory. `provider_health.json` and `coding_plan_international_health.json` record sanitized checks, which are not benchmark predictions.

During the actual eight-case smoke run, the standard API began returning `Arrearage`: six cases failed due to account arrears. The other two cases completed ReAct but were incorrect. All eight remain in the denominator (0/8); this is a selected stress set, not an Overall estimate. One format retry occurred and no case ended in a schema-format failure. The complete audit is in `smoke8_audit.json`.

No Qwen full7805 run was started after these account failures. The standard API needs a working account/profile before another fresh Qwen evaluation. The GPT full7805 and independent192 processes continue with their original source snapshots and separate outputs. No Qwen results replace any GPT results.

## Minimal code change

Qwen previously received JSON-object mode without the local strict-schema checks used for GPT. The same schema text, bounded format retries, action-ID validation and evidence requirements now apply to both providers. Qwen keeps its own API parameters; GPT completion parameters are not sent to Qwen. The ReAct policy, candidate rules and evidence budgets are unchanged.

The Qwen API, ReAct and predicted-category grounding suites pass 44 tests. The category-input suite adds four passing tests when run from the isolated runtime. The broader 34-test real-scene suite has the same pre-existing one failure and one error in both baseline and candidate; see `compatibility_comparison.json`. The repository `.env` activates R18, so legacy tests that expect another inference path require isolation; this is not a new source regression. A fixed runtime was created from committed source `4ea57f8` before evaluation. The eight-case smoke set is reused only as a mechanism/stability check and cannot estimate Overall. Development comparisons use the existing dev32 selection. The final result must come from a separate complete 7,805-row run, with all failures retained.

The full-run auditor now accepts an explicitly frozen Qwen provider configuration while retaining GPT defaults and the official 7,805-row CLI denominator. Actual call model/provider mismatches fail verification. It also recognizes the existing empty-category-pool abstention only when the prediction is null, all candidate pools are empty, and the recorded reason matches the source's category-gate branch. This remains an incorrect item in the denominator. Seven audit tests pass, including provider mismatch, failures, incomplete cohorts and forged abstention coverage.

## Launch

`run_qwen_r18.sh` loads only the verified external Qwen profile, explicitly activates R18 and Top-5, and supports `SEEGROUND_RUNTIME_PROJECT` for a fixed source snapshot. Each fresh experiment needs a new output directory. Frozen protocol/source manifests record the exact selection, model and code before launch.
