# R18 Qwen Token Plan independent single pass — 2026-10-22

The user supplied a Token Plan-specific endpoint and updated credential. Text and an actual scene image both returned HTTP 200 with model `qwen3.7-plus`; see `provider_health.json`. Earlier Coding Plan authentication errors and standard-API arrears belong to different profiles/endpoints and are not evidence that this Token Plan profile is unavailable. Credentials stay in a mode-600 file outside Git.

The original smoke used fixed inference source `4ea57f8`, including model-selected ReAct inspect/measure, predicted Top-5 hypotheses and shared local action-schema validation. That original 245-file source snapshot remains unchanged in `outputs/r18_qwen_single_pass_20261022/runtime_4ea57f8`; the fixed full run below uses `fd2ebe8`.

The eight-case smoke selection is a pre-existing failure-focused mechanism test, not an Overall estimate. Every failure and abstention stays in its denominator. All Token Plan results use fresh output directories and cannot be merged with standard Qwen or GPT results. No historical correct-result accumulation is allowed.

The final gate remains a complete, independently executed 7,805-row official cohort with at least 66% Overall. The smoke set and incomplete full-run checkpoints do not satisfy that gate. Source hashes, provider/model, input identity and complete-row auditing must match the frozen manifest.

## Repeated-measurement crash and fixed runtime

The original Token Plan smoke completed all eight selected cases: one correct, four other predictions and three runtime failures. All three failures were `KeyError: objects`, triggered when the model repeated a measurement. The runtime correctly rejected the duplicate and recorded a no-evidence error observation; geometry compaction then incorrectly indexed that error as successful measurement data.

Commit `fd2ebe8` fixes the compaction condition so no-evidence errors remain available to the next ReAct decision without being encoded as geometry. A new unit test first reproduced the same crash, then passed after the one-line fix. The 24 ReAct, 11 Qwen API and 10 predicted-category grounding tests all pass (45 total).

The fixed snapshot is `runtime_fd2ebe8/`; its 245 Python hashes are recorded in its source manifest. `measurefix_protocol.json` and `measurefix_freeze_manifest.json` apply to the new smoke and planned full run. Earlier runtime snapshots, outputs, failures and the unstarted old full-run preparation remain intact. The fixed smoke repeats all eight cases in a new directory; it does not replace only the three failed items or combine best answers.

## Fixed smoke audit and full-run launch

The fixed smoke completed all eight cases with eight integer predictions, zero fatal errors and zero structural audit issues. Three predictions were correct. This failure-focused set is a mechanism test, so its 37.5% result is not an estimate of full-benchmark accuracy. See `smoke8_measurefix_audit.json`.

The fresh 7,805-row full run started at 2026-10-22 05:05:53 UTC (13:05:53 Asia/Shanghai), using source `fd2ebe86e2ed47fbad0e947c7d6e733a349e3e76`, eight workers and API concurrency eight. Before launch, all 245 runtime Python hashes, the official selection hash and prediction input hash were rechecked; the output contained zero evaluations, predictions or run records. See `full7805_measurefix_preflight.json` and `full7805_measurefix_launch.json`. A launch record does not establish completion or a passing score.

The branch `codex/r18-qwen-token-plan-20261022` includes the inference source, tests, launch script, input manifests and the inherited 140 prediction data files under `outputs/r8_grounding_input_20261020/20261020_r8_object_predictions/`. Every one of those data files was compared against the actual runtime input before this freeze. The runtime snapshot can be reconstructed from the recorded source commit and checked against `runtime_fd2ebe8/source_manifest.json`; generated runtime copies and live logs are not needed in Git.

Credentials remain outside the repository in the private profile referenced by the launcher. All current benchmark outputs remain separate from earlier Qwen and GPT runs. The full run must still finish and pass the complete-denominator audit before reporting an Overall score or claiming the 66% goal is met.
