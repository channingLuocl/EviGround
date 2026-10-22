# Qwen thinking comparison — 2026-10-22

The target remains one independently executed 7,805-query Top-5 run with at least 66% Overall, retaining model-selected ReAct evidence acquisition. No failed outcome, abstention or earlier correct result may be removed or merged into that run.

This study first compares thinking enabled with the completed eight-case non-thinking mechanism smoke. Both use the same fixed `fd2ebe8` inference source, category inputs, query selection, ReAct tools and tool budgets. The sole inference setting change is `enable_thinking=true`, which uses the client's existing 8,192-token minimum. The launcher defaults to non-thinking and now honors an explicit `SEEGROUND_ENABLE_THINKING` override. Existing running processes and the remote frozen branch are unaffected.

The test selection was chosen earlier to reproduce failures. It cannot estimate Overall accuracy. `protocol.json` records the selection, baseline, source, output and expected configuration; `launch.json` identifies the live process. The run manifest confirms thinking enabled, and actual successful API usage contains reasoning tokens. A complete eight-row audit and paired comparison are required before drawing even a mechanism-test conclusion.

`baseline_diagnostic_checkpoint.json` records an incomplete baseline snapshot and specific failure reasons. In those examples, the true target is retained in the candidate pool, while incorrect reference selection, observer placement or ordinal grouping still produces a wrong answer. One separate API failure returned HTTP 429 with `insufficient_quota`; subsequent requests succeeded, and that failure remains in the baseline denominator. No retry policy or endpoint was changed on this evidence.

The original non-thinking full run continues in its fixed runtime and separate output. Any promising setting must subsequently be evaluated on a complete independent cohort and ultimately all 7,805 official rows. Neither a partial full run nor a small mechanism smoke proves the goal achieved.

A separate non-thinking Qwen validation run uses the already fixed 192-row holdout selected with seed `20261022`, excluding 288 previously used development/validation rows. Its protocol and launch record are `validation192_protocol.json` and `validation192_launch.json`. This is a fresh execution with its own output, using the same frozen runtime and counting every failure. It provides a broader check while the full official run continues; results cannot be merged across outputs or providers.

## Completed thinking comparison

Both complete eight-case runs produced three correct predictions, zero fatal errors and zero structural audit issues. Thinking corrected one previously wrong case and regressed one previously correct case. The sum of per-case pipeline times increased from 357.03 to 1,475.75 seconds (about 4.1 times). Global thinking was not promoted to the full run. See `smoke8_audit.json` and `paired_smoke8_comparison.json`; this selected, small set does not establish a general model ranking.

## Bounded-trajectory regression

The fixed non-thinking full run exposed another real failure on row 767: fourteen repeated requests for an already acquired measurement followed the first successful measurement. Rejected duplicates correctly consumed no evidence tools, but the controller exhausted its decision loop without ever requiring a final answer. Two new tests reproduced that exact failure before the fix.

Commit `1658bd0` reserves the last bounded decision turn for `finish` and prevents evidence guards from reopening tools on that last turn. Missing evidence stays disclosed in `unresolved`; no object ID is substituted by the controller. All 47 relevant tests passed in an isolated source copy. The real regression case then completed its 15-step trajectory with a valid final prediction and no runtime error. Its prediction was still incorrect, so this is a verified reliability fix, not evidence of an accuracy gain. The earlier full-run failure remains unchanged. See `finalturn_smoke1_audit.json` and the corresponding source manifest and protocol.

## Signed spatial order experiment

Commit `f4da78f` adds an explicit increasing-forward coordinate order and axis semantics to a model-selected observer-frame measurement. The origin and facing coordinates continue to use measured object centers, and the model still chooses the reference groups, tools and final target. This adds no category rule or oracle target selection. All 48 relevant tests pass, including different left-to-right and forward orders and negative forward coordinates.

The real test reruns four observed directional failures plus the first four already-correct directional controls sorted by official row number, all in a fresh output and fixed source copy. `axis_order_smoke8_manifest.json` records the exact selection rule. This is a mechanism development set, not independent validation; a win here would require broader validation before full-run promotion. The original full run and independent 192-row run still use `fd2ebe8` without this experimental change.
