# All-false gpt-6.1-sol v1 — independent manual retry

Baseline: the completed 9508-query cumulative result in
outputs/r18-visual109-v1/results.json, Acc@0.25 6131/9508 (64.48%).
Select **all 3377 rows that fail Acc@0.25** in that baseline. This includes
wrong non-null answers, abstentions, and runtime/API errors. Selection uses
evaluation outcomes only in the parent; workers receive scene, original query,
Top-5 predictions and geometry, never target ID, IoU, prior answer or audit.

Use the frozen visual109_runtime and visual109_policy.py without changing
their candidate, atlas, ReAct, schema or three-action rules. The only intended
inference-setting changes from the r18 manifest are:

- API base http://106.14.25.135:3000/v1, OpenAI-compatible Chat Completions;
- requested model gpt-6.1-sol;
- private credential file $HOME/.config/seeground/scanrefer-gpt61-sol.env.

The key is stored owner-only, outside the experiment, and is never written to
the protocol, command line, manifest, console summary or code. Its file hash
is frozen so a changed credential cannot silently continue the same run.
The endpoint uses **plaintext HTTP**, as supplied by the user; the token and
request content travel without transport encryption. The endpoint is a
third-party compatible gateway; its /v1/models listed the requested model
and two synthetic image/strict-JSON probes returned HTTP 200. Those probes
do not establish full-run accuracy, reliability, billing or official OpenAI
model availability. They used about 22.4k prompt tokens each, so total usage
and cost could be substantial and are not estimated from that sample.

An offline inventory of the 3377 rows found 140 scenes and at most 102
classifier-covered instances per scene. With the unchanged visual109 policy,
the round can require about **111,899 instance tiles and 10,798 atlas pages**
across repeated queries. These are planned counts, not completed work.
The initial request has up to 12 images; subsequent ReAct calls can retain
additional inspection images. Eight workers and eight API slots match r18.

Output: outputs/r18-all-false-gpt61-sol-v1, exactly one new round.
Every selected outcome replaces its old outcome, including new mistakes and
errors. The 6131 currently correct rows remain untouched. There is no
best-of-oracle merge, no RL and no second automatic round. All result types
are cached equally. A partial trajectory without a prediction or evaluation
requires audited recovery before resume. Completed runs cannot launch again.

The displayed accuracy is a **cumulative cross-version** metric on a
known-failure development subset, not a full-set single-pass or independent
test accuracy. The frozen r18 code/input hashes, completed source hashes,
selection, executable files, policy files, credential fingerprint and model
settings are verified before any benchmark model call.

prepare and preflight make no benchmark model calls. The full retry is
deliberately manual:

    cd /Volumes/KESU/experiments/scanrefer_r18_gt65
    bash retry_all_false_gpt61.sh preflight
    bash retry_all_false_gpt61.sh run
