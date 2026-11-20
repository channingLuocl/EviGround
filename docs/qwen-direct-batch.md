# User-started Nr3D full local exploration

This entry point uses the v3 historical-coverage/replanning correction to the
diagnostic003 direct qwen3.8-max protocol. See qwen-direct-v3-loop-fix.md.
It does NOT run the old eviground CLI/model and does NOT load the earlier003
prediction. Strict native Monitor and the model-owned tentative exit remain;
the v3 novelty detector and visible planning contract are corrected. B/R/T and
the artificial32K bound are record-only. Original and targeted
protocol checks plus offline batch checks do not establish semantic correctness.

--preflight is offline ONLY, validates the7805 selection/130scenes, original cloud
hashes, instance geometry and same-source Top5 tables. Reuses existing verified
cache, materializes missing tables from original verified prediction rows without
classification, downloads, GPU or model calls. Missing paths fail explicitly.
Shared external clouds/geometry remain read-only, one asset-index holds hashes;
workers verify source hashes per sample, no7805copies. Source is frozen ONCE per
output; four per-query input references, actual model responses/images/trace remain
per query. Full annotation manifest is copied once parent-side for scoring, never
passed to worker inputs. Worker requests contain only scene/csv_row/query plus
read-only scene/category paths/hash identities. Query projection discards target_id.
Parent reads that sample annotation only after atomically freezing the saved
prediction/failure; correct/wrong/null/error all remain; null/errors count wrong.
Development csv_row12 scene0565_00 is marked previously used/GT revealed, so this
exploration is not independent generalization or official/paper score.

Default workers=1, one new subprocess per sample; no cross-sample global Monitor
or monkeypatch reuse. One output lock prevents parallel accidental duplication.
Same command resumes only identical source/assets/selection/runtime. All saved
completed evaluation rows resume, including failures/null, never just correct.
Started-only/interrupted/unknown worker state is not blindly re-sent; summaries
preserve unknown usage distinctly. First Ctrl+C/SIGTERM stops scheduling, drains
current sample normally and saves its terminal/evaluation. Second Ctrl+C sends
SIGINT to its isolated worker, which saves interrupted trace and request receipts
when possible; incomplete sample excluded from completed denominator. The same
command stops for attention on that interrupted case, does not reset its budget.
Hard process kill may lose last local state; original receipts remain and block
implicit retries. Normal in-flight responses are saved; unknowns never assumed0.

Each completed sample emits BatchProgress correct/completed=percentage and
completed/7805; details include scene/csv_row/pred/GT/status, strict/tentative/null/
failed cumulative counts, actual B/R/T, response/repair/error counts, known tokens,
unknown usage, elapsed. Existing BatchProgress provides speed-derived ETA. Within
sample request_sent/response_saved stages and20-second heartbeat show liveness,
R/T/API and saved latest stage without printing model prompt or credentials.
summary.json and per-sample/evaluation/manifest JSON use existing atomic storage;
progress.jsonl append is flushed/fsynced; individual logs can be tailed. No extra
model reviewer per sample, no monetary price invented, no history total API cap.
No server/GPU requests, no formal score or paper changes. User alone starts and
controls full runtime. Preparing/testing this entry point sends no API request.

## Verified local command (user executes)

```sh
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_direct_full.command
```

No cwd assumption, no key in command; uses the exact existing Python3.12 virtual environment that ran003 and
independent private config. Default outputs:
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/outputs/qwen-direct/full-nr3d-20261120-v3

```sh
tail -f /Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/outputs/qwen-direct/full-nr3d-20261120-v3/runner.log
```

The exact same command resumes. All130shared files/hashes verified before any
inference, including resume; per-worker hashes verified again. Source snapshot,
asset-index and sanitized inference-selection archived once; original annotation
copy stays parent-evaluation/ and is never referenced by worker request. A terminal
failure with unknown remote execution is saved and counted wrong, then parent
stops scheduling for attention, never reruns that request automatically. An
interrupted unfinished sample remains outside completed denominator and blocks
implicit retry; the console explains where to inspect. No accumulated historical
API ceiling. Cost fields are known provider token totals and explicit unknown
usage counts, not fabricated currency estimates or inference-call guesses.

Offline full asset check:7805distinct queries,130scenes,3413classifier objects,
453451827bytes of original scene point-cloud files.125prepared caches copied once
read-only;5missing prepared tables materialized from original verified export:
scene0599_00,scene0633_00,scene0695_00,scene0701_00,scene0702_00. Full source is
20261020_r8_object_predictions, predictions SHA34aa94cbc455324fdd28ddfd21a3832bba743eac25c32d58037c9bc460b1da9e,
config SHAde36aaaaea37c20062c3e4f9c9cc9b5fb039a810ddfa644b9a40f124dd6d6dba.
Uses /Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs; desktop copy
is incomplete and intentionally NOT used for the batch. External volume must stay
mounted. Preflight records data/qwen_direct/full-asset-preflight.json and full-assets.json.

Expected terminal shape (illustrative FAKE-worker numbers, not real results):
```
[------------------------] 5/7805 0% | 40s | ETA ... | True 2/5 = 40.00%
[completed sample] {"scene":"scene_fake","csv_row":4,"pred":null,"GT":7,"correct_this":false,"strict":2,"tentative":1,"abstained":1,"failed":1,"BRT":{"B_used":1,"R_used":3,"T_used":2},"http_responses_known":5,"unknown_usage_calls":0,...}
{"event":"sample_stage","stage":"request_sent","B":1,"R":3,"T":2,"calls":4,...}
```
JSON details preserve raw protocol flags separately from strict/tentative/null/error
counters. Accuracy denominator includes every COMPLETED saved failure/null, no
unfinished sample. BatchProgress emits live completed fraction/elapsed/ETA; every
completion and stage flushes. runner.log tees terminal safely inside Python (no
shell tee that could break parent on Ctrl+C); progress.jsonl flushed/fsynced and
atomic results.json/summary.json provide tail/read snapshots. Per-case console.log
and episode/live-progress.json retain phase heartbeats. Original003 result/trace/
frozen code unchanged; no actual API/full-run startup during preparation/testing.


## v2 correction: actual worker dependencies, not just fake-worker readiness

v1 is preserved as an ENGINEERING-FAILURE run:71saved samples/evaluations, all
PackageNotFoundError for spacy, B/R/T/API0. No prediction was produced. This0/71
is NOT model quality. Earlier61was an earlier snapshot; original records and
scored denominator are NOT deleted, relabeled in-place or silently skipped.
The validation gap was ours: fake workers and prepare-only did not run the real
parser startup. The launcher used base Conda Python rather than003's .venv;
symlink-target equality does NOT preserve pyvenv.cfg/site-packages. Base Python
has neither spacy nor en_core_web_sm.003's existing virtualenv has spacy3.8.16
and en_core_web_sm3.8.0; both remain necessary. No package exception swallowed,
no syntax-backend/model/Monitor/strict/tentative policy changed, no pip install.

The v2 correction made the ABSOLUTE launcher select the exact003 virtualenv executable WITHOUT
resolving it to base Python; sys.prefix is frozen. That new output was v2, NEVER v1resume.
Every launch/prepare-only runs the exact newly frozen real sample worker with
--bootstrap-check, same cleaned environment/read-only inputs, and no key read.
Network and requests are explicitly denied in that child. It loads real point
cloud/category/geometry, loads actual spaCy model and analyzes original query,
builds/validates the actual parser provider Schema, then intentionally exits at
FIRST API BOUNDARY before a request receipt or HTTP attempt. Success requires
ready marker AND bootstrap_ready terminal AND zero calls. Failure blocks all
scheduling and prints real exception/log. No annotation target is sent to child.
Each offline attempt is archived under offline-bootstrap/, not a scored qNNNNN.

First zero-request fatal sample result pauses scheduling immediately, preserves
that failure in the completed denominator, prints its type/detail. Reinvoking
a run with such a saved global startup failure also pauses; no silent skip.
Fixes require fresh frozen output. Zero-request failures and unknown remote
execution remain distinct. v1raw artifacts are hash-inventoried outside v1.

Offline-only preparation via the EXACT user launcher:
```sh
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_direct_full.command --prepare-only
```
Without --prepare-only the user starts full inference. We did NOT execute it.
The prior complete130scene audit remains valid; shared input checks are reused,
no new export/classification or full data audit is required by this dependency fix.

v2 also freezes known-gt-exposure.json:71v1parent-scored identities (including the
previous003development sample), no target values. Each matching v2evaluation is
marked previously_development_sample=true; full run remains development exploration.
This provenance does not change inference candidates/prompts or skip any sample.

## v3 correction: near-repeat acquisition and explicit replanning

The same launcher now defaults to a fresh v3 output. It does not resume v2 or
resend q00171's interrupted request. V2 has19completed evaluations and one
interrupted episode; all remain separate, including the sent-only receipt with
unknown remote execution/usage. The v3 run starts from the original sanitized
selection with fresh per-query state, no v2 DAG, prediction or budget import and
no score mixing. Existing71historically GT-exposed identities already cover all
19v2evaluated identities; q00171 itself was already historically exposed in v1.
No annotations were decoded for this correction or its replay. This remains a
development exploration, never independent generalization or an official score.

Historical normalized coverage replaces raw observation hashing. Fine camera,
pixel and receipt changes cannot perpetually reset stagnation; the same Qwen
tentative exit remains once plus at most one structural repair. Explicit compact
Monitor schemas and instructions explain completing supported old bindings and
selecting a ready pending tool task in one response, or replacing the plan when
its available tools are wrong. Native Monitor rules and evidence requirements
are unchanged. V3 source, regression fixture and protocol notes freeze together.

115affected offline tests passed. Both initial and delta/replacement production
wire schemas passed local checks using archived actual images; this is not
remote-provider acceptance. Run --prepare-only for the actual frozen worker's
network-denied bootstrap (parser boundary, HTTP0); only the user starts inference.
