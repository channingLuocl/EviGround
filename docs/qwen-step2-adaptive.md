# Step2 overflow-adaptive revision, 2026-11-20

Local branch20261120_step2, unchanged Step1 Git base
8a44a23897687d8fb35a766b71bf67c008767940. No commit/push, model change, extra
model/agent, B/R/T limit, new GPU service or live model request is authorized in
this preparation. Existing v3 and initial step2-smoke source/results stay intact.
The initial Step2 rules/assumptions remain in qwen-step2-context.md. This revision
adds actual handling for overflow; it does not establish all-sample feasibility.

## Adaptive final-payload handling

1. Build and count the normal compact payload, exact generation schema, actual
   image bytes and output reservation. If its engineering upper bound is
   **<32768**, return that payload unchanged. No tighter metadata, resize or
   additional model call occurs. The initial-Monitor schema correction below is
   a separately declared protocol/interface change, not an adaptive transform.
2. Only if oversized, try reversible metadata factoring: exact repeated object
   score-type defaults and constant table columns; parse the previous invalid
   JSON proposal and reference its exact error values by field path instead of
   quoting them again. Preserve the latest actionable validator/rule feedback,
   full previous proposal values, original utterance, DAG, active/allowed IDs,
   category/reference scope and ALL acquired pair/frame/group measurements.
   Assert exact round-trip restoration before considering the candidate. Raw
   proposal text and full errors remain in the original local journal. This
   encoding changes representation, not raw numerical values or conclusions.
3. Recount that complete final candidate. If still oversized, try send-side
   image maximum dimensions896,768,640,512, in that order, always preserving
   aspect ratio and ALL image views. Each attempt starts from original bytes;
   it never repeatedly resamples the previous reduced result or enlarges small
   images. PIL LANCZOS -> derived PNG is **lossy spatial downsampling**, not
   lossless evidence compression. Take the first fitting attempted package;
   if no attempted package fits, retain the smallest bound as an explicit refusal.
4. Every attempt is counted independently including its actual image dimensions,
   newly added resolution warning/table, schema, framing allowance and unchanged
   role output envelope. Strict boundary32768 is rejected. Attempts are CPU data
   transformations, not additional reasoning consultations. One dispatcher call
   is charged as before; all attempts are recorded in context-packing.json.
5. The existing strict send gate, returned provider-usage checks, capacity terminal
   types, batch pause/no retry/no skip and separate prediction freeze still apply.

No primary measurement, directed pair, comparison member, observer frame, raw
sign, precision or order is removed. No program target/binding/exclusion/node
completion or evidence-sufficiency decision is introduced. Current native state
and Monitor are not changed by adaptive payload transforms.

## What the model actually sees

The base view's hashes/visible-pixel counters remain ORIGINAL-render metadata.
When images are reduced, an explicit model-visible table gives each attachment's
original and sent dimensions. A warning states that ONLY sent-resolution pixels
are seen: archived original detail/material/absence cannot be asserted from IDs,
hashes or original pixel counters. Preserve uncertainty, obtain discriminating
evidence or abstain if critical detail is unreadable. No crop, hidden view drop
or pretense that full-resolution pixels were delivered is allowed.

The journal stores original-to-sent SHA-256, original/sent dimensions, separate
x/y scale factors, byte counts and lossy flags. Runtime receipts archive each
unique original input image alongside the actual sent image; identical bytes
may share a file. Offline replay keeps original files read-only in their existing
source and writes actual derived images under its own new output for inspection.
All image views remain represented, but downsampled images are NOT the original
pixel evidence. Readability and model semantic success are **not verified** by
capacity rescue. The512 floor is for the maximum side; short dimensions and
individual montage tiles may be much smaller. No lower maximum side is attempted. Remaining
text-heavy cases are explicit counterexamples, not silently clipped successes.

The private tokenizer, schema linearization/framing and vision contract remain
unproven; this is the same conservative local engineering gate plus returned
usage audit, not an absolute provider-count guarantee. Model qwen3.8-max,
role output limits1600/4096/3072 plus10 cushion, raw native semantic validation,
record-only B/R/T, rollback/annealing and same-Qwen tentative protocol are unchanged.

## Paired offline results

Use the exact163 original request/receipt identities and SHA values from the
initial final-replay report; do not expand the roster as v3 continues. No target
annotations are read. Historical GT-exposed identity flags are unchanged.

- Before:136fit,27refused. After:153fit,10refused,0packaging failures.
- All136 originally fitting requests remain initial-stage fits; the adaptive
  layer does not resize/repack them. The separate initial-schema fix is explicit.
-17 actual adaptive rescues:1metadata-only and16using traced derived images.
  These are capacity rescues only; no new model response or accuracy is claimed.
-Q00171 original six paths remain below the local limit, including the last
  repair32748. Tentative first/one-repair offline previews remain29720/29931;
  their actual routine executes without changing the native Monitor snapshot.

Selected real requests (final bound INCLUDES output reserve):

| Case/call | Before | Adaptive | Selected stage |
|---|---:|---:|---|
| q00021/006 |34197|32519|max-side640|
| q00021/007 |34796|32542|max-side512|
| q00030/006 |33814|32136|max-side640|
| q00173/006 |32900|31204|max-side896|
| q00173/007 |36105|32601|max-side640|
| q00184/009 |32994|31287|metadata only|

All10 remaining refusals, after actual final-payload counting at the minimum
attempted512 maximum image side:

| Case/call | Final bound |
|---|---:|
| q00021/008 |34496|
| q00021/009 |34463|
| q00021/010 |34464|
| q00047/007 |33267|
| q00048/005 |33966|
| q00048/006 |35169|
| q00048/007 |36515|
| q00172/006 |33505|
| q00173/008 |33370|
| q00187/007 |33153|

These retain long raw text/measurements/actionable proposal feedback plus all
image views and output reserve. Each report gives the text/image/framing/output
breakdown. No numeric or semantic evidence cut is used to force a pass. This
does not prove that no smaller semantically sufficient packet could exist;
model-directed retrieval is a possible later investigation, not implemented here.

## Separate initial-smoke structural failure

The user-run initial step2-smoke/q00171 has4HTTP200 calls, all actual
prompt+reserved-output below32768. It ended B0/R6/T1: three first-action responses
wrapped monitor in replace_plan before any DAG existed. This is a STRUCTURAL
failure, not capacity refusal or proof of semantic infeasibility. The real plan
also assigned finish to a non-verification node; merely stripping its wrapper
would not make it native-valid. Old requests/responses/results remain unchanged;
no new GT is read.

The generation schema now REQUIRES first monitor's six bare top-level fields:
nodes,verification,requirement_nodes,current_node_id,reopen_node_ids,revision_reason;
extra replace_plan/node_updates wrappers are forbidden. It masks non-verification
finish tools and requires verification n11/finish. The first-turn textual contract
explicitly forbids wrappers. Later delta/replace_plan alternatives remain, and
the full original local native Schema/Monitor still enforces the actual semantics.
The program never unwraps/repairs a model action. A frozen fixture records the
actual invalid wrapper and a prior valid real bare Monitor; offline tests reject
the former and accept the latter at the generation-schema boundary. This is an
interface regression check, not a new real first-turn success claim.

## Reproduce and start manually

Paired report/attempts/derived pixels are under
outputs/qwen-direct/step2-adaptive-validation-20261120/replay/.
The separate initial-smoke-structure-audit.json records the old genuine failure.
149 targeted offline tests pass, including the resized32767/32768 boundary,
unchanged first-fit payload, exact pair/frame round trips, original-byte archive,
zero sends after all attempts fail, tentative preservation and real wrapper shape.

```sh
/Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python scripts/check_qwen_step2_context.py --all-saved
/Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python -m pytest -q tests/unit/test_qwen_step2_context.py tests/unit/test_qwen_direct_fresh.py tests/unit/test_qwen_direct_batch.py tests/unit/test_dag_delta.py tests/unit/test_dag_monitor.py tests/unit/test_policy_annealing.py tests/unit/test_prompt_tables.py
```

NEW dedicated entry, default only a fresh row171 task:

```sh
# Offline dependency/parser/schema/final-payload boundary check, HTTP0
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_step2_adaptive_smoke.command --prepare-only
# USER ALONE starts the new genuine diagnostic task
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_step2_adaptive_smoke.command
```

New output: outputs/qwen-direct/step2-adaptive-smoke-20261120/.
Do not resume/overwrite the initial step2-smoke-20261120 directory. Original
source/assets/runtime are frozen separately; no old DAG/answer/prediction is
inherited. No all-data run, additional API agent, commit or remote push occurs.
New real provider acceptance, image readability and semantic correctness remain
for user-controlled validation; report flow, evidence support and correctness
separately after prediction freeze.
