# Step2: strict local 32K payload experiment, 2026-11-20

This file describes the initial fixed-packing revision and its preserved smoke
output. The current overflow-adaptive revision and NEW launcher are documented
in `qwen-step2-adaptive.md`; old frozen source/results are not updated in place.

Local branch `20261120_step2`, based on Step1 commit
`8a44a23897687d8fb35a766b71bf67c008767940`. This is an exploration, not an
all-sample pass or an absolute proof about the private provider tokenizer.
No real model request, remote service operation, commit or push is part of this
preparation. The running Step1 v3 uses its own unchanged frozen source/results.

## Executed contract

Every parser, parser repair, action, action repair, tentative evaluation/repair
and future call through the central dispatcher computes the FINAL transmitted
messages, generation schema, actual image bytes and output reservation before
HTTP. A bound **>=32768 rejects before send**, regardless of B/R/T. There is no
context-policy bypass or accumulated API-call ceiling. Original native state,
DAG, observations and action ledger stay local and unchanged by payload packing.
All model bindings, exclusions, completions, targets and rollback remain explicit.

The **engineering** bound is:

```
UTF8 bytes of actual text + serialized final response_format
+ 1024 framing base + 128/message + 16/content-part
+ sum(actual-image Qwen-style smart-resize patch counts + 2/image)
+ requested max_tokens + 10 output-overrun cushion
< 32768
```

Requested output limits changed explicitly: parser/parser repair<=1600,
action<=4096, action structural repair<=3072, tentative/repair<=3072. An existing
smaller parser reservation remains smaller. This is a concrete parameter change
from8192, not just a context compression claim. All163 historical completions in
the audited snapshot fit their new role envelope (maximum3265); that is not a
guarantee that new, differently prompted answers fit. Provider truncation or
invalid output is retained as failure, never concatenated into a fake success.

The existing `max_tokens` parameter is retained; it approximately limits output,
with thinking disabled. Official documentation also describes approximate
`max_completion_tokens` and an up-to10-token deviation for that newer parameter.
The explicit10 cushion here is an ENGINEERING assumption for this private route,
not evidence of an exact contractual10-token cap on its `max_tokens`. Returned
usage exceeding the reservation stops. No switch to an untested API parameter,
historical usage ratio or arbitrary byte discount is used to make requests fit.

The byte bound assumes tokenization does not expand original UTF8 bytes. The
framing allowance assumes limited private chat-template/Schema overhead; `$defs`
linearization and exact qwen3.8-max tokenization are not publicly specified for
this Token Plan endpoint. The vision bound assumes the documented Qwen-family
32-pixel-factor resizing applies, using a deliberately conservative512x512
minimum,2621440 maximum and two markers. `max_pixels` now resides beside
`image_url`, as documented, not inside its URL object. Static, supported local
images only; unsupported media/framing fields and animated/extreme-aspect inputs
fail closed. Original pixels are not resized by the packer. These assumptions
are recorded in each preflight and are NOT an absolute provider-count proof.

After a response, both `prompt_tokens + reserved output` and
`prompt_tokens + completion_tokens` must remain strictly below32768; prompt
usage must not exceed the preflight input bound, and output usage must be valid
and within the reserved envelope. Missing/violating usage stops with
`provider_capacity_violation`. Oversize final payloads use `capacity_rejected`.
Both save the full journal and pause batch scheduling, including on resume,
without retry, silent skip, alternative target, or budget reset. Packaging
errors also record the failed consultation before HTTP. A refusal is not a
successful terminal prediction; B/R/T accounting remains record-only.

## Model-visible information

- Every object ID and Top5 category alternative/rank remains available through
  a shared category vocabulary; classifier scores remain fallible priors.
- Active candidates, lexical possible references, model-bound/focused IDs keep
  full catalog geometry and gaps. Unrelated catalog geometry/gaps are explicitly
  archived, listed by object/field in `context-packing.json`; this is NOT a
  lossless whole-scene catalog. Native `measure` can obtain any known object's
  geometry and `inspect` can establish appearance/category. The program does
  not bind lexical matches or select targets. Unseen gaps cannot be cited.
- Every acquired raw measurement is atomized by original directed pair,
  comparison group and observer frame. Exact numeric values, row order, groups
  and reference/frame identities survive round-trip decoding; shared catalogs
  merge only identical raw atoms. No mixing frames, rounding, balancing or
  geometry-based target selection occurs. All actual acquired evidence remains.
- Every unique actual image remains attached; only byte-identical images share
  one attachment. Original view->attachment->SHA mapping is explicit. Image IDs
  and disk paths are never presented as seen pixels. No image quality reduction
  or program-selected semantic evidence pruning is used to force a fit.
- Full accepted current DAG/requirements, active targets, exact latest rejection
  and rollback are retained. Old action snapshots/repeated rejected reasoning
  stay in the original local ledger; the last executable action parameters are
  visible. Structural feedback excludes branches for a different explicitly
  proposed action/terminal shape and deduplicates exact path/validator/error
  records. The unchanged complete raw proposal/errors/schema paths are archived.
- Shared tables/strings compress the selected view without changing its values.
  System guidance is condensed. The provider generation schema is lighter
  (Monitor is an object plus a separate complete field contract), while the
  **original full native action Schema and Monitor** still validate locally.
  Initial/delta/replace_plan shapes, exact quotes, semantic evidence, dependency,
  exclusion, rollback and uniqueness requirements are not program-completed.
  This new presentation's model comprehension is not yet empirically verified.
- Tentative retains its dedicated same-Qwen, allowed-intersect-active/null,
  original evidence/main-rivals/uncertainty instructions; one evaluation plus
  at most one structural repair, no Monitor mutation or strict acceptance.

There is no extra agent or free reasoning. No semantic assertion is upgraded by
packing. Essential evidence that still cannot fit is rejected and saved; these
results do not establish that a smaller semantically sufficient package cannot
exist. An on-demand selector/retrieval protocol is not claimed implemented.

## Offline evidence

The audited Step1 v3 snapshot has163 HTTP200 requests:27parser,3parser repair,
133semantic (including locally invalid responses). **136fit,27do not fit** the
new local gate; no packaging failures. Original-payload byte bounds cover all
163 original provider prompt usages; the133 reported visual token counts equal
the image formula. This checks OLD payloads only. Newly packed payloads have
**no new provider usage**, so historical usage never certifies their actual count.

| Case/path | Final bound incl. output | Result |
|---|---:|---|
| q00171 parser |6571|fits|
| q00171 first action |23155|fits|
| q00171 first structural repair |26486|fits|
| q00171 T02 action |26509|fits|
| q00171 T03 action |30583|fits|
| q00171 last repair (old actual prompt33400) |32748|fits, only20 spare|
| q00030 T04, explicit selected frame/42pairs |33814|refuse|
| q00048 T03,56pairs/3conditional frames |36220|refuse|
| q00173 call8 |38244|refuse|
| q00021 calls8–10 |38016 /37983 /37984|refuse|
| q00171 constructed tentative first/repair |29720 /29931|fits|

The targeted set is9/15fit,6refusals, including every q00171 request. Tentative
previews execute the real dispatcher/tentative routine with the archived native
state and an explicitly synthetic structural-error message; no model answer or
prediction is fabricated. The native Monitor snapshot remains unchanged.

Report location: `outputs/qwen-direct/step2-local-validation-20261120/all-saved/`.
It includes source request/receipt SHA, final redacted payloads, per-call preflight,
selected/archived catalog scope, actual attachment hashes, exact raw measurement
coverage and explicit counterexamples. Replay reads requests/receipts/images/
native state only, never target annotations. The prior71 historically GT-exposed
identities still apply, including171; this is diagnostic development, not unseen
generalization. Do not mix Step1 scores or claim all163 passed.

## Reproduce and user-controlled smoke

Exact stable interpreter, no installs, GPU or remote service:

```sh
/Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python scripts/check_qwen_step2_context.py --all-saved
/Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python -m pytest -q tests/unit/test_qwen_step2_context.py tests/unit/test_qwen_direct_fresh.py tests/unit/test_qwen_direct_batch.py tests/unit/test_dag_delta.py tests/unit/test_dag_monitor.py tests/unit/test_policy_annealing.py tests/unit/test_prompt_tables.py
```

Dedicated entry, **default only csv_row171**, fresh model/Monitor state. No old
answers/DAG are imported. Explicit rows restrict diagnostic cases, not B/R/T/API
budgets. Source/input/runtime identities freeze to an independent output;
the parent reads its annotation only after freezing a terminal prediction/failure.

```sh
# Offline real-worker dependency/parser/schema/payload boundary check; HTTP0
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_step2_smoke.command --prepare-only
# USER ALONE runs this to request one new real diagnostic task
/Users/luocl/.codex/worktrees/qwen-direct-fresh-20261120/EviGround/scripts/run_qwen_step2_smoke.command
```

Output: `outputs/qwen-direct/step2-smoke-20261120/`. Default row171 is a known
diagnostic case. A new trajectory may differ, hit capacity, truncate output or
fail native/semantic validation. Inspect actual receipts/usage and evaluate flow,
evidence sufficiency and frozen-prediction correctness separately before any
further manually authorized small sample. No full run or new remote push occurs.

Official source references checked2026-11-20 (private endpoint details remain unknown):
https://help.aliyun.com/zh/model-studio/vision
https://help.aliyun.com/zh/model-studio/qwen-api-reference/
