# V3 local loop correction, 2026-11-20

This is a new frozen diagnostic candidate for user-controlled restart. No live
model request or full run is authorized during this preparation. The remote GPU
remains shut down; all work is local to codex/qwen-direct-fresh-20261120.

## Observed cause

V2 q00171 (`The lamp not next to the beds.` including the original stored quote
marks) was interrupted with B64/R81/T66:64inspect actions and one rejected
measure.74API receipts exist:73completed and one sent-only. The last request's
remote execution and usage remain unknown, never zero and never retried here.
Its accepted DAG/candidate set stayed unchanged from T02 through T66; later
isolated views changed azimuth by one degree. Raw observation hashing mistakenly
counted new receipt IDs, camera/pixel metadata as progress every time.

Native Monitor already allows completing supported old tasks and selecting a
different ready pending tool task, and permits model-explicit replace_plan. It
was not an impossible protocol. The particular accepted graph only offered an
inspect binding task and finish verification. The model needed to explicitly
replan to permit measurement or add the appropriate task. The previous generic
provider Monitor object omitted exact replacement structure and did not make
the transition distinction clear enough.

## Minimal change

Progress now tracks historical accepted states and acquisition coverage, with
sorted ID groups, context,15-degree camera bins, newly visible instance IDs and
raw measurement facts/frames. Replaying A/B views/groups, swapping ID order,
changing receipt/node IDs, pixel counts or file paths cannot keep resetting it.
Angles wrap at360. A new context, unvisited coarse view, newly visible object,
measurement/comparison/frame or accepted state resets stagnation. Counts of
new coverage and original evidence IDs are logged per decision. This15-degree
coverage rule is an engineering proxy, not proof of information gain; some
useful within-bin changes can be missed, and coarse new views may still add no
semantic evidence. It never declares a node completed or evidence sufficient.

After three consecutive no-progress transitions, the existing same-Qwen
tentative evaluation runs once, with at most one structural repair. The exact
repeated-native-rejection trigger still works. No B/R/T, local32K or cumulative
API cap was added. All native evidence, rollback, category and uniqueness rules
remain unchanged on the strict path. Tentative predictions remain distinct,
accepted=false and strict_monitor_accepted=false; the program selects no target.

Qwen now sees exact compact initial or delta/replace_plan field schemas. A single
visible contract explains that allowed_actions limits the next tool at that node,
not which evidence can complete a supported binding. The model supplies all
bindings, conclusions, dependencies, permitted tools and requirement mappings.
Delta cannot change task definitions; an explicit complete replacement can.
The original full action Schema and native Monitor still validate every response.

## Verification and provenance

-115affected tests passed: direct runner/batch, DAG delta/Monitor, annealing and
  prompt tables. The original exact-rejection tentative regression passes.
-Actual66saved full states and a no-GT acquisition projection both replay to
  first stagnation trigger T26. This says what v3 would detect on the existing
  trajectory, not what a new real Qwen episode would do or predict.
-Initial and delta/replacement production choose schemas locally validate with
  actual archived image bytes (2and3images), with HTTP0. Local provider-wire
  checks and a network-denied real-worker parser bootstrap do not prove remote
  provider acceptance. No new real Qwen response is claimed.
-Fixture contains expanded acquisition metadata, accepted plan changes and
  SHA-256 of original state/history files; no target annotation or prediction.
  This work reads no GT. Existing provenance already marks q00171 historically
  GT-exposed in v1, so it is not an unseen independent sample.
-V2 source/results/receipts remain untouched. The external v3-local-validation
  directory records829before-hashes, raw/full-state replay and wire checks.
  Fresh v3 source hashes, input hashes and interpreter prefix freeze together;
  no old outputs are an inference dependency. Outputs/scores are not mixed.

Reproduce the115checks with the existing exact virtualenv Python executable:

```sh
/Users/luocl/.codex/worktrees/laya-qwen-fresh-20261119/EviGround/.venv/bin/python -m pytest -q tests/unit/test_qwen_direct_fresh.py tests/unit/test_qwen_direct_batch.py tests/unit/test_dag_delta.py tests/unit/test_dag_monitor.py tests/unit/test_policy_annealing.py tests/unit/test_prompt_tables.py
```

The unchanged absolute user launcher now points to full-nr3d-20261120-v3.
Its --prepare-only mode freezes and performs an isolated real-worker bootstrap,
with socket/HTTP disabled and no key read; it schedules no scored sample. The
user alone runs the launcher without that option to start the new experiment.
Remote provider compatibility and real semantic behavior still need that run.
