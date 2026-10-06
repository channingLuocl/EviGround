# SeeGround Active Diagnostic Handoff

## Goal

The current objective is to improve real-scene Nr3D grounding. Category matching creates the initial target candidates. The model then chooses the next evidence dimension from the query and current candidate differences. Python tools produce auditable evidence and update the live candidate set. The loop continues until one candidate remains or records insufficient evidence.

The outer ReAct planner now defaults to JEV with text/JSON state only. Qwen remains the query parser and internal visual verifier. Individual evidence algorithms are the next work.

The intended bounded ReAct loop is:

```text
category candidates -> model action -> Python tool -> observation and candidate state -> next action or finish
```

Private chain-of-thought is not stored. The model returns a short action reason; the trace stores tool inputs, observations, candidate updates, and stop status.

## Repository State

- Branch: `qwen-vl-baseline`
- Latest implementation commit: `25ec80d Guide model away from repeated evidence actions`
- Main implementation: `reproductions/SeeGround/local/active_diagnostic.py`
- Integration: `reproductions/SeeGround/local/real_scene.py`
- Batch integration: `reproductions/SeeGround/local/batch_nr3d.py`
- Tests: `reproductions/SeeGround/local/test_active_diagnostic.py`
- Local documentation: `reproductions/SeeGround/local/README_本地运行.md`

The worktree contains unrelated untracked files under `viewpoint_exploration/`. Do not add or delete them while continuing this task.

## Action Protocol

The model returns `evidence` with one of `position`, `distance`, `size`, `appearance`, or `viewpoint`; `finish` with a target ID; or `stop` with a reason. For example: `{"action":"evidence","dimension":"position","reason":"check reference relations"}`. Computation, rendering, and visual verification are internal steps. Legacy outer actions are rejected.

Rules:

- `evidence position` returns candidate-level `pass/fail/unknown` relation evidence.
- `evidence distance` does the same for explicit `nearest`/`farthest` rankings.
- Only a definite `fail` across all available reference interpretations removes a candidate.
- Passing or unknown alternative interpretations preserve a candidate.
- `evidence size` exposes soft AABB evidence and must not turn `skinny`, `small`, or `large` into an unstated hard threshold. Size and viewpoint summaries do not certify candidates.
- `evidence appearance` exposes shape metrics and visual needs, renders one group, and performs one internal visual verification. It accepts at most 6 known focus IDs including a live target, with optional camera/context parameters. Defaults are the first 6 live targets, azimuth 90, elevation 25, scene context.
- A visual `no` can exclude a marked live candidate. `yes` does not prove all other candidates are wrong. `uncertain` preserves the candidate. Verification receives only target-bound attributes.
- All operations return a shared envelope with coverage, missing evidence, internal steps, and optional next evidence suggestions. The controller appends candidate updates and the current evidence state.
- Invalid parameters and identical requests on unchanged evidence do not consume evidence budget. Relevant state changes or different appearance groups/views permit rechecking. Visual verification failures retain unknown candidates but consume the completed evidence operation; all internal calls are recorded.
- `--review-limit` counts evidence operations, with at most one render and one internal visual call per appearance operation. `internal_usage` records internal steps separately; total API usage includes internal verification.
- While evidence budget remains, finish requires resolved evidence and a unique live target; otherwise the controller requests further evidence or stop. A budget-exhausted guess remains unaccepted when unresolved.
- A final choice outside the live candidate set is rejected. An empty live set falls back to the initial category set but is not accepted as resolved.

## Implementation

`active_diagnostic.py` contains the public validator `evidence_request()` and executor `execute_evidence()`, with the existing computation/render helpers inside. `active_ground()` retains the outer ReAct loop. Query parsing/rewriting and initial audit remain unchanged; individual evidence algorithms are the next work.

`react_decision.py` adapts the existing TypeSafe Choice API to next-task selection and a separate final target Choice. Appearance options cover groups of at most six live targets with scene/isolated context and inspection azimuths 90/180. These are a bounded inspection menu, not recovered speaker views. JEV task reasons are selected option descriptions, not generated model explanations. Raw answers, probabilities, warnings, usage and model names remain in call records; confidence never certifies grounding.

Set `SEEGROUND_REACT_BACKEND=jev`, `TYPESAFE_API_KEY`, and `SEEGROUND_JEV_MODEL` for JEV; the existing `SEEGROUND_*` Qwen settings serve parsing and visual checks. `SEEGROUND_REACT_BACKEND=qwen` retains a text-only planner comparison. Missing JEV configuration fails before parsing, without fallback. Batch manifests bind the planner backend/model and adapter code to prevent cache reuse across configurations.

`diagnose()` remains the deterministic audit function. Its category verdict logic respects an explicit active visual `no` even when annotated GT categories are trusted.

## Tests

Run from `reproductions/SeeGround`:

```bash
PYTHONPATH=local python -m unittest discover -s local -p 'test*.py'
python -m py_compile local/active_diagnostic.py local/test_active_diagnostic.py
git diff --check
```

Active tests cover candidate-level statuses, conservative removal, repeated actions, invalid request recovery, focus limits, visual exclusion/preservation, internal call failures, target attribute scope, and selectable dimensions. A `real_scene.run_query()` integration test uses an actual CPU render with mocked model responses to verify image encoding, query rewriting, GT table integration, API accounting, and saved traces. These tests do not measure real-model grounding accuracy.

## Server Smoke Test

The server already has prepared `scene0149_00` files. Use a new output directory:

```bash
cd ~/paper
git pull --ff-only
cd reproductions/SeeGround

.venv/bin/python local/real_scene.py \
  --scene-dir local_outputs/scene0149_00 \
  --query 'Choose the skinny cabinet between the stove and the wall.' \
  --mode active_diagnostic \
  --object-table gt \
  --review-limit 8 \
  --run-name active_between_07
```

Inspect `local_outputs/scene0149_00/active_between_07/prediction.json`, especially `active_trace.steps`, `observation.candidate_evidence`, `observation.candidate_update`, `remaining_candidate_ids`, `decision_status`, and `stop_reason`.

For a compact trace dump, load `prediction.json`, print `predicted_id`, `decision_status`, `stop_reason`, `category_candidates`, and `final_candidate_ids`, then iterate through `active_trace.steps` and print each action, observation tool, failure flag, remaining candidates, candidate updates, candidate evidence, and visual checks. The earlier conversation contains a complete shell snippet for this inspection.

## Development Smoke-Test History

Development query:

```text
Choose the skinny cabinet between the stove and the wall.
```

With GT categories, the initial candidates were:

```text
[2, 3, 4, 8, 9, 10, 21, 22, 25]
```

- `active_between_01`: predicted 2; missing; many candidates remained.
- `active_between_02`: predicted 3; missing; a side render confirmed 3 visually but several candidates remained uncertain.
- `active_between_03`: predicted 4; missing; invalid renders and a partial visual check.
- `active_between_04` and `active_between_05`: failed because repeated actions were fatal.
- `active_between_06`: predicted 10; missing; position removed no candidates, then the model repeated `compute appearance` until finishing.

These are development smoke tests, not benchmark results. Do not infer accuracy from them.

## Known Limitations

1. `between` can have multiple wall/reference interpretations. A candidate is retained when at least one interpretation passes or is unknown.
2. Wall geometry is a finite planar proxy fitted from scene points, not a complete semantic reconstruction.
3. AABB shape metrics are not intrinsic object dimensions. `skinny` remains soft evidence.
4. Point-cloud renders can be occluded or low resolution. `uncertain` must not become `no`.
5. The model may still ignore action constraints or repeat an action. The latest code adds explicit controller feedback; verify this with the next smoke test.
6. A `model_finish` with multiple remaining candidates correctly produces `decision_status=missing`.

## Recommended Next Work

1. Verify `compute position`: candidate statuses must match stored `between_triplets`; definite failures should be removed.
2. Verify `compute appearance`: shape differences should be visible without claiming color/material certification.
3. Verify `render`: focus sets should be small, relevant, and use valid camera angles.
4. Verify `visual_check`: only explicit `no` removes candidates; `uncertain` remains.
5. Verify the model moves to an unused dimension after duplicate-action correction.
6. Only after the single-query trace is coherent, run a small multi-query smoke test. Do not call it a score improvement without a fixed held-out plan.

The user has authorized JEV as the outer planner. Keep the evidence/state semantics and benchmark protocol stable while testing the new model routing.
