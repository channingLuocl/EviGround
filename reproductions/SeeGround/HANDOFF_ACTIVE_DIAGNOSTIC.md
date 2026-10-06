# SeeGround Active Diagnostic Handoff

## Goal

The current objective is to improve real-scene Nr3D grounding. Category matching creates the initial target candidates. The model then chooses the next evidence dimension from the query and current candidate differences. Python tools produce auditable evidence and update the live candidate set. The loop continues until one candidate remains or records insufficient evidence.

JEV is out of scope for this phase. Prioritize candidate disambiguation and operation quality.

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

The model may return `compute` with one of `viewpoint`, `position`, `distance`, `appearance`, or `size`; `render` with focus IDs and camera parameters; `visual_check` with `yes/no/uncertain` per rendered candidate; or `finish` with a target ID.

Rules:

- `compute position` returns candidate-level `pass/fail/unknown` relation evidence.
- `compute distance` does the same for explicit `nearest`/`farthest` rankings.
- Only a definite `fail` across all available reference interpretations removes a candidate.
- Passing or unknown alternative interpretations preserve a candidate.
- `compute size` exposes soft AABB evidence and must not turn `skinny`, `small`, or `large` into an unstated hard threshold.
- `compute appearance` exposes shape metrics and visual needs; color/material evidence still requires a rendered view.
- `render` accepts at most 6 focus IDs. Prefer a few target alternatives plus specific references.
- `visual_check=no` can exclude a candidate. `yes` does not prove all other candidates are wrong. `uncertain` preserves the candidate.
- Failed or repeated actions are recorded in `active_trace` and do not consume successful tool budget. Controller feedback is added to the next prompt.
- A final choice outside the live candidate set is rejected. An empty live set falls back to the initial category set but is not accepted as resolved.

## Implementation

`active_diagnostic.py` contains `position_candidate_evidence()`, `distance_candidate_evidence()`, `apply_computed_evidence()`, `compute_observation()`, `render_action()`, `_check_payload()`, and `active_ground()`.

`diagnose()` remains the deterministic audit function. Its category verdict logic respects an explicit active visual `no` even when annotated GT categories are trusted.

## Tests

Run from `reproductions/SeeGround`:

```bash
PYTHONPATH=local python -m unittest discover -s local -p 'test*.py'
python -m py_compile local/active_diagnostic.py local/test_active_diagnostic.py
git diff --check
```

The current local suite passes **130 tests**. Active tests cover candidate-level statuses, conservative removal, repeated actions, invalid render recovery, focus limits, visual exclusion/preservation, and selectable dimensions.

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

Do not add JEV or change the final benchmark protocol until active operation semantics and traces are stable.
