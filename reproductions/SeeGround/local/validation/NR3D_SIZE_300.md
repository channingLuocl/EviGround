# NR3D Size Validation: 300 Fresh Utterances

The fixed manifest is `nr3d_size_300.json`: 300 unique descriptions from 92
ScanNet validation scenes and 38 target categories. The six scenes used during
earlier debugging are excluded. Selection uses seed 20261006, correct human
listener guesses, size-language near the target category, available GT/pred
OLT files, and valid stimulus instance IDs. Scene round-robin limits dominance
by one scene. Model predictions and agreement with geometric proxies were not
used to select examples.

The original NR3D utterances are preserved. Lexical target binding is approximate;
some size descriptors may still refer to a reference object. These cases need
scope review when interpreting failures. The script runs against the full scene
OLT rather than restricting inference to the NR3D stimulus candidates.

There are 17 strict simple size requests (14 relative adjectives/comparatives and 3
superlatives), followed by 283 compound descriptions. Across the whole selection,
125 rows use ordinary adjectives, 127 contain a comparative, and 48 contain a
superlative without a comparative. These tags are descriptive, not a guarantee
that all language forms are implemented. This is a diagnostic sample with oracle
categories, not a complete NR3D benchmark score. Keep accepted wrong answers in
the results.

## Prepare

From `reproductions/SeeGround`:

```bash
.venv/bin/python local/prepare_size_validation.py \
  --manifest local/validation/nr3d_size_300.json
```

Downloads four ScanNet files per scene, reuses existing downloads/prepared scenes,
checks point-cloud / OLT alignment and verifies stimulus IDs. No inference API
calls are made. All 92 scenes must be available for a complete run.

## Run and Resume

```bash
.venv/bin/python local/run_size_validation.py \
  --output local_outputs/nr3d_size_300_v1 --workers 4 --limit 30
```

The first 30 contain all 17 simple requests plus 13 compound descriptions. Then
continue in the same output directory with the unchanged code/settings:

```bash
.venv/bin/python local/run_size_validation.py \
  --output local_outputs/nr3d_size_300_v1 --workers 4
```

Each case runs in its own process with only the original query, scene, and CSV
row provenance. Target and stimulus IDs are used only for evaluation. Completed
case evaluations are reused. API/process failures count as incorrect; add
`--retry-errors` to retry those failed cases. Missing/ambiguous model decisions
are completed evaluations and are retained. A new code/model/review-budget
configuration requires a fresh output directory.

`results.json` is updated after every completion. It includes overall and grouped
accuracy, accepted accuracy, accepted wrong count, error count, decision statuses,
and per-case logs, predictions, stop reasons and missing conditions. The full
selection and code hashes are saved in `manifest.json`.
