# SeeGround ScanRefer category stage

This directory runs object-category prediction only. It does not run visual
grounding, a Qwen model, rendering, language parsing, or classifier training.

SeeGround's ScanRefer preparation reads Mask3D `ins_labels` directly from an
existing NPZ file. That published export contains one category per instance,
not a full class-score vector. Here the released ScanNet200 validation Mask3D
checkpoint is rerun and its parent-query foreground probabilities are retained
to export Top-5. This Top-5 is a reproduction extension, not a number originally
reported by SeeGround. The authors' exact original proposal scores are not
available, so identical proposals or paper results are not guaranteed.

## Inputs and prediction

```bash
python baselines/seeground/prepare_categories.py \
  --root /path/to/ScanRefer --output /path/to/category-data

python baselines/seeground/scanrefer_categories.py \
  --data /path/to/category-data --source /path/to/Mask3D \
  --checkpoint /path/to/scannet200_val.ckpt --output /path/to/category-output
```

Use `--limit 2` and a separate output directory for a smoke run. GPU dependencies
are PyTorch, MinkowskiEngine, torch-scatter, pointnet2 CUDA operators, hydra-core,
numpy, scipy, and scikit-learn. Input preparation additionally needs plyfile.

For the PyTorch 2.8 / CUDA 12.8 environment used in this run, apply
`cuda128_compat.patch` inside the pinned MinkowskiEngine source before building.
It disables a conflicting NVTX profiling annotation, adds explicit Thrust
headers, and preserves the map's custom deleter while avoiding an ambiguous
smart-pointer conversion. It does not change model weights or numerical kernels.
The run records before/after source hashes in `CUDA_COMPATIBILITY.json`.

The inference files contain only raw xyz, RGB, and scene superpoints. GT class
strings and instance memberships are stored separately and read only for
evaluation. No GT information is used to select or rank categories.

The Mask3D validation recipe uses 150 queries, 2cm voxelization, segment masks,
DBSCAN with eps 0.95m and min_samples 1, and the top 750 query/class pairs.
Identical nonempty masks are collapsed by confidence before object evaluation.
SeeGround's lookup-table filter is also applied: proposals with Top-1 wall,
floor, ceiling, object, or objects are excluded. The remaining proposals retain
their model's original Top-5 ranking, including generic names if ranked lower.
The complete 200 foreground score vector is saved, including two unused heads;
category ranking uses the 198 trained object categories. Class 0 maps to chair
because the official training labels use `clamp(label_index - 2, min=0)` after
excluding wall and floor. The selected label must equal the original Top-1.

## Evaluation protocol

- Evaluation population: distinct `(scene_id, object_id)` targets referenced in
  `ScanRefer_filtered_val.json`, counted once regardless of description count.
- Association: class-blind, one-to-one predicted-mask / GT-mask IoU >= 0.5,
  maximizing the number of valid matches, then total IoU.
- Every target stays in the denominator; unmatched targets count as errors.
- Exact Top-1/Top-5 use the original ScanNet aggregation label, with no synonym
  or plural normalization. Results against the query's object name are saved
  separately, since these two label strings may differ.
- ScanRefer uses predicted proposals, so these metrics include detection and
  association failures. They should not be compared as the same pure
  GT-instance classification protocol as the earlier 3,413-object Nr3D result.

Outputs: `protocol.json`, per-scene JSON and compressed masks/class scores,
`object_results.json`, and `metrics.json`. Evaluation can be repeated on CPU
using the same arguments plus `--evaluate-only` (source/checkpoint not needed).

Focused CPU checks:

```bash
cd baselines/seeground
python -m unittest test_category_metrics -v
```

Pinned sources are listed in `UPSTREAM.json`. Mask3D commit:
`11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5`; checkpoint SHA-256:
`6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534`.
