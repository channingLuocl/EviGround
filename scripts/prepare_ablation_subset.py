"""Freeze a scene-stratified Nr3D subset without consulting model outcomes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def allocate(counts, total, minimum=0):
    keys = sorted(counts)
    if not minimum * len(keys) <= total <= sum(counts.values()):
        raise ValueError("Sample size cannot satisfy the requested coverage")
    ideal = {key: total * counts[key] / sum(counts.values()) for key in keys}
    quotas = {key: max(minimum, int(ideal[key])) for key in keys}
    while sum(quotas.values()) < total:
        key = max((k for k in keys if quotas[k] < counts[k]),
                  key=lambda k: (ideal[k] - quotas[k], k))
        quotas[key] += 1
    while sum(quotas.values()) > total:
        key = max((k for k in keys if quotas[k] > minimum),
                  key=lambda k: (quotas[k] - ideal[k], k))
        quotas[key] -= 1
    return quotas


def stratum(row):
    return ("easy" if row["easy"] else "hard") + ":" + (
        "dependent" if row["view_dep"] else "independent")


def prepare(source, output, size, seed):
    document = json.loads(source.read_text())
    rows = document["rows"]
    if len({row["csv_row"] for row in rows}) != len(rows):
        raise ValueError("Source query identifiers must be unique")
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene"]].append(row)
    quotas = allocate({s: len(r) for s, r in scenes.items()}, size, minimum=1)
    rng = random.Random(seed)
    selected = []
    for scene in sorted(scenes):
        groups = defaultdict(list)
        for row in scenes[scene]:
            groups[stratum(row)].append(row)
        group_quotas = allocate({g: len(r) for g, r in groups.items()}, quotas[scene])
        for group in sorted(groups):
            candidates = sorted(groups[group], key=lambda row: row["csv_row"])
            selected.extend(rng.sample(candidates, group_quotas[group]))
    selected.sort(key=lambda row: row["csv_row"])
    assert len(selected) == size
    assert len({row["csv_row"] for row in selected}) == size
    assert {row["scene"] for row in selected} == set(scenes)
    full_counts = Counter(stratum(row) for row in rows)
    subset_counts = Counter(stratum(row) for row in selected)
    sampling = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_queries": len(rows), "selected_queries": size, "seed": seed,
        "method": "proportional scene quotas with at least one query per scene; "
                  "proportional easy/hard x viewpoint quotas within each scene",
        "outcomes_used_for_selection": False,
        "source_scenes": len(scenes), "selected_scenes": len(quotas),
        "scene_quotas": quotas,
        "strata": {group: {"source": full_counts[group],
                           "selected": subset_counts[group],
                           "proportion_difference_pp": round(100 * (
                               subset_counts[group] / size - full_counts[group] / len(rows)), 4)}
                   for group in sorted(full_counts)},
    }
    result = {**document, "description": "Frozen Nr3D ablation subset",
              "sampling": sampling, "rows": selected}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError("Refusing to replace a different frozen subset")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(json.dumps({k: v for k, v in sampling.items() if k != "scene_quotas"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.size, args.seed)
