"""Freeze scene-disjoint diagnostic splits before new model inference."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


SEED = "eviground-scanrefer-r19-scene-split-v1"


def order(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def prepare(manifest, output):
    source = json.loads(manifest.read_text())
    scenes = defaultdict(list)
    for row in source["rows"]:
        scenes[row["scene"]].append(row)
    # These scenes were inspected in the interrupted baseline; never hold them out.
    observed = {"scene0011_00", "scene0015_00"}
    if not observed <= scenes.keys():
        raise ValueError("Previously inspected scenes are missing")
    dev_scenes = observed | set(sorted(scenes.keys() - observed, key=order)[:46])
    heldout_scenes = scenes.keys() - dev_scenes
    if dev_scenes & heldout_scenes:
        raise ValueError("Development/heldout scenes overlap")
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, selected in (("dev", dev_scenes), ("heldout", heldout_scenes)):
        rows = []
        for scene in sorted(selected, key=order):
            rows.extend(sorted(scenes[scene], key=lambda r: order(r["csv_row"]))[:2])
        result = {k: v for k, v in source.items() if k != "rows"}
        result.update({"selection_seed": SEED, "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                       "purpose": name, "selection_rule": "Two SHA256-ordered queries per scene; scene-disjoint; no outcome filtering",
                       "previously_inspected_scenes": sorted(observed),
                       "not_full_benchmark": True, "rows": rows})
        path = output / f"{name}{len(rows)}_manifest.json"
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to change frozen selection: {path}")
        if not path.exists():
            path.write_text(content)
        summaries[name] = {"queries": len(rows), "scenes": len(selected), "path": str(path)}
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
