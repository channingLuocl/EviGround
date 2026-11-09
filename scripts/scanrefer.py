"""Prepare ScanRefer query manifests and report fixed-denominator IoU accuracy."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


# ScanRefer/lib/dataset.py maps raw names into these ScanNet18 categories.
CLASSES = {
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "others",
}


def input_file(value):
    path = Path(value).expanduser()
    if str(path).startswith("/absolute/path/to/"):
        raise argparse.ArgumentTypeError(
            f"{path} 是示例占位路径，请替换为本机实际文件路径。"
        )
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"输入文件不存在：{path}")
    return path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    annotations = json.loads(args.annotations.read_text())
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Expected the nonempty official ScanRefer annotation list")
    with args.labels.open() as handle:
        labels = {
            row["raw_category"]: (
                row["nyu40class"] if row["nyu40class"] in CLASSES else "others"
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }
    objects, keys = defaultdict(dict), set()
    for item in annotations:
        scene, target = item["scene_id"], int(item["object_id"])
        key = (scene, target, str(item["ann_id"]))
        if key in keys:
            raise ValueError(f"Duplicate annotation: {key}")
        keys.add(key)
        if Path(scene).name != scene or scene in (".", ".."):
            raise ValueError(f"Invalid scene ID: {scene}")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ValueError(f"Empty description: {key}")
        name = " ".join(item["object_name"].split("_"))
        category = labels.get(name, "others")
        if target in objects[scene] and objects[scene][target] != category:
            raise ValueError(f"Conflicting categories for {scene}/{target}")
        objects[scene][target] = category
    counts = {scene: Counter(items.values()) for scene, items in objects.items()}
    rows = [
        {
            "csv_row": index,
            "dataset": "scanrefer",
            "scene": item["scene_id"],
            "target_id": int(item["object_id"]),
            "ann_id": str(item["ann_id"]),
            "query": item["description"],
            "unique": counts[item["scene_id"]][
                objects[item["scene_id"]][int(item["object_id"])]
            ] == 1,
        }
        for index, item in enumerate(annotations)
    ]
    manifest = {
        "dataset": "scanrefer",
        "unique_policy": "official ScanRefer ScanNet18 classes over distinct referred objects",
        "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "rows": rows,
    }
    if args.output.exists() and json.loads(args.output.read_text()) != manifest:
        raise ValueError("Manifest already exists with different inputs; choose another output")
    write_json(args.output, manifest)
    print(json.dumps({"queries": len(rows), "scenes": len(objects),
                      "unique": sum(r["unique"] for r in rows),
                      "multiple": sum(not r["unique"] for r in rows)}, indent=2))


def report(args):
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    expected = {r["csv_row"]: r for r in rows}
    if manifest.get("dataset") != "scanrefer" or not rows or len(expected) != len(rows):
        raise ValueError("Expected a nonempty, unique ScanRefer manifest")
    if any(type(r.get("unique")) is not bool for r in rows):
        raise ValueError("Each manifest row needs a boolean Unique/Multiple label")
    results = json.loads(args.results.read_text())["results"]
    observed = {}
    for result in results:
        index = result["csv_row"]
        if index in observed or index not in expected:
            raise ValueError(f"Duplicate or unselected query: {index}")
        if any(result.get(key) != expected[index][key]
               for key in ("scene", "query", "target_id")):
            raise ValueError(f"Result and manifest disagree: {index}")
        iou = result["iou_3d"]
        if isinstance(iou, (float, int)) and 1 < iou <= 1 + 1e-12:
            iou = 1.0  # Legacy floating-point results can exceed the physical bound by a few ulps.
        if (isinstance(iou, bool) or not isinstance(iou, (float, int))
                or not math.isfinite(iou) or not 0 <= iou <= 1):
            raise ValueError(f"Invalid 3D IoU for query {index}")
        # Errors and abstentions remain failures, even if stale geometry is present.
        observed[index] = 0 if result.get("error") or result.get("predicted_id") is None else iou

    def metrics(selection):
        total = len(selection)
        return {
            "total": total,
            "processed": sum(r["csv_row"] in observed for r in selection),
            "acc25": sum(observed.get(r["csv_row"], 0) >= 0.25 for r in selection) / total if total else None,
            "acc50": sum(observed.get(r["csv_row"], 0) >= 0.5 for r in selection) / total if total else None,
        }

    value = {
        "dataset": "scanrefer",
        "proposal_source_declared": args.proposals,
        "complete": len(observed) == len(rows),
        "pending": len(rows) - len(observed),
        "errors": sum(bool(r.get("error")) for r in results),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "unique": metrics([r for r in rows if r["unique"]]),
        "multiple": metrics([r for r in rows if not r["unique"]]),
        "overall": metrics(rows),
    }
    write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--annotations", type=input_file, required=True)
    prepare_parser.add_argument("--labels", type=input_file, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--manifest", type=input_file, required=True)
    report_parser.add_argument("--results", type=input_file, required=True)
    report_parser.add_argument("--proposals", choices=("mask3d", "gt", "other"), required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else report)(args)


if __name__ == "__main__":
    main()
