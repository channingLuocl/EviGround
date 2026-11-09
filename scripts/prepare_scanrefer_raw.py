"""Collect the ScanRefer validation meshes and evaluation-only GT on KESU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData
import requests


BASE = "https://kaldir.vc.in.tum.de/scannet/v2/scans"
SUFFIXES = (".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json",
            ".aggregation.json")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare_scene(scene, targets, args):
    directory = args.output / "raw_scans" / scene
    directory.mkdir(parents=True, exist_ok=True)
    sources = []
    for suffix in SUFFIXES:
        name = scene + suffix
        path = directory / name
        url = f"{BASE}/{scene}/{name}"
        candidates = [args.cache / name, args.scannet / scene / name]
        cached = next((p for p in candidates if p.is_file()), None)
        if not path.exists():
            if shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise ValueError("Less than 2 GiB free on data disk")
            temporary = path.with_name(path.name + ".part")
            if cached:
                shutil.copyfile(cached, temporary)
            else:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
            temporary.replace(path)
        sources.append({"file": name, "sha256": digest(path), "url": url,
                        "local_cache": str(cached) if cached else None,
                        "bytes": path.stat().st_size})
    vertices = PlyData.read(directory / (scene + "_vh_clean_2.ply"))["vertex"]
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    segments = np.asarray(json.loads((directory / (scene + SUFFIXES[2])).read_text())["segIndices"])
    if len(xyz) != len(segments) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid mesh/segments: {scene}")
    line = next(line for line in (directory / (scene + ".txt")).read_text().splitlines()
                if line.startswith("axisAlignment"))
    matrix = np.asarray(list(map(float, line.split("=", 1)[1].split()))).reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid axis alignment: {scene}")
    aligned = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3]
    groups = json.loads((directory / (scene + SUFFIXES[3])).read_text())["segGroups"]
    objects, seen = [], set()
    for group in groups:
        object_id = int(group["objectId"])
        if object_id in seen:
            raise ValueError(f"Duplicate GT object: {scene}/{object_id}")
        seen.add(object_id)
        points = aligned[np.isin(segments, group["segments"])]
        if not len(points):
            continue
        low, high = points.min(0), points.max(0)
        objects.append({"bbox_id": object_id, "target": group["label"],
                        "bbox_3d": np.r_[(low + high) / 2, high - low].tolist()})
    missing = targets - {o["bbox_id"] for o in objects}
    if missing:
        raise ValueError(f"Missing referenced GT objects: {scene}/{sorted(missing)}")
    gt_path = args.output / "evaluation_gt" / scene / "objects_gt.json"
    save(gt_path, objects)
    return {"scene": scene, "vertices": len(xyz), "gt_objects": len(objects),
            "referenced_gt_objects": len(targets), "raw_files": sources,
            "evaluation_gt_sha256": digest(gt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scannet", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["rows"]
    targets = {}
    for row in rows:
        scene = row["scene"]
        if Path(scene).name != scene or scene in (".", ".."):
            raise ValueError(f"Invalid scene ID: {scene}")
        targets.setdefault(scene, set()).add(row["target_id"])
    records, errors = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare_scene, scene, ids, args): scene
                   for scene, ids in sorted(targets.items())}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(json.dumps({"scene": scene, "completed": len(records),
                                  "total": len(targets)}), flush=True)
            except Exception as exc:
                errors.append({"scene": scene, "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps(errors[-1]), flush=True)
            save(args.output / "raw_inputs_status.json", {
                "complete": len(records) == len(targets), "completed_scenes": len(records),
                "total_scenes": len(targets), "errors": errors})
    save(args.output / "raw_inputs.json", {
        "complete": not errors, "manifest_sha256": digest(args.manifest),
        "queries": len(rows), "scenes": sorted(records, key=lambda r: r["scene"]),
        "errors": errors, "gt_usage": "evaluation only; never an inference proposal source"})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
