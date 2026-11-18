"""Adapt the verified classifier export without reading its evaluation_only payloads."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from eviground.categories.hypotheses import (
    VERSION,
    SCORE_TYPE,
    MASK3D_SCORE_TYPE,
    NATIVE_SCORE_TYPES,
    gate_categories,
    load_category_objects,
    sha256,
)

EXPORT_VERSION = "pointnext-sampling-vote-r8-export-v1"
NATIVE_EXPORT_VERSION = "eviground-native-category-export-v1"


def load_export(source):
    source = Path(source)
    if source.is_dir():

        def read(name):
            return (source / name).read_bytes()

        archive = None
    else:
        archive = ZipFile(source)
        markers = [
            n
            for n in archive.namelist()
            if n.endswith("/EXPORT_VERIFIED.json") or n == "EXPORT_VERIFIED.json"
        ]
        if len(markers) != 1:
            archive.close()
            raise ValueError("Expected one classifier export root in ZIP")
        prefix = markers[0][: -len("EXPORT_VERIFIED.json")]

        def read(name):
            return archive.read(prefix + name)

    try:
        marker = json.loads(read("EXPORT_VERIFIED.json"))
        if marker.get("status") != "passed":
            raise ValueError("Need a verified classifier export")
        payloads = {}
        for name in ("predictions.json", "config.json"):
            raw = read(name)
            if hashlib.sha256(raw).hexdigest() != marker["files"][name]:
                raise ValueError(f"Verified classifier file changed: {name}")
            payloads[name] = json.loads(raw)
    finally:
        if archive is not None:
            archive.close()
    data, config = (payloads["predictions.json"], payloads["config.json"])
    native = config.get("version") == NATIVE_EXPORT_VERSION
    score_type = NATIVE_SCORE_TYPES.get(config.get("category_source")) if native else SCORE_TYPE
    version = NATIVE_EXPORT_VERSION if native else EXPORT_VERSION
    if (
        data.get("version") != version
        or config.get("version") != version
        or data["summary"].get("score_type") != score_type
        or config.get("score_type") != score_type
        or (native and score_type is None)
        or (not data["summary"].get("complete"))
        or (data["summary"].get("errors") != 0)
    ):
        raise ValueError(
            "Need a complete verified export with its original category score type"
        )
    rows = data["results"]
    sources = {r["scene"]: r["cloud_sha256"] for r in config["sources"]}
    keys = {(r["scene"], r["bbox_id"]) for r in rows}
    if (
        not rows
        or len(keys) != len(rows)
        or len(sources) != len(config["sources"])
        or (len(rows) != data["summary"]["count"])
        or (len(sources) != data["summary"]["scenes"])
        or ({r["scene"] for r in rows} != set(sources))
    ):
        raise ValueError("classifier object/scene coverage is inconsistent")
    for row in rows:
        if (
            row.get("status") != "completed"
            or row.get("score_type") != score_type
            or type(row["bbox_id"]) is not int
            or (row["predicted_object"] != row["top5"][0]["category"])
        ):
            raise ValueError("Invalid classifier prediction row")
        gate_categories(row["top5"], score_type=score_type)
    return (rows, sources, marker["files"], config)


def prepare_tables(source, scene_root, output, scenes, top_k, proposal_policy=None):
    rows, sources, hashes, config = load_export(source)
    native = config["version"] == NATIVE_EXPORT_VERSION
    gate = {"top_k": top_k, "score_type": config["score_type"]}
    files, counts = ({}, [])
    policy = json.loads(Path(proposal_policy).read_text()) if proposal_policy is not None else None
    if native and policy is not None:
        raise ValueError("Native frontend already defines its proposals; do not apply an old proposal policy")
    if policy is not None and (policy.get("version") != "mask-nms-proposal-policy-v1"
                              or sha256(scene_root / "protocol.json") != policy["source_protocol_sha256"]):
        raise ValueError("Proposal policy belongs to a different scene frontend")
    for scene in sorted(scenes):
        if Path(scene).name != scene or scene in (".", "..") or scene not in sources:
            raise ValueError(f"Unknown classifier scene: {scene}")
        cloud_path = scene_root / scene / "scene.npz"
        if sha256(cloud_path) != sources[scene]:
            raise ValueError(f"Point cloud differs from classifier input: {scene}")
        with np.load(cloud_path, allow_pickle=False) as cloud:
            known = set(map(int, np.unique(cloud["ids"])))
        if native:
            source_record = next(row for row in config["sources"] if row["scene"] == scene)
            instances_path = scene_root / scene / "scene_instances.json"
            if sha256(instances_path) != source_record["instances_sha256"]:
                raise ValueError(f"Native instance geometry changed: {scene}")
            if "masks_sha256" in source_record:
                if sha256(scene_root / scene / "masks.npz") != source_record["masks_sha256"]:
                    raise ValueError(f"Native prediction masks changed: {scene}")
            geometry = {o["bbox_id"]: o["bbox_3d"] for o in json.loads(instances_path.read_text())}
            classified = {r["bbox_id"]: r["bbox_3d"] for r in rows if r["scene"] == scene}
            if config["score_type"] == MASK3D_SCORE_TYPE:
                expected = known - {-1}
            else:
                expected = set(source_record["candidate_ids"])
                if len(expected) != len(source_record["candidate_ids"]) or not expected <= known:
                    raise ValueError(f"Invalid original SeeGround candidate set: {scene}")
                candidate_path = scene_root / scene / 'objects.json'
                if sha256(candidate_path) != source_record['candidate_geometry_sha256']:
                    raise ValueError(f'Original SeeGround candidate geometry changed: {scene}')
                geometry = {o['bbox_id']: o['bbox_3d'] for o in json.loads(candidate_path.read_text())}
                if set(geometry) != expected:
                    raise ValueError(f'Original SeeGround candidate coverage changed: {scene}')
            if set(classified) != expected or any(geometry.get(i) != box for i, box in classified.items()):
                raise ValueError(f"Native categories must cover every aligned instance: {scene}")
        objects = []
        selection = policy["scenes"][scene] if policy is not None else None
        if selection is not None:
            if selection["cloud_sha256"] != sources[scene] or selection["instances_sha256"] != sha256(scene_root / scene / "scene_instances.json"):
                raise ValueError(f"Proposal policy geometry changed: {scene}")
            classified = {r["bbox_id"] for r in rows if r["scene"] == scene}
            if not selection["kept_ids"] or not set(selection["kept_ids"]) <= classified:
                raise ValueError(f"Proposal policy has unknown or empty representatives: {scene}")
            if "proposal_scores" in selection:
                if sha256(scene_root / scene / "masks.npz") != selection["masks_sha256"]:
                    raise ValueError(f"Proposal detection scores belong to changed masks: {scene}")
                scores = selection["proposal_scores"]
                if set(scores) != {str(i) for i in classified} or any(
                    isinstance(s, bool) or not isinstance(s, (float, int)) or not 0 <= s <= 1
                    for s in scores.values()
                ):
                    raise ValueError(f"Invalid proposal detection scores: {scene}")
        for row in sorted(
            (r for r in rows if r["scene"] == scene), key=lambda r: r["bbox_id"]
        ):
            if row["bbox_id"] not in known:
                raise ValueError(
                    f"Unknown classified instance: {scene}/{row['bbox_id']}"
                )
            if selection is not None and row["bbox_id"] not in selection["kept_ids"]:
                continue
            retained, audit = gate_categories(row["top5"], **gate)
            objects.append(
                {
                    "bbox_id": row["bbox_id"],
                    "target": row["predicted_object"],
                    "bbox_3d": row["bbox_3d"],
                    "top5": row["top5"],
                    "category_hypotheses": retained,
                    "category_gate_audit": audit,
                    "category_status": (
                        "single_hypothesis" if len(retained) == 1 else "uncertain"
                    ),
                }
            )
            if native and "proposal_detection_score" in row:
                objects[-1]["proposal_detection_score"] = row["proposal_detection_score"]
            if selection is not None and "proposal_scores" in selection:
                objects[-1]["proposal_detection_score"] = selection["proposal_scores"][str(row["bbox_id"])]
            counts.append(len(retained))
        value = {
            "version": VERSION,
            "scene": scene,
            "gate": gate,
            "source_predictions_sha256": hashes["predictions.json"],
            "source_config_sha256": hashes["config.json"],
            "cloud_sha256": sources[scene],
            "objects": objects,
        }
        if native:
            value["category_source"] = config["category_source"]
            value["frontend_protocol_sha256"] = config["frontend_protocol_sha256"]
            value["instances_sha256"] = source_record["instances_sha256"]
            if "masks_sha256" in source_record:
                value["masks_sha256"] = source_record["masks_sha256"]
            if 'candidate_geometry_sha256' in source_record:
                value['candidate_geometry_sha256'] = source_record['candidate_geometry_sha256']
        if selection is not None:
            value["proposal_policy_sha256"] = sha256(proposal_policy)
            value["suppressed_instance_ids"] = sorted(classified - set(selection["kept_ids"]))
        path = output / scene / f"objects_top{top_k}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Category inputs changed; choose a new output directory")
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        load_category_objects(path)
        files[scene] = sha256(path)
    return {
        "version": VERSION,
        "source_version": config["version"],
        **({"category_source": config["category_source"],
            "frontend_protocol_sha256": config["frontend_protocol_sha256"]} if native else {}),
        "gate": gate,
        "files": files,
        "source_predictions_sha256": hashes["predictions.json"],
        "source_config_sha256": hashes["config.json"],
        "retained_count_histogram": {str(i): counts.count(i) for i in range(6)},
        "mean_retained": float(np.mean(counts)),
        "instances": len(counts),
        **({"proposal_policy_sha256": sha256(proposal_policy)} if policy is not None else {}),
    }
