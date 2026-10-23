"""Synthetic, evaluation-independent fixtures for the current workflow."""

import json
from pathlib import Path
import numpy as np
from eviground.categories.export import EXPORT_VERSION
from eviground.categories.hypotheses import SCORE_TYPE, sha256


def response(payload):
    return {
        "_content": json.dumps(payload),
        "_finish_reason": "stop",
        "_request": {"latency_s": 0, "api_attempts": [{"status": "success"}]},
    }


def parsed(target="chair", anchors=()):
    return {
        "Target": [target],
        "Anchor": list(anchors),
        "Relations": [],
        "Viewpoint": "",
        "Observer": {
            "origin": {"kind": "unspecified", "category": "", "phrase": ""},
            "facing": {"category": "", "group": False, "phrase": ""},
        },
        "Attributes": [],
        "Unhandled": [],
    }


def make_export(root):
    root = Path(root)
    source, scenes = root / "export", root / "scenes"
    source.mkdir()
    scene = scenes / "synthetic"
    scene.mkdir(parents=True)
    names = [
        ("stool", "chair", "box", "bag", "sofa"),
        ("chair", "clothes", "box", "bag", "sofa"),
        ("table", "desk", "box", "bag", "sofa"),
    ]
    rng = np.random.default_rng(123)
    points, labels, rows = [], [], []
    for oid, categories in enumerate(names, 1):
        box = [float(oid * 2), 0.0, 0.5, 1.0, 1.0, 1.0]
        points.append(rng.uniform(-0.5, 0.5, (150, 3)) + box[:3])
        labels.extend([oid] * 150)
        rows.append(
            {
                "scene": "synthetic",
                "bbox_id": oid,
                "bbox_3d": box,
                "status": "completed",
                "predicted_object": categories[0],
                "score_type": SCORE_TYPE,
                "top5": [
                    {"category": name, "score": score, "rank": rank}
                    for rank, (name, score) in enumerate(
                        zip(categories, (3.4, 2.6, 1.5, 0.1, -0.2)), 1
                    )
                ],
                "evaluation_only": {"gt_category": "SECRET_CLASSIFIER_EVAL_LABEL"},
            }
        )
    np.savez(
        scene / "scene.npz",
        xyz=np.concatenate(points),
        ids=np.asarray(labels),
        rgb=np.tile([80, 140, 200], (len(labels), 1)).astype(np.uint8),
    )
    objects = [
        {
            "bbox_id": r["bbox_id"],
            "target": "SECRET_SCENE_GT_LABEL",
            "bbox_3d": r["bbox_3d"],
        }
        for r in rows
    ]
    objects.append(
        {
            "bbox_id": 99,
            "target": "SECRET_SUPPLEMENT_GT_LABEL",
            "bbox_3d": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        }
    )
    (scene / "scene_instances.json").write_text(json.dumps(objects))
    (scene / "objects_gt.json").write_text(json.dumps(objects))
    (source / "predictions.json").write_text(
        json.dumps(
            {
                "version": EXPORT_VERSION,
                "summary": {
                    "complete": True,
                    "errors": 0,
                    "count": 3,
                    "scenes": 1,
                    "score_type": SCORE_TYPE,
                },
                "results": rows,
                "evaluation_only": {"answer": "SECRET_EVAL_ANSWER"},
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "version": EXPORT_VERSION,
                "score_type": SCORE_TYPE,
                "sources": [
                    {"scene": "synthetic", "cloud_sha256": sha256(scene / "scene.npz")}
                ],
            }
        )
    )
    (source / "EXPORT_VERIFIED.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "files": {
                    n: sha256(source / n) for n in ("predictions.json", "config.json")
                },
            }
        )
    )
    return source, scenes, rows
