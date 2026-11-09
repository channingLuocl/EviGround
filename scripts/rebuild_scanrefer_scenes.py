"""Rebuild transferred Mask3D scenes locally and require server-identical bytes."""

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_scanrefer_proposals import expand_proposals, read_mesh
from prepare_scanrefer_raw import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.data_root / "raw_inputs.json").read_text())
    assert raw["complete"], "Raw inputs are incomplete"
    records = []
    for row in raw["scenes"]:
        scene = row["scene"]
        folder = args.scene_root / scene
        marker = folder / "PREPARED.json"
        if not marker.exists():
            continue
        expected = json.loads(marker.read_text())
        assert expected["status"] == "passed", scene
        for name, checksum in expected["files"].items():
            if name != "scene.npz":
                assert digest(folder / name) == checksum, (scene, name)
        cloud = folder / "scene.npz"
        checksum = expected["files"]["scene.npz"]
        if not cloud.exists() or digest(cloud) != checksum:
            directory = args.data_root / "raw_scans" / scene
            for item in row["raw_files"]:
                if not item["file"].endswith(".aggregation.json"):
                    assert digest(directory / item["file"]) == item["sha256"], scene
            xyz, rgb, _, matrix = read_mesh(directory, scene)
            with np.load(folder / "masks.npz", allow_pickle=False) as masks:
                assert int(masks["vertices"]) == len(xyz), scene
                values = np.unpackbits(masks["packed_masks"], axis=0,
                                       count=len(xyz)).astype(bool)
            aligned = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3]
            points, colors, ids, objects, _ = expand_proposals(aligned, rgb, values)
            assert objects == json.loads((folder / "scene_instances.json").read_text()), scene
            temporary = folder / "scene.rebuilding.npz"
            np.savez_compressed(temporary, xyz=points, rgb=colors, ids=ids)
            actual = digest(temporary)
            if actual != checksum:
                raise ValueError(f"{scene}: local bytes differ; retrieve the original "
                                 f"server scene.npz instead ({actual} != {checksum})")
            temporary.replace(cloud)
            del xyz, rgb, values, aligned, points, colors, ids
        records.append({"scene": scene, "scene_sha256": checksum,
                        "server_byte_identical": True})
        if len(records) % 10 == 0:
            print(json.dumps({"verified_scenes": len(records)}), flush=True)
    result = {"status": "passed", "complete": len(records) == len(raw["scenes"]),
              "scenes": len(records), "expected_scenes": len(raw["scenes"]),
              "method": "Local reconstruction from original predicted masks; "
                        "every scene.npz must match the original server SHA256",
              "records": records}
    save(args.data_root / "LOCAL_SCENE_RECONSTRUCTION.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
