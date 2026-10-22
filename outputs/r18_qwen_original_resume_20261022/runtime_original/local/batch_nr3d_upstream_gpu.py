"""Small, resumable Nr3D trial with the upstream PyTorch3D renderer."""

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_nr3d import select_rows, summarize
from inference.projection import render_point_cloud_with_pytorch3d_with_objects
from real_scene import parse_query, select_objects
from run import ground, load_objects


SOURCE = "https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view"


def render_and_ground(scene_dir, query, output):
    objects_path = scene_dir / "objects.json"
    objects = load_objects(objects_path)
    parsed = parse_query(query)
    targets = select_objects(parsed["Target"], objects) or objects
    anchors = select_objects(parsed["Anchor"], objects) or targets
    cloud = np.load(scene_dir / "scene.npz", allow_pickle=False)
    xyz = cloud["xyz"].astype(np.float32)
    rgb = cloud["rgb"].astype(np.float32)
    if rgb.max() > 1:
        rgb /= 255
    output.mkdir(parents=True, exist_ok=True)
    image = render_point_cloud_with_pytorch3d_with_objects(
        objects=objects,
        targets=targets,
        anchors=anchors,
        center=xyz.mean(axis=0),
        scan_pc=np.concatenate((xyz, rgb), axis=1),
        save_dir=str(output),
        image_size=680,
        draw_id=True,
        draw_img=True,
        device="cuda",
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "query": query,
                "parsed_query": parsed,
                "target_ids": [obj["bbox_id"] for obj in targets],
                "anchor_ids": [obj["bbox_id"] for obj in anchors],
                "renderer": "upstream PyTorch3D",
                "ground_truth_supplied_to_model": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    ground(
        SimpleNamespace(
            image=image,
            objects=str(objects_path),
            query=query,
            dry_run=False,
            output=str(output / "prediction.json"),
        )
    )


def write_report(output, results, manifest):
    stats = summarize(results)
    (output / "results.json").write_text(
        json.dumps({"summary": stats, "manifest": manifest, "results": results}, ensure_ascii=False, indent=2)
    )
    cards = []
    for index, result in enumerate(results):
        esc = lambda value: html.escape(str(value))
        cards.append(
            f'<article><h2>{index + 1}. {"正确" if result.get("correct") else "错误或未完成"}</h2>'
            f'<p>{esc(result["query"])}</p><p>正确编号：{result["target_id"]}；'
            f'预测编号：{esc(result.get("predicted_id", "未完成"))}</p>'
            f'<p>{esc(result.get("response", result.get("error", "")))}</p>'
            f'<img loading="lazy" src="q{index + 1:03d}/rendered.png"></article>'
        )
    (output / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>SeeGround · Nr3D GPU 小样本试跑</title>"
        "<style>body{max-width:960px;margin:40px auto;font:18px/1.7 system-ui;padding:20px;background:#f3f5f8}"
        "article{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%}</style>"
        f"<h1>SeeGround · Nr3D GPU 小样本试跑</h1><p>完成 {stats['completed']}/{stats['total']}，"
        f"正确 {stats['correct']}/{stats['total']}。</p><p>单场景、固定 CSV 选题；使用 Qwen API 与原仓库 "
        "PyTorch3D 渲染。这不是论文完整基准成绩。</p>" + "".join(cards)
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "local_outputs/downloads/nr3d_official.csv")
    parser.add_argument("--scene", default="scene0011_00")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "local_outputs/nr3d_upstream_gpu_trial_10")
    args = parser.parse_args()
    rows, available = select_rows(args.csv, args.scene, args.count)
    scene_dir = ROOT / "local_outputs" / args.scene
    objects = load_objects(scene_dir / "objects.json")
    candidate_ids = {obj["bbox_id"] for obj in objects}
    if any(row["target_id"] not in candidate_ids for row in rows):
        raise ValueError("选定描述包含候选清单缺失的目标；请检查数据对齐")
    manifest = {
        "source": SOURCE,
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "scene": args.scene,
        "selection": "first N rows in CSV with matching scan_id and correct_guess=True",
        "available": available,
        "rows": rows,
        "objects_sha256": hashlib.sha256((scene_dir / "objects.json").read_bytes()).hexdigest(),
        "renderer": "upstream PyTorch3D",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("输出目录已有不同实验，请更换目录")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    for index, row in enumerate(rows):
        if args.stop_after is not None and index >= args.stop_after:
            break
        output = args.output / f"q{index + 1:03d}"
        prediction_path = output / "prediction.json"
        result = dict(row)
        try:
            if not prediction_path.exists():
                render_and_ground(scene_dir, row["query"], output)
            prediction = json.loads(prediction_path.read_text())
            if prediction["query"] != row["query"]:
                raise ValueError("缓存查询不一致")
            result.update(predicted_id=prediction["predicted_id"], response=prediction["response"], model=prediction["model"])
            result["correct"] = result["predicted_id"] == row["target_id"]
        except (ValueError, OSError, KeyError, IndexError) as exc:
            result.update(error=str(exc), correct=False)
        results.append(result)
        stats = write_report(args.output, results, manifest)
        print(f'{index + 1}/{len(rows)}: predicted={result.get("predicted_id")} expected={row["target_id"]} correct={result["correct"]}', flush=True)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
