#!/usr/bin/env python3
"""Download only mesh assets required by a JSONL manifest.

Files are written atomically through a .part sibling, so interrupted downloads
are retried on the next invocation while completed files are skipped.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


BASE_URL = "http" + "://kaldir.vc.in.tum.de/scannet/v2/scans/"


def scene_ids(manifest):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted({row["scene_id"] for row in rows})


def download(url, destination, timeout):
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="按 JSONL manifest 直接下载 ScanNet mesh 与元数据")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="ScanNet 根目录；文件写到 scans/<scene_id>/")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.timeout < 1:
        raise SystemExit("timeout 必须大于 0")
    suffixes = ("_vh_clean_2.ply", ".txt")
    failures = []
    for scene_id in scene_ids(args.manifest):
        scene_dir = args.output_root / "scans" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for suffix in suffixes:
            destination = scene_dir / f"{scene_id}{suffix}"
            if destination.is_file() and destination.stat().st_size > 0:
                print(f"skip {destination}", flush=True)
                continue
            url = BASE_URL + scene_id + "/" + destination.name
            print(f"download {url}", flush=True)
            try:
                download(url, destination, args.timeout)
                print(f"done {destination} {destination.stat().st_size} bytes", flush=True)
            except Exception as error:
                failures.append({"scene_id": scene_id, "suffix": suffix, "error": str(error)})
                print(f"failed {destination}: {error}", flush=True)
    print(json.dumps({"scenes": len(scene_ids(args.manifest)), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
