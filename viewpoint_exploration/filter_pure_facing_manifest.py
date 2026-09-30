#!/usr/bin/env python3
"""Filter an existing facing-anchor manifest without changing its sample draw."""

import argparse
import json
from pathlib import Path

from prepare_facing_anchor_manifest import is_pure_direction


def main():
    parser = argparse.ArgumentParser(description="从既有 facing-anchor 清单筛出纯方向查询")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("输出文件已存在；请使用新路径")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if is_pure_direction(row["utterance"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    print(json.dumps({"input_queries": len(rows), "pure_direction_queries": len(kept), "excluded_compositional_queries": len(rows) - len(kept)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
