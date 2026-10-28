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


def accounting(output):
    """Each physical trajectory occurs once: current query or an archived replacement."""
    directories = [p for pattern in ('q*', 'retry_archive/round*/q*')
                   for p in output.glob(pattern) if p.is_dir()]
    totals = Counter()
    unknown_usage = 0
    trajectories = 0
    for directory in directories:
        path = directory / 'run.json'
        trajectories += path.exists()
        ledger = directory / 'api_attempts.jsonl'
        if ledger.exists():
            events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            starts = [row['attempt_id'] for row in events if row['event'] == 'start']
            ends = {row['attempt_id']: row for row in events if row['event'] == 'end'}
            if len(starts) != len(set(starts)) or set(ends) - set(starts):
                raise ValueError(f'Invalid API attempt ledger: {ledger}')
            calls = [{'api_attempts': [ends.get(key, {'usage': None}) for key in starts]}]
        elif path.exists():
            calls = json.loads(path.read_text()).get('calls', [])
        else:
            calls = []
        for call in calls:
            attempts = call.get('api_attempts')
            diagnostic = call.get('api_diagnostic') or {}
            if not isinstance(attempts, list):
                attempts = diagnostic.get('api_attempts')
            if not isinstance(attempts, list):
                attempts = [] if call.get('cache_hit') else [{'usage': call.get('usage')}]
            totals['api_attempts'] += len(attempts)
            for attempt in attempts:
                usage = attempt.get('usage')
                if not usage:
                    unknown_usage += 1
                    continue
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    totals[key] += usage.get(key, 0) or 0
                totals['cached_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
                totals['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
    session_files = [p for p in (output / 'execution_sessions').glob('*.json')
                     if not p.name.startswith('._')]
    elapsed = sum((json.loads(p.read_text()).get('elapsed_s') or 0) for p in session_files)
    return {**dict(totals), 'trajectories': trajectories, 'attempts_without_token_usage': unknown_usage,
            'wall_elapsed_s_completed_sessions': elapsed,
            'incomplete_sessions': sum('finished_at' not in json.loads(p.read_text()) for p in session_files),
            'cost': None, 'cost_note': 'Provider billing/prices unavailable; no currency cost inferred from tokens'}


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

    by_id = {r['csv_row']: r for r in results}

    def metrics(selection):
        total = len(selection)
        return {
            "total": total,
            "processed": sum(r["csv_row"] in observed for r in selection),
            "correct25": sum(observed.get(r["csv_row"], 0) >= 0.25 for r in selection),
            "correct50": sum(observed.get(r["csv_row"], 0) >= 0.5 for r in selection),
            "failed25": sum(observed.get(r["csv_row"], 0) < 0.25 for r in selection),
            "incorrect_answers25": sum(r['csv_row'] in by_id and not by_id[r['csv_row']].get('error')
                                       and by_id[r['csv_row']].get('predicted_id') is not None
                                       and observed[r['csv_row']] < .25 for r in selection),
            "errors": sum(bool(by_id.get(r['csv_row'], {}).get('error')) for r in selection),
            "abstentions": sum(r['csv_row'] in by_id and not by_id[r['csv_row']].get('error')
                                and by_id[r['csv_row']].get('predicted_id') is None for r in selection),
            "acc25": sum(observed.get(r["csv_row"], 0) >= 0.25 for r in selection) / total if total else None,
            "acc50": sum(observed.get(r["csv_row"], 0) >= 0.5 for r in selection) / total if total else None,
        }

    value = {
        "dataset": "scanrefer",
        "evaluation_policy": "complete_single_pass" if len(observed) == len(rows) else "single_pass_in_progress",
        "accounting": accounting(args.results.parent),
        "proposal_source_declared": args.proposals,
        "complete": len(observed) == len(rows),
        "pending": len(rows) - len(observed),
        "errors": sum(bool(r.get("error")) for r in results),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "unique": metrics([r for r in rows if r["unique"]]),
        "multiple": metrics([r for r in rows if not r["unique"]]),
        "overall": metrics(rows),
    }
    history_path = args.results.parent / 'retry_history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())['rounds']
        value.update(evaluation_policy='cumulative_retry_false',
                     retry_rounds=len(history),
                     original_single_pass_acc25=history[0]['correct_before']/len(rows),
                     newly_correct_by_round=[r['newly_correct'] for r in history])
    if (args.results.parent / 'retry_pending.json').exists():
        value['evaluation_policy'] = 'cumulative_retry_false_in_progress'
    first = args.results.parent / 'first_pass/scanrefer_metrics.json'
    if first.exists() and history_path.exists():
        first_value = json.loads(first.read_text())
        value['first_pass'] = {k: first_value[k] for k in ('overall', 'unique', 'multiple', 'accounting')}
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
