"""Audit the explicitly requested GPT E0-E2 / Qwen E3-E5 comparison."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from summarize_nr3d_ablation import audit_condition, paired, read, require


def source_complete(source):
    study = Path(source["study"])
    for name in source["conditions"]:
        for slot in (1, 2):
            path = study / "runs" / name / f"key_slot_{slot}/results.json"
            if not path.exists() or not read(path)["summary"]["complete"]:
                return False
    return True


def summarize(plan, output):
    cases, metrics, audits = {}, [], {}
    expected = identity = subset_hash = reference_protocol = None
    common_api = None
    for source in plan["sources"]:
        study = Path(source["study"])
        protocol = read(study / "protocol.json")
        require(protocol["model"] == source["model"], "Source model differs from plan")
        selection = study / "nr3d_subset_1500.json"
        digest = hashlib.sha256(selection.read_bytes()).hexdigest()
        require(digest == protocol["subset_sha256"] == plan["subset_sha256"], "Subset changed")
        selected = {row["csv_row"]: row for row in read(selection)["rows"]}
        hashes = read(study / "source_identity.json")
        for relative, value in hashes.items():
            require(hashlib.sha256((study / "source" / relative).read_bytes()).hexdigest() == value,
                    f"Frozen inference source changed: {relative}")
        if expected is None:
            expected, identity, subset_hash, reference_protocol = selected, hashes, digest, protocol
        require(selected == expected and hashes == identity and digest == subset_hash,
                "Studies differ in frozen data or inference code")
        require(len(selected) == protocol["queries"] == 1500, "Wrong evaluation scale")
        for field in ("conditions", "nonterminal_evidence_budget", "initial_evidence",
                      "thinking_enabled", "completion_token_floor", "temperature"):
            require(protocol[field] == reference_protocol[field], f"Changed shared setting: {field}")
        for name in source["conditions"]:
            require(name not in cases, "Duplicate condition source")
            condition_workers = set()
            for slot in (1, 2):
                manifest = read(study / "runs" / name / f"key_slot_{slot}/manifest.json")
                api = manifest["api"]
                require(manifest["workers"] == api["api_concurrency"],
                        f"{name}: inconsistent recorded concurrency")
                condition_workers.add(manifest["workers"])
                shared = {key: value for key, value in api.items()
                          if key not in ("api_slot_dir", "api_concurrency", "allow_http")}
                if common_api is None:
                    common_api = shared
                require(shared == common_api, f"{name}: unexpected API parameter difference")
            require(len(condition_workers) == 1, f"{name}: shard concurrency differs")
            rows, values, audit = audit_condition(
                study, name, protocol["conditions"][name], protocol, expected,
                {key.removeprefix("eviground/"): value for key, value in hashes.items()})
            require(audit["responses_without_model_record"] == 0, f"{name}: missing model records")
            values.update({"model": source["model"], "study": str(study),
                           "workers_per_key": condition_workers.pop()})
            cases[name] = rows
            metrics.append(values)
            audits[name] = audit
            print(f"Audited {name} / {source['model']}: {len(rows)} queries", flush=True)
    require(set(cases) == {f"E{i}" for i in range(6)}, "All six conditions are required")
    models = {row["condition"]: row["model"] for row in metrics}
    comparisons = {}
    for label in [f"E0-E{i}" for i in range(1, 6)] + ["E4-E1", "E4-E3"]:
        left, right = label.split("-")
        comparisons[label] = {**paired(cases[left], cases[right]),
                              "left_model": models[left], "right_model": models[right],
                              "same_model": models[left] == models[right]}
    for row in metrics:
        row["e0_minus_condition_pp"] = paired(cases["E0"], cases[row["condition"]])["left_minus_right_pp"]
        row["same_model_as_e0"] = row["model"] == models["E0"]
    note = ("Cross-model differences include both model and module changes and cannot isolate module effects. "
            "All original errors and null predictions remain in the 1,500-query denominator.")
    output.mkdir(parents=True, exist_ok=True)
    result = {"model_by_condition": models, "queries_per_condition": 1500,
              "total_condition_query_outcomes": sum(len(rows) for rows in cases.values()),
              "subset_sha256": subset_hash, "metrics": metrics, "audit": audits,
              "paired_comparisons": comparisons, "interpretation": note,
              "usage_note": "API-reported tokens; no monetary charge calculation"}
    (output / "verified_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    lines = ["# Nr3D comparison with recorded model per condition", "", note, "",
             "| Condition | Model | Overall | Easy | Hard | View-dep. | View-indep. | E0 minus condition (pp) | Errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        values = [f"{row[key + '_accuracy_pct']:.2f}" for key in
                  ("overall", "easy", "hard", "view_dependent", "view_independent")]
        lines.append(f"| {row['condition']} | {row['model']} | " + " | ".join(values)
                     + f" | {row['e0_minus_condition_pp']:+.2f} | {row['errors']} |")
    lines.extend(["", "Auxiliary paired comparisons:"])
    for label in ("E4-E1", "E4-E3"):
        comparison = comparisons[label]
        lines.append(f"- {label}: {comparison['left_minus_right_pp']:+.2f} pp; same model: {comparison['same_model']}")
    (output / "ablation_results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    if args.wait:
        while not all(source_complete(source) for source in plan["sources"]):
            for source in plan["sources"]:
                if source_complete(source):
                    continue
                pid = source["runner_pid"]
                state = subprocess.run(["ps", "-p", str(pid), "-o", "state="],
                                       capture_output=True, text=True).stdout.strip()
                require(state and not state.startswith("Z"), f"Runner {pid} stopped before completion")
                os.kill(pid, 0)
                require(read(Path(source["study"]) / "status.json").get("phase") != "technical_failure",
                        f"Runner {pid} reported a terminal failure")
            time.sleep(30)
    summarize(plan, args.plan.parent)


if __name__ == "__main__":
    main()
