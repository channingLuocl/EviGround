"""Audit all six completed conditions before producing the comparison table."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(left, right):
    require(set(left) == set(right), "Paired comparisons require identical query IDs")
    left_only = sum(left[k]["correct"] and not right[k]["correct"] for k in left)
    right_only = sum(right[k]["correct"] and not left[k]["correct"] for k in left)
    return {"queries": len(left), "left_only_correct": left_only,
            "right_only_correct": right_only,
            "left_minus_right_pp": 100 * (left_only - right_only) / len(left)}


def audit_condition(study, name, variant, protocol, expected, code_hashes):
    rows, audit = {}, {"verified_predictions": 0, "technical_errors": 0,
                       "verified_model_responses": 0, "responses_without_model_record": 0}
    for slot in range(2):
        directory = study / "runs" / name / f"key_slot_{slot+1}"
        manifest, result = read(directory / "manifest.json"), read(directory / "results.json")
        selected = read(study / f"nr3d_shard_{slot+1}.json")["rows"]
        require(manifest["rows"] == selected, f"{name}/{slot}: selection changed")
        require(manifest["code_sha256"] == code_hashes, f"{name}/{slot}: inference code mismatch")
        require(manifest["models"] == {"EVIGROUND_MODEL": protocol["model"],
                                      "EVIGROUND_BASE_URL": protocol["base_url"]},
                f"{name}/{slot}: model or endpoint mismatch")
        require(manifest["review_limit"] == protocol["nonterminal_evidence_budget"],
                f"{name}/{slot}: evidence budget mismatch")
        policy = {"inspect": bool(variant["inspect"]), "measure": bool(variant["measure"]),
                  "category_inspection_required": bool(variant["guard"])}
        require(manifest["policy"]["evidence_tools"] == policy, f"{name}/{slot}: policy mismatch")
        require(manifest["categories"]["gate"]["top_k"] == variant["top_k"],
                f"{name}/{slot}: category gate mismatch")
        summary = result["summary"]
        require(summary["complete"] and summary["processed"] == len(selected)
                and summary["pending"] == 0, f"{name}/{slot}: incomplete condition")
        require(len(result["results"]) == len(selected), f"{name}/{slot}: missing result rows")
        for row in result["results"]:
            key = row["csv_row"]
            require(key in expected and key not in rows, f"{name}: unknown/duplicate query")
            require(all(row.get(k) == v for k, v in expected[key].items()),
                    f"{name}/{key}: result does not match frozen input")
            require(type(row.get("correct")) is bool, f"{name}/{key}: invalid correctness")
            require(row["category_top_k"] == variant["top_k"], f"{name}/{key}: wrong category gate")
            case = directory / f"q{key:05d}"
            require(read(case / "evaluation.json") == row, f"{name}/{key}: inconsistent saved result")
            if row.get("error"):
                require(not row["correct"], f"{name}/{key}: error counted as correct")
                audit["technical_errors"] += 1
            else:
                prediction = read(case / "prediction.json")
                require(prediction["gt_categories_supplied_to_model"] is False
                        and prediction["query_target_id_supplied_to_model"] is False,
                        f"{name}/{key}: prediction provenance failure")
                require(prediction["query"] == row["query"]
                        and prediction["predicted_id"] == row["predicted_id"],
                        f"{name}/{key}: prediction/evaluation mismatch")
                require(row["correct"] == (row["predicted_id"] == row["target_id"]),
                        f"{name}/{key}: invalid Nr3D metric")
                trace = prediction["active_trace"]
                if prediction["category_target_ids"]:
                    require(trace["react_evidence_policy"] == policy, f"{name}/{key}: trace policy mismatch")
                    available = [a for a in ("inspect", "measure") if policy[a]] + ["finish"]
                    require(trace["react_available_actions"] == available,
                            f"{name}/{key}: trace action set mismatch")
                    require(all(a["action"] in available for a in trace["react_actions"]),
                            f"{name}/{key}: disabled tool was requested")
                audit["verified_predictions"] += 1
            run = read(case / "run.json")
            for call in run.get("calls", []):
                model = call.get("response_model")
                if model is not None:
                    require(model == protocol["model"], f"{name}/{key}: response model mismatch")
                    audit["verified_model_responses"] += 1
                elif "response" in call:
                    audit["responses_without_model_record"] += 1
            rows[key] = row
    require(set(rows) == set(expected), f"{name}: missing frozen queries")
    metrics = {"condition": name, "queries": len(rows),
               "correct": sum(r["correct"] for r in rows.values()),
               "errors": audit["technical_errors"],
               "nulls": sum(r["predicted_id"] is None for r in rows.values()),
               "api_attempts": sum(r.get("api_attempts", 0) for r in rows.values()),
               "reported_total_tokens": sum((u or {}).get("total_tokens", 0)
                   for r in rows.values() for u in r.get("call_usage", []))}
    for label, field, value in (("overall", None, None), ("easy", "easy", True),
                                ("hard", "easy", False), ("view_dependent", "view_dep", True),
                                ("view_independent", "view_dep", False)):
        group = [r for r in rows.values() if field is None or r[field] == value]
        metrics[label + "_accuracy_pct"] = 100 * sum(r["correct"] for r in group) / len(group)
    return rows, metrics, audit


def summarize(study):
    protocol = read(study / "protocol.json")
    selection = study / "nr3d_subset_1500.json"
    require(hashlib.sha256(selection.read_bytes()).hexdigest() == protocol["subset_sha256"],
            "Frozen subset hash changed")
    expected = {r["csv_row"]: r for r in read(selection)["rows"]}
    require(len(expected) == protocol["queries"], "Wrong evaluation scale")
    identity = read(study / "source_identity.json")
    for relative, digest in identity.items():
        require(hashlib.sha256((study / "source" / relative).read_bytes()).hexdigest() == digest,
                f"Source snapshot changed: {relative}")
    code_hashes = {k.removeprefix("eviground/"): v for k, v in identity.items()}
    cases, metrics, audits = {}, [], {}
    for name, variant in protocol["conditions"].items():
        rows, values, audit = audit_condition(study, name, variant, protocol, expected, code_hashes)
        cases[name] = rows; metrics.append(values); audits[name] = audit
        print(f"Audited {name}: {len(rows)} queries, {audit}", flush=True)
    comparisons = {f"E0-{name}": paired(cases["E0"], cases[name]) for name in cases if name != "E0"}
    comparisons.update({name: paired(cases[name[:2]], cases[name[3:]])
                        for name in protocol["auxiliary_comparisons"]})
    for row in metrics:
        row["e0_minus_condition_pp"] = paired(cases["E0"], cases[row["condition"]])["left_minus_right_pp"]
    result = {"model": protocol["model"], "queries_per_condition": len(expected),
              "metrics": metrics, "paired_comparisons": comparisons, "audit": audits,
              "usage_note": "API-reported tokens, not a calculation of monetary charges"}
    (study / "verified_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (study / "ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    lines = [f"# Nr3D ablation — {protocol['model']}", "",
             f"Each condition uses the same {len(expected)} descriptions. Errors and nulls remain in the denominator.", "",
             "| Condition | Overall | Easy | Hard | View-dep. | View-indep. | E0 minus condition (pp) | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        values = [f"{row[k + '_accuracy_pct']:.2f}" for k in
                  ("overall", "easy", "hard", "view_dependent", "view_independent")]
        lines.append(f"| {row['condition']} | " + " | ".join(values) +
                     f" | {row['e0_minus_condition_pp']:+.2f} | {row['errors']} |")
    lines.extend(["", "Positive E0-minus-condition differences favor E0; negative differences favor the ablated condition.",
                  "", "Auxiliary paired comparisons:"])
    for name in protocol["auxiliary_comparisons"]:
        lines.append(f"- {name}: {comparisons[name]['left_minus_right_pp']:+.2f} pp")
    (study / "ablation_results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/ablation_20261104")
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.wait_for_pid:
        while read(args.study / "status.json").get("phase") != "complete":
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                raise ValueError("Experiment runner exited before all six conditions completed")
            state = subprocess.run(["ps", "-p", str(args.wait_for_pid), "-o", "state="],
                                   capture_output=True, text=True).stdout.strip()
            require(state and not state.startswith("Z"), "Experiment runner is no longer active")
            time.sleep(30)
    summarize(args.study)


if __name__ == "__main__":
    main()
