"""Audit complete synthetic module studies, including failures in the denominator."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from eviground.evaluation.reporting import evaluate, trajectory_metrics
from eviground.grounding.settings import settings
from tests.unit.test_ablation_runner import load_runner


path = Path(__file__).resolve().parents[2] / "scripts/summarize_nr3d_ablation.py"
spec = importlib.util.spec_from_file_location("ablation_summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class AblationSummaryTests(unittest.TestCase):
    def create_study(self, study):
        runner = load_runner({})
        selection = [{"csv_row": i, "scene": "synthetic", "query": "the chair", "target_id": 2,
                      "easy": bool(i % 2), "view_dep": bool(i // 2)} for i in range(4)]
        save(study / "nr3d_subset_1500.json", {"rows": selection})
        source = study / "source/eviground/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_text("# offline fixture\n")
        hashes = {"eviground/fixture.py": hashlib.sha256(source.read_bytes()).hexdigest()}
        save(study / "source_identity.json", hashes)
        protocol = {"suite": "functional_modules_v1", "baseline": "A0", "conditions": runner.VARIANTS,
                    "model": "offline-model", "base_url": "https://offline.example/v1", "queries": 4,
                    "subset_sha256": hashlib.sha256((study / "nr3d_subset_1500.json").read_bytes()).hexdigest(),
                    "successful_tool_budget": 12, "repair_allowance": 6, "max_decision_calls": 19,
                    "decision_max_tokens": 4000, "annealing_schedule": runner.SCHEDULE, "auxiliary_comparisons": []}
        save(study / "protocol.json", protocol)
        for slot in (0, 1):
            selected = selection[slot::2]
            save(study / f"nr3d_shard_{slot + 1}.json", {"rows": selected})
            for name, variant in runner.VARIANTS.items():
                directory = study / "runs" / name / f"key_slot_{slot + 1}"
                with patch.dict(os.environ, runner.environment("unit-test-key", variant, 1)):
                    policy = settings()
                manifest = {"rows": selected, "code_sha256": {"fixture.py": hashes["eviground/fixture.py"]},
                            "models": {"EVIGROUND_MODEL": protocol["model"], "EVIGROUND_BASE_URL": protocol["base_url"]},
                            "review_limit": 12, "policy": policy, "categories": {"gate": {"top_k": variant["top_k"]}}}
                save(directory / "manifest.json", manifest)
                results = []
                for item in selected:
                    case = directory / f"q{item['csv_row']:05d}"
                    available = [a for a in ("inspect", "measure") if variant[a]] + ["finish"]
                    trace = {"react_evidence_policy": policy["evidence_tools"], "react_available_actions": available,
                             "react_actions": [{"action": "finish"}], "react_observations": [],
                             "interface": "multimodal_react_dag_monitor" if variant["monitor"] else "multimodal_react",
                             "policy_annealing_enabled": bool(variant["annealing"]),
                             "policy_annealing_history": [{}] if variant["annealing"] else [],
                             "react_budget": {"accounting": "successful_tools", "evidence_limit": 12,
                                              "repair_limit": 6, "max_decision_calls": 19},
                             "termination_cause": "supported_finish"}
                    prediction = {"query": item["query"], "predicted_id": 2, "category_top_k": variant["top_k"],
                                  "gt_categories_supplied_to_model": False, "query_target_id_supplied_to_model": False,
                                  "category_target_ids": [2], "active_trace": trace,
                                  "call_usage": [{"total_tokens": 100}]}
                    save(case / "prediction.json", prediction)
                    row = evaluate(item, case / "prediction.json", case / "console.log", variant["top_k"])
                    row["category_candidate_recall"] = True
                    if name == "A1" and item["csv_row"] == 0:
                        # The worker failed after acquiring one observation, which still costs a tool call.
                        trace["react_actions"] = [{"action": "measure"}]
                        trace["react_observations"] = [{"tool": "measure", "focus_ids": [1, 2]}]
                        trace.pop("termination_cause")
                        row.update(predicted_id=None, correct=False, error="offline failure", **trajectory_metrics(trace))
                        row.pop("category_candidate_recall")
                    save(case / "run.json", {"active_trace": trace,
                                              "calls": [{"response_model": protocol["model"], "usage": {"total_tokens": 100}}]})
                    save(case / "evaluation.json", row)
                    results.append(row)
                save(directory / "results.json", {"summary": {"complete": True, "processed": len(selected), "pending": 0},
                                                   "results": results})

    def test_unified_five_row_report_and_failure_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            self.create_study(study)
            summary.summarize(study)
            result = summary.read(study / "verified_summary.json")
            self.assertEqual(list(result["paired_comparisons"]), ["A0-A1", "A0-A2", "A0-A3", "A0-A4"])
            variant = result["metrics"][1]
            self.assertEqual(variant["overall_accuracy_pct"], 75)
            self.assertEqual(variant["a0_minus_condition_pp"], 25)
            self.assertEqual(variant["mean_successful_tools"], .25)
            self.assertEqual(variant["errors"], 1)
            self.assertEqual(variant["cases_without_candidate_recall"], 1)
            table = (study / "ablation_results.md").read_text()
            self.assertEqual(table.count("| Condition |"), 1)
            for i in range(5):
                self.assertEqual(table.count(f"| A{i} |"), 1)

    def test_audit_rejects_changed_module_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            self.create_study(study)
            path = study / "runs/A2/key_slot_1/manifest.json"
            changed = summary.read(path)
            changed["policy"]["policy_annealing_enabled"] = False
            save(path, changed)
            with self.assertRaisesRegex(ValueError, "annealing mismatch"):
                summary.summarize(study)

    def test_paired_comparison_requires_identical_query_ids(self):
        left = {1: {"correct": True}, 2: {"correct": False}}
        right = {1: {"correct": False}, 2: {"correct": False}}
        self.assertEqual(summary.paired(left, right)["left_minus_right_pp"], 50)
        with self.assertRaisesRegex(ValueError, "identical query IDs"):
            summary.paired(left, {1: {"correct": True}})

    def test_archived_e_protocol_still_uses_legacy_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            self.create_study(study)
            protocol = summary.read(study / "protocol.json")
            protocol.pop("suite")
            protocol.pop("baseline")
            protocol["nonterminal_evidence_budget"] = protocol.pop("successful_tool_budget")
            protocol["conditions"] = {name.replace("A", "E"): value for name, value in protocol["conditions"].items()}
            for i in range(5):
                (study / "runs" / f"A{i}").rename(study / "runs" / f"E{i}")
            protocol["conditions"]["E5"] = protocol["conditions"]["E3"]
            shutil.copytree(study / "runs/E3", study / "runs/E5")
            protocol["auxiliary_comparisons"] = ["E4-E1", "E4-E3"]
            save(study / "protocol.json", protocol)
            summary.summarize(study)
            result = summary.read(study / "verified_summary.json")
            self.assertEqual(len(result["metrics"]), 6)
            self.assertIn("E0-E5", result["paired_comparisons"])
            self.assertIn("E4-E3", result["paired_comparisons"])
            self.assertEqual(result["metrics"][1]["e0_minus_condition_pp"], 25)
