"""Module boundaries, shared scheduling and resource accounting, without API calls."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding import react
from eviground.grounding.annealing import policy_state
from eviground.grounding.settings import settings
from eviground.evaluation.reporting import trajectory_metrics
from tests.helpers import parsed, response
from tests.unit.test_ablation_runner import load_runner


QUERY = "the chair"


def finish():
    return {"action": "finish", "reason": "initial evidence supports chair 2",
            "focus_ids": [], "context": None, "azimuth_deg": None,
            "elevation_deg": None, "frame": None, "predicted_id": 2,
            "evidence": [{"source_quote": QUERY, "object_ids": [2], "observation": "chair 2"}],
            "unresolved": []}


def verified_plan():
    common = {"source_quotes": [QUERY], "condition_ids": ["q0"], "status": "completed",
              "bound_object_ids": [2], "evidence_ids": ["initial"], "conclusion": "supported by initial evidence"}
    return {"nodes": [{**common, "id": "bind", "kind": "binding", "goal": "compare chairs",
                       "depends_on": [], "allowed_actions": [], "excluded_target_ids": [1]},
                      {**common, "id": "verify", "kind": "verification", "goal": "verify target",
                       "depends_on": ["bind"], "allowed_actions": ["finish"], "excluded_target_ids": []}],
            "current_node_id": "verify", "reopen_node_ids": [], "revision_reason": "initial evidence sufficient"}


class PolicyAnnealingTests(unittest.TestCase):
    def run_ground(self, variant_id, responses=None, budget=12, repairs=6):
        runner = load_runner({})
        env = runner.environment("unit-test-key", runner.VARIANTS[variant_id], 1)
        env["EVIGROUND_DAG_REPAIR_LIMIT"] = str(repairs)
        objects = [{"bbox_id": i, "target": "chair", "bbox_3d": [i, 0, .5, 1, 1, 1],
                    "category_hypotheses": [{"category": "chair", "rank": 1, "gap_from_top1": 0}]}
                   for i in (1, 2)]
        result_action = finish()
        if runner.VARIANTS[variant_id]["monitor"]:
            result_action["monitor"] = verified_plan()
        responses = [result_action] if responses is None else responses
        trace = {"steps": [], "internal_usage": {"render": 0, "compute": 0, "visual_check": 0}}
        views = [{"image": "initial.png", "kind": "overview"}, {"image": "top.png", "kind": "top"}]
        with (tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env),
              patch("eviground.rendering.overview.scene_views", return_value=views),
              patch("eviground.rendering.inspection.render_action") as render,
              patch.object(react, "image_part", side_effect=lambda p: {"image": p}),
              patch.object(react, "chat", side_effect=[response(a) for a in responses]) as chat):
            result = react.ground(QUERY, parsed(), objects, objects,
                                  np.zeros((2, 3)), np.zeros((2, 3)), np.arange(2),
                                  Path(out), [], trace, budget)
            return result, chat.call_args_list, render.call_count

    def test_progress_boundaries_and_early_tool_deadline(self):
        states = [policy_state(t, 4, 3 - t) for t in range(4)]
        self.assertEqual([s["stage"] for s in states], ["explore", "focus", "verify", "deadline"])
        self.assertEqual([s["progress"] for s in states], [0, 1 / 3, 2 / 3, 1])
        self.assertEqual(policy_state(0, 19, 0)["stage"], "deadline")
        self.assertEqual(policy_state(0, 1, 12, last_decision=True)["stage"], "deadline")
        self.assertIn("counterevidence", states[2]["correction_policy"])

    def test_all_five_variants_can_finish_early_with_matched_limits(self):
        initial = []
        for name in ("A0", "A1", "A2", "A3", "A4"):
            with self.subTest(condition=name):
                result, calls, renders = self.run_ground(name)
                self.assertEqual(result["predicted_id"], 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(renders, 0)
                trace = result["active_trace"]
                self.assertEqual(trace["react_budget"]["max_decision_calls"], 19)
                self.assertEqual(trace["react_budget"]["evidence_limit"], 12)
                self.assertEqual(trace["react_budget"]["repair_limit"], 6)
                self.assertEqual(trace["react_budget"]["successful_tool_calls"], 0)
                self.assertEqual(calls[0].kwargs["max_tokens"], 4000)
                self.assertEqual("dag_monitor_history" in trace, name != "A2")
                self.assertEqual(bool(trace["policy_annealing_history"]), name != "A1")
                state = json.loads(calls[0].args[0][1]["content"][0]["text"])
                self.assertEqual("policy_annealing" in state, name != "A1")
                initial.append(trace["react_initial_input"])
                if name == "A3":
                    schema = calls[0].kwargs["response_schema"]["schema"]
                    self.assertEqual(schema["properties"]["action"]["enum"], ["finish"])
                    self.assertEqual(state["policy_annealing"]["stage"], "deadline")
                    self.assertFalse(trajectory_metrics(trace)["budget_exhausted"])
                with tempfile.TemporaryDirectory() as directory:
                    runner = load_runner({})
                    runner.STUDY = Path(directory)
                    prediction_path = runner.STUDY / "prediction.json"
                    prediction_path.write_text(json.dumps(result))
                    runner.check_pilots(name, [{"summary": {"complete": True, "errors": 0},
                                               "results": [{"csv_row": 1, "prediction_path": str(prediction_path)}]}])
        self.assertTrue(all(p == initial[0] for p in initial[1:]))

    def test_plain_react_failed_tool_preserves_evidence_budget_and_advances_annealing(self):
        measure = {**finish(), "action": "measure", "predicted_id": None,
                   "focus_ids": [1, 2], "evidence": []}
        invalid = {**measure, "focus_ids": [999]}
        result, calls, _ = self.run_ground("A2", [invalid, measure, finish()], budget=1, repairs=1)
        trace = result["active_trace"]
        self.assertEqual(result["predicted_id"], 2)
        self.assertEqual(trace["react_budget"]["successful_tool_calls"], 1)
        self.assertEqual(trace["react_budget"]["failed_checks"], 1)
        states = [json.loads(c.args[0][1]["content"][0]["text"]) for c in calls]
        self.assertEqual([s["remaining_tool_calls"] for s in states], [1, 1, 0])
        self.assertEqual([s["policy_annealing"]["progress"] for s in states], [0, .5, 1])
        self.assertEqual(calls[-1].kwargs["response_schema"]["schema"]["properties"]["action"]["enum"], ["finish"])
        self.assertEqual(trajectory_metrics(trace), {"successful_tool_calls": 1, "decision_calls": 3,
                         "repair_calls": 1, "budget_exhausted": True, "repair_exhausted": False})

    def test_plain_react_repair_exhaustion_abstains(self):
        invalid = {**finish(), "action": "measure", "predicted_id": None,
                   "focus_ids": [999], "evidence": []}
        result, calls, _ = self.run_ground("A2", [invalid, invalid], budget=2, repairs=1)
        self.assertIsNone(result["predicted_id"])
        self.assertEqual(len(calls), 2)
        trace = result["active_trace"]
        self.assertEqual(trace["termination_cause"], "repair_exhausted")
        self.assertEqual(trace["react_budget"]["successful_tool_calls"], 0)

    def test_new_switches_and_schedule_enter_run_identity(self):
        with patch.dict(os.environ, {"EVIGROUND_POLICY_ANNEALING_ENABLED": "0"}):
            before = settings()
        with patch.dict(os.environ, {"EVIGROUND_POLICY_ANNEALING_ENABLED": "1"}):
            after = settings()
        self.assertNotEqual(before, after)
        self.assertIn("policy_annealing_schedule", after)
        for variable in ("EVIGROUND_POLICY_ANNEALING_ENABLED", "EVIGROUND_SUCCESSFUL_TOOL_BUDGET"):
            with patch.dict(os.environ, {variable: "invalid"}), self.assertRaises(ValueError):
                settings()
