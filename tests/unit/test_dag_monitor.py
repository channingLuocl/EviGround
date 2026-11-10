"""DAG transitions, reversible exclusions and grounding integration, without API calls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding import react
from eviground.grounding.dag_monitor import DagMonitor
from eviground.grounding.settings import dag_monitor_enabled, dag_repair_limit, settings
from tests.helpers import parsed, response


QUERY = "the middle chair"


def node(name, kind, dependencies=(), completed=False, excluded=()):
    return {"id": name, "goal": name, "kind": kind,
            "source_quotes": [QUERY], "condition_ids": ["q0"],
            "depends_on": list(dependencies),
            "allowed_actions": ["finish"] if kind == "verification" else ["inspect", "measure"],
            "status": "completed" if completed else "pending",
            "bound_object_ids": [1, 2, 3],
            "evidence_ids": ["initial"] if completed else [],
            "conclusion": "initial geometry supports this binding" if completed else "",
            "excluded_target_ids": list(excluded)}


def plan():
    return {"nodes": [node("group", "binding", completed=True),
                      node("compare", "comparison", ["group"]),
                      node("verify", "verification", ["compare"])],
            "current_node_id": "compare", "reopen_node_ids": [],
            "revision_reason": "resolve middle within the bound comparison group"}


def action(kind="measure", target=None):
    return {"action": kind, "reason": "check the middle condition", "focus_ids": [1, 2, 3]
            if kind != "finish" else [], "context": None,
            "azimuth_deg": None, "elevation_deg": None, "frame": None,
            "predicted_id": target,
            "evidence": [{"source_quote": QUERY, "object_ids": [target],
                          "observation": "comparison supports this chair"}]
            if target is not None else [], "unresolved": []}


class DagMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2},
                                  ["inspect", "measure", "finish"])

    def test_graph_array_order_does_not_determine_execution_order(self):
        p = plan()
        p["nodes"].reverse()
        event = self.monitor.update(p, action(), [])
        self.assertEqual(event["current_node_id"], "compare")
        self.assertEqual(event["active_target_ids"], [1, 2])

    def test_invalid_graphs_do_not_mutate_accepted_state(self):
        self.monitor.update(plan(), action(), [])
        before = deepcopy(self.monitor.plan)
        for variant in ("cycle", "absent", "duplicate", "future", "failed", "unknown_id",
                        "unquoted", "coverage", "unsupported_exclusion", "skip_prerequisite"):
            p = plan()
            if variant == "cycle":
                p["nodes"][0]["depends_on"] = ["compare"]
            elif variant == "absent":
                p["nodes"][1]["depends_on"] = ["missing"]
            elif variant == "duplicate":
                p["nodes"][1]["id"] = "group"
            elif variant in ("future", "failed"):
                p["nodes"][0]["evidence_ids"] = ["observation_1"]
            elif variant == "unknown_id":
                p["nodes"][0]["bound_object_ids"] = [999]
            elif variant == "unquoted":
                p["nodes"][0]["source_quotes"] = ["the red chair"]
            elif variant == "coverage":
                for n in p["nodes"]:
                    n["condition_ids"] = []
            elif variant == "unsupported_exclusion":
                p["nodes"][1]["excluded_target_ids"] = [2]
            else:
                p["nodes"][0].update(status="pending", conclusion="", evidence_ids=[])
            observations = [{"evidence_id": "observation_1", "tool": "measure",
                             "error": "bad frame", "no_evidence_acquired": True}]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.monitor.update(p, action(), observations)
            self.assertEqual(self.monitor.plan, before)
            self.assertEqual(len(self.monitor.history), 1)

    def test_finish_needs_unique_target_and_completed_final_stage(self):
        for variant in ("incomplete", "not_unique", "unresolved", "wrong_current"):
            p = plan()
            for n in p["nodes"]:
                n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
            p["nodes"][1]["excluded_target_ids"] = [2]
            p["current_node_id"] = "verify"
            a = action("finish", 1)
            if variant == "incomplete":
                p["nodes"][-1].update(status="pending", evidence_ids=[], conclusion="")
            elif variant == "not_unique":
                p["nodes"][1]["excluded_target_ids"] = []
            elif variant == "unresolved":
                a["unresolved"] = ["comparison membership uncertain"]
            else:
                p["current_node_id"] = "compare"
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.monitor.update(p, a, [])

    def test_rollback_restores_candidates_and_preserves_independent_branch(self):
        p = plan()
        p["nodes"][1].update(status="completed", conclusion="chair 2 is outside the middle",
                             evidence_ids=["initial"], excluded_target_ids=[2])
        p["nodes"].insert(2, node("appearance", "attribute", completed=True))
        p["nodes"].insert(3, node("check", "relation", ["compare"]))
        p["nodes"][-1]["depends_on"] = ["check", "appearance"]
        p["current_node_id"] = "check"
        self.monitor.update(p, action("inspect"), [])
        self.assertEqual(self.monitor.active_ids(), [1])
        revised = deepcopy(p)
        revised["reopen_node_ids"] = ["group"]
        revised["current_node_id"] = "group"
        # Even a stale completed downstream node is reset by the program.
        event = self.monitor.update(revised, action("inspect"), [])
        self.assertEqual(event["reactivated_target_ids"], [2])
        by_id = {n["id"]: n for n in self.monitor.plan["nodes"]}
        self.assertEqual(by_id["appearance"]["status"], "completed")
        self.assertEqual(by_id["compare"]["status"], "pending")
        self.assertEqual(by_id["compare"]["evidence_ids"], [])
        self.assertEqual(by_id["compare"]["excluded_target_ids"], [])
        self.assertEqual(self.monitor.history[0]["active_target_ids"], [1])
        # Target pruning does not erase the physical comparison members.
        self.assertEqual(by_id["compare"]["bound_object_ids"], [1, 2, 3])

    def test_revised_binding_automatically_invalidates_dependents(self):
        self.monitor.update(plan(), action(), [])
        p = plan()
        p["nodes"][0]["bound_object_ids"] = [1, 3]
        p["current_node_id"] = "group"
        event = self.monitor.update(p, action("inspect", None) | {"focus_ids": [1, 3]}, [])
        self.assertEqual(set(event["invalidated_node_ids"]), {"group", "compare", "verify"})

    def test_regenerated_downstream_nodes_cannot_keep_stale_completion(self):
        self.monitor.update(plan(), action(), [])
        p = plan()
        p["reopen_node_ids"] = ["group"]
        p["current_node_id"] = "group"
        p["nodes"][1].update(id="new_compare", status="completed",
                             evidence_ids=["initial"], conclusion="stale comparison",
                             excluded_target_ids=[2])
        p["nodes"][2]["depends_on"] = ["new_compare"]
        event = self.monitor.update(p, action("inspect"), [])
        replacement = self.monitor.plan["nodes"][1]
        self.assertEqual(replacement["status"], "pending")
        self.assertEqual(replacement["excluded_target_ids"], [])
        self.assertIn("new_compare", event["invalidated_node_ids"])
        self.assertEqual(event["active_target_ids"], [1, 2])

    def test_tool_can_examine_unbound_comparison_members(self):
        p = plan()
        p["nodes"][1]["bound_object_ids"] = [1, 2]
        event = self.monitor.update(p, action(), [])
        self.assertEqual(event["current_node_id"], "compare")
        self.assertEqual(self.monitor.plan["nodes"][1]["bound_object_ids"], [1, 2])

    def test_exclusions_use_cited_observations_not_only_winner_bindings(self):
        p = plan()
        p["nodes"][1].update(status="completed", bound_object_ids=[1],
                             evidence_ids=["observation_1"], conclusion="chair 2 is outside the group",
                             excluded_target_ids=[2])
        p["current_node_id"] = "verify"
        observation = {"evidence_id": "observation_1", "tool": "measure", "focus_ids": [1, 2, 3]}
        self.monitor.update(p, action("finish"), [observation])
        self.assertEqual(self.monitor.active_ids(), [1])

    def test_exclusions_cannot_cite_an_observation_of_other_objects(self):
        p = plan()
        p["nodes"][1].update(status="completed", evidence_ids=["observation_1"],
                             conclusion="chair 2 is outside the group", excluded_target_ids=[2])
        p["current_node_id"] = "verify"
        observation = {"evidence_id": "observation_1", "tool": "measure", "focus_ids": [1, 3]}
        with self.assertRaisesRegex(ValueError, "cited evidence"):
            self.monitor.update(p, action("finish"), [observation])

    def test_initial_evidence_can_rule_out_an_unbound_rival(self):
        p = plan()
        p["nodes"][1].update(status="completed", bound_object_ids=[1],
                             evidence_ids=["initial"], conclusion="initial geometry rules out chair 2",
                             excluded_target_ids=[2])
        p["current_node_id"] = "verify"
        self.monitor.update(p, action("finish"), [])
        self.assertEqual(self.monitor.active_ids(), [1])

    def test_reordering_completed_bindings_does_not_reopen_a_stage(self):
        self.monitor.update(plan(), action(), [])
        p = plan()
        p["nodes"][0]["bound_object_ids"].reverse()
        event = self.monitor.update(p, action(), [])
        self.assertEqual(event["invalidated_node_ids"], [])

    def test_each_parsed_condition_must_remain_in_the_plan(self):
        query_parse = parsed()
        query_parse["condition_records"] = [{"source_quote": "middle",
                                              "condition_type": "relation:other"}]
        monitor = DagMonitor(QUERY, query_parse, {1, 2, 3}, {1, 2},
                             ["inspect", "measure", "finish"])
        with self.assertRaisesRegex(ValueError, "cover all"):
            monitor.update(plan(), action(), [])
        p = plan()
        p["nodes"][1]["condition_ids"].append("q1")
        monitor.update(p, action(), [])

    def test_initial_evidence_only_mode_can_finish_without_fictitious_tools(self):
        monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2}, ["finish"])
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", conclusion="supported by initial evidence",
                     evidence_ids=["initial"])
            if n["kind"] != "verification":
                n["allowed_actions"] = []
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["current_node_id"] = "verify"
        monitor.update(p, action("finish", 1), [])
        self.assertEqual(monitor.active_ids(), [1])

    def test_switch_is_strict_and_enters_run_identity(self):
        with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "0"}):
            baseline = settings()
            self.assertFalse(dag_monitor_enabled())
        with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "1"}):
            self.assertTrue(dag_monitor_enabled())
            self.assertNotEqual(baseline, settings())
        with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                dag_monitor_enabled()

    def test_repair_budget_is_bounded_and_enters_run_identity(self):
        with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "1",
                                     "EVIGROUND_DAG_REPAIR_LIMIT": "6"}):
            self.assertEqual(dag_repair_limit(), 6)
            original = settings()
        with patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "1",
                                     "EVIGROUND_DAG_REPAIR_LIMIT": "3"}):
            self.assertNotEqual(settings(), original)
        for raw in ("-1", "abc", "1.5"):
            with patch.dict(os.environ, {"EVIGROUND_DAG_REPAIR_LIMIT": raw}):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    dag_repair_limit()


class DagGroundingTests(unittest.TestCase):
    def run_ground(self, responses, limit, alternative_category=False, repairs=1):
        objects = [{"bbox_id": i, "target": "chair", "bbox_3d": [i, 0, .5, 1, 1, 1],
                    "category_hypotheses": [{"category": "chair", "rank": 1,
                                             "gap_from_top1": 0}]}
                   for i in (1, 2, 3)]
        if alternative_category:
            objects[0]["category_hypotheses"] = [
                {"category": "stool", "rank": 1, "gap_from_top1": 0},
                {"category": "chair", "rank": 2, "gap_from_top1": .2}]
        views = [{"image": "initial.png", "kind": "overview"},
                 {"image": "top.png", "kind": "top"}]
        trace = {"steps": [], "internal_usage": {"render": 0, "compute": 0, "visual_check": 0}}
        with (tempfile.TemporaryDirectory() as out,
              patch.dict(os.environ, {"EVIGROUND_DAG_MONITOR_ENABLED": "1",
                                      "EVIGROUND_DAG_REPAIR_LIMIT": str(repairs),
                                      "EVIGROUND_INSPECT_ENABLED": "1", "EVIGROUND_MEASURE_ENABLED": "1",
                                      "EVIGROUND_CATEGORY_INSPECTION_REQUIRED": "1"}),
              patch("eviground.rendering.overview.scene_views", return_value=views),
              patch.object(react, "image_part", side_effect=lambda p: {"image": p}),
              patch.object(react, "chat", side_effect=[response(r) for r in responses]) as chat):
            result = react.ground(QUERY, parsed(), objects, objects[:2],
                                  np.zeros((3, 3)), np.zeros((3, 3)), np.arange(3),
                                  Path(out), [], trace, limit)
        return result, chat.call_args_list

    def verified_plan(self, evidence_id):
        p = plan()
        for n in p["nodes"][1:]:
            n.update(status="completed", evidence_ids=[evidence_id], conclusion="comparison supports chair 1")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["current_node_id"] = "verify"
        return p

    def test_plan_repair_preserves_the_full_evidence_budget(self):
        bad = plan()
        bad["nodes"][0]["evidence_ids"] = ["future"]
        responses = [action() | {"monitor": bad}, action() | {"monitor": plan()},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        result, calls = self.run_ground(responses, 1)
        self.assertEqual(result["predicted_id"], 1)
        state = json.loads(calls[1].args[0][1]["content"][0]["text"])
        self.assertEqual(state["remaining_tool_calls"], 1)
        self.assertEqual(state["remaining_plan_repairs"], 0)
        self.assertEqual(result["active_trace"]["dag_monitor_budget"]["successful_tool_calls"], 1)
        self.assertEqual(result["active_trace"]["dag_monitor_budget"]["failed_checks"], 1)

    def test_final_plan_can_be_repaired_after_tools_are_exhausted(self):
        responses = [action() | {"monitor": plan()},
                     action("finish", 1) | {"monitor": plan()},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_1")}]
        result, calls = self.run_ground(responses, 1)
        self.assertEqual(result["predicted_id"], 1)
        self.assertEqual(len(calls), 3)
        for call in calls[1:]:
            self.assertEqual(call.kwargs["response_schema"]["schema"]["properties"]["action"]["enum"], ["finish"])

    def test_invalid_tool_request_uses_repair_allowance_not_evidence(self):
        responses = [action() | {"focus_ids": [], "monitor": plan()},
                     action() | {"monitor": plan()},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        result, calls = self.run_ground(responses, 1)
        self.assertEqual(result["predicted_id"], 1)
        state = json.loads(calls[1].args[0][1]["content"][0]["text"])
        self.assertEqual(state["remaining_tool_calls"], 1)
        self.assertNotIn("observation_1", state["supporting_evidence_ids"])
        self.assertEqual(result["active_trace"]["dag_monitor_budget"]["failed_checks"], 1)

    def test_repeated_invalid_plans_cannot_retry_indefinitely(self):
        invalid = action("finish", 1) | {"monitor": plan()}
        result, calls = self.run_ground([invalid] * 3, 12, repairs=2)
        self.assertIsNone(result["predicted_id"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["active_trace"]["dag_monitor_budget"]["successful_tool_calls"], 0)
        self.assertIn("repair budget", result["decision_reasons"][0])

    def test_measure_then_verified_unique_finish_preserves_full_candidate_pool(self):
        p = plan()
        first = action() | {"monitor": p}
        finished = deepcopy(p)
        for n in finished["nodes"][1:]:
            n.update(status="completed", evidence_ids=["observation_1"], conclusion="comparison supports chair 1")
        finished["nodes"][1]["excluded_target_ids"] = [2]
        finished["current_node_id"] = "verify"
        result, calls = self.run_ground([first, action("finish", 1) | {"monitor": finished}], 1)
        self.assertEqual(result["predicted_id"], 1)
        trace = result["active_trace"]
        self.assertEqual(trace["stop_reason"], "dag_monitor_finish")
        self.assertEqual(trace["dag_monitor_active_target_ids"], [1])
        self.assertEqual(trace["react_initial_input"]["allowed_target_ids"], [1, 2])
        state = json.loads(calls[1].args[0][1]["content"][0]["text"])
        self.assertEqual(state["observations"][0]["evidence_id"], "observation_1")
        self.assertEqual(state["task_dag"]["current_node_id"], "compare")
        self.assertIn("monitor", calls[0].kwargs["response_schema"]["schema"]["required"])

    def test_invalid_finish_at_budget_abstains_instead_of_forcing_a_target(self):
        invalid = action("finish", 1) | {"monitor": plan()}
        result, calls = self.run_ground([invalid, invalid], 1)
        self.assertIsNone(result["predicted_id"])
        self.assertEqual(result["active_trace"]["stop_reason"], "dag_monitor_abstained")
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["scene_evidence"]["unresolved"])
        self.assertEqual(result["active_trace"]["internal_usage"]["compute"], 0)

    def test_monitor_does_not_bypass_alternative_category_inspection_guard(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["current_node_id"] = "verify"
        finished = action("finish", 1) | {"monitor": p}
        result, _ = self.run_ground([finished, finished], 1, alternative_category=True)
        self.assertIsNone(result["predicted_id"])
        self.assertIn("alternative-only", result["scene_evidence"]["unresolved"][0])
