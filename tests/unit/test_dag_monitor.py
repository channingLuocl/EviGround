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
                        "unquoted", "coverage"):
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
            observations = [{"evidence_id": "observation_1", "tool": "measure",
                             "error": "bad frame", "no_evidence_acquired": True}]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.monitor.update(p, action(), observations)
            self.assertEqual(self.monitor.plan, before)
            self.assertEqual(len(self.monitor.history), 1)

    def test_rollback_is_retained_when_a_stale_finish_is_rejected(self):
        self.monitor.update(plan(), action(), [])
        revised = plan()
        for n in revised["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        revised["nodes"][0]["conclusion"] = "revised group binding"
        revised["nodes"][1]["excluded_target_ids"] = [2]
        revised["current_node_id"] = "verify"
        with self.assertRaisesRegex(ValueError, "Rollback accepted"):
            self.monitor.update(revised, action("finish", 1), [])
        self.assertEqual(len(self.monitor.history), 2)
        self.assertEqual(self.monitor.plan["nodes"][0]["status"], "pending")
        self.assertEqual(self.monitor.active_ids(), [1, 2])
        self.assertEqual(self.monitor.context([])["ready_node_ids"], ["group"])
        repaired = deepcopy(self.monitor.plan)
        for n in repaired["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="rechecked against initial evidence")
        repaired["nodes"][1]["excluded_target_ids"] = [2]
        self.monitor.update(repaired, action("finish", 1), [])
        self.assertEqual(self.monitor.active_ids(), [1])

    def test_verified_finish_derives_a_missing_redundant_final_binding(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["nodes"][-1]["bound_object_ids"] = []
        p["current_node_id"] = "verify"
        event = self.monitor.update(p, action("finish", 1), [])
        self.assertEqual(self.monitor.plan["nodes"][-1]["bound_object_ids"], [1])
        self.assertTrue(event["normalized_final_binding"])
        self.assertEqual(p["nodes"][-1]["bound_object_ids"], [])

    def test_final_binding_normalization_does_not_bypass_verification(self):
        for variant in ("incomplete", "not_unique", "unresolved", "conflicting_binding"):
            p = plan()
            for n in p["nodes"]:
                n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
            p["nodes"][1]["excluded_target_ids"] = [2]
            p["nodes"][-1]["bound_object_ids"] = []
            p["current_node_id"] = "verify"
            a = action("finish", 1)
            if variant == "incomplete":
                p["nodes"][-1].update(status="pending", evidence_ids=[], conclusion="")
            elif variant == "not_unique":
                p["nodes"][1]["excluded_target_ids"] = []
            elif variant == "unresolved":
                a["unresolved"] = ["observer frame uncertain"]
            else:
                p["nodes"][-1]["bound_object_ids"] = [2]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.monitor.update(p, a, [])

    def test_repeated_missing_final_binding_does_not_trigger_false_rollback(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["nodes"][-1]["bound_object_ids"] = []
        p["current_node_id"] = "verify"
        self.monitor.update(p, action("finish", 1), [])
        event = self.monitor.update(p, action("finish", 1), [])
        self.assertEqual(event["invalidated_node_ids"], [])
        self.assertEqual(self.monitor.plan["nodes"][-1]["status"], "completed")

    def test_malformed_rollback_does_not_change_accepted_state(self):
        self.monitor.update(plan(), action(), [])
        before = deepcopy(self.monitor.plan)
        revised = plan()
        revised["nodes"][0]["conclusion"] = "revised group binding"
        revised["nodes"][1]["depends_on"] = ["verify"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.monitor.update(revised, action("finish", 1), [])
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

    def test_tool_continues_with_unobserved_rivals_retained(self):
        p = plan()
        p["nodes"][0].update(evidence_ids=["observation_1"], excluded_target_ids=[2])
        observation = {"evidence_id": "observation_1", "tool": "render", "focus_ids": [1, 3]}
        event = self.monitor.update(p, action(), [observation])
        self.assertEqual(self.monitor.active_ids(), [1, 2])
        self.assertEqual(self.monitor.plan["nodes"][0]["excluded_target_ids"], [])
        self.assertEqual(event["discarded_exclusions"], [{"node_id": "group", "target_ids": [2],
                                                        "reason": "outside_cited_evidence"}])
        self.assertEqual(p["nodes"][0]["excluded_target_ids"], [2])
        self.assertEqual(self.monitor.context([observation])["discarded_exclusions"],
                         event["discarded_exclusions"])

    def test_pending_node_cannot_exclude_targets_but_can_acquire_evidence(self):
        p = plan()
        p["nodes"][1]["excluded_target_ids"] = [2]
        event = self.monitor.update(p, action(), [])
        self.assertEqual(self.monitor.active_ids(), [1, 2])
        self.assertEqual(self.monitor.plan["nodes"][1]["status"], "pending")
        self.assertEqual(self.monitor.plan["nodes"][1]["excluded_target_ids"], [])
        self.assertEqual(event["discarded_exclusions"][0]["reason"], "node_not_completed")

    def test_exclusions_cannot_make_unobserved_rivals_ineligible_at_finish(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["observation_1"], conclusion="supported")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["nodes"][-1]["bound_object_ids"] = []
        p["current_node_id"] = "verify"
        observation = {"evidence_id": "observation_1", "tool": "measure", "focus_ids": [1, 3]}
        with self.assertRaisesRegex(ValueError, "active targets: \\[1, 2\\]"):
            self.monitor.update(p, action("finish", 1), [observation])
        self.assertEqual(self.monitor.active_ids(), [1, 2])
        self.assertEqual(self.monitor.plan["nodes"][1]["excluded_target_ids"], [])
        self.assertEqual(self.monitor.plan["nodes"][-1]["bound_object_ids"], [])
        self.assertNotIn("normalized_final_binding", self.monitor.history[-1])
        self.assertIn("action_error", self.monitor.history[-1])

    def test_discarding_exclusions_preserves_supported_exclusions(self):
        monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2, 3}, ["inspect", "measure", "finish"])
        p = plan()
        p["nodes"][0].update(evidence_ids=["observation_1"], excluded_target_ids=[2, 3])
        observation = {"evidence_id": "observation_1", "tool": "measure", "objects": [{"id": 1}, {"id": 2}]}
        monitor.update(p, action(), [observation])
        self.assertEqual(monitor.plan["nodes"][0]["excluded_target_ids"], [2])
        self.assertEqual(monitor.active_ids(), [1, 3])

    def test_malformed_graph_cannot_commit_discarded_exclusions(self):
        self.monitor.update(plan(), action(), [])
        before = deepcopy(self.monitor.plan)
        p = plan()
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["nodes"][0]["depends_on"] = ["verify"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.monitor.update(p, action(), [])
        self.assertEqual(self.monitor.plan, before)
        self.assertEqual(len(self.monitor.history), 1)

    def test_context_reports_only_usable_evidence_object_scopes(self):
        observations = [{"tool": "render", "evidence_id": "observation_1", "focus_ids": [1, 3]},
                        {"tool": "measure", "evidence_id": "observation_2", "objects": [{"id": 2}]},
                        {"tool": "measure", "evidence_id": "observation_3", "focus_ids": [2],
                         "error": "failed", "no_evidence_acquired": True},
                        {"tool": "monitor_check", "evidence_id": "observation_4", "focus_ids": [2]}]
        self.assertEqual(self.monitor.context(observations)["supporting_evidence_objects"],
                         {"initial": [1, 2, 3], "observation_1": [1, 3], "observation_2": [2]})

    def test_required_category_inspection_reopens_a_nonterminal_stage(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["current_node_id"] = "verify"
        self.monitor.update(p, action("finish", 1), [])
        event = self.monitor.require_inspection(1, [])
        stages = {n["id"]: n for n in self.monitor.plan["nodes"]}
        self.assertEqual(self.monitor.plan["current_node_id"], "compare")
        self.assertEqual(self.monitor.ready_ids(), ["compare"])
        self.assertEqual(stages["group"]["status"], "completed")
        self.assertEqual(stages["compare"]["status"], "pending")
        self.assertIn("inspect", stages["compare"]["allowed_actions"])
        self.assertEqual(stages["verify"]["status"], "pending")
        self.assertEqual(stages["verify"]["allowed_actions"], ["finish"])
        self.assertEqual(self.monitor.active_ids(), [1, 2])
        self.assertEqual(event["required_inspection_ids"], [1])
        self.assertEqual(event["transition"], "category_inspection_required")
        self.assertEqual(self.monitor.evidence_ids([]), {"initial"})

    def test_inspection_reopening_does_not_add_nodes_to_a_full_graph(self):
        nodes = [node("group", "binding", completed=True)]
        for i in range(10):
            nodes.append(node(f"stage_{i}", "relation", [nodes[-1]["id"]], completed=True))
        nodes[-1]["excluded_target_ids"] = [2]
        nodes.append(node("verify", "verification", [nodes[-1]["id"]], completed=True))
        p = {"nodes": nodes, "current_node_id": "verify", "reopen_node_ids": [], "revision_reason": "verify"}
        self.monitor.update(p, action("finish", 1), [])
        self.monitor.require_inspection(1, [])
        self.assertEqual(len(self.monitor.plan["nodes"]), 12)
        self.assertEqual(self.monitor.ready_ids(), ["stage_9"])

    def test_inspection_reopening_keeps_reasoning_only_descendants_executable(self):
        p = plan()
        for n in p["nodes"]:
            n.update(status="completed", evidence_ids=["initial"], conclusion="supported")
        p["nodes"][0]["bound_object_ids"] = [1]
        p["nodes"][1].update(bound_object_ids=[3], allowed_actions=[], excluded_target_ids=[2])
        p["current_node_id"] = "verify"
        self.monitor.update(p, action("finish", 1), [])
        self.monitor.require_inspection(1, [])
        self.assertEqual(self.monitor.plan["current_node_id"], "group")
        self.assertEqual(self.monitor.plan["nodes"][1]["allowed_actions"], ["inspect"])
        self.monitor.update(deepcopy(self.monitor.plan), action("inspect"), [])

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

    def test_rejected_stale_finish_supplies_reset_plan_to_the_next_turn(self):
        revised = self.verified_plan("observation_1")
        revised["nodes"][0]["conclusion"] = "revised group binding"
        repaired = deepcopy(revised)
        for n in repaired["nodes"]:
            n["evidence_ids"] = ["observation_1"]
        responses = [action() | {"monitor": plan()},
                     action("finish", 1) | {"monitor": revised},
                     action("finish", 1) | {"monitor": repaired}]
        result, calls = self.run_ground(responses, 1)
        self.assertEqual(result["predicted_id"], 1)
        state = json.loads(calls[2].args[0][1]["content"][0]["text"])
        self.assertEqual(state["ready_node_ids"], ["group"])
        self.assertTrue(all(n["status"] == "pending" for n in state["task_dag"]["nodes"]))
        self.assertEqual(state["active_target_ids"], [1, 2])
        self.assertIn("observation_1", state["supporting_evidence_ids"])
        self.assertEqual(state["remaining_tool_calls"], 0)
        self.assertIn("Rollback accepted", state["observations"][-1]["error"])
        history = result["active_trace"]["dag_monitor_history"]
        self.assertEqual(len(history), 3)
        self.assertIn("action_error", history[1])
        self.assertEqual(result["active_trace"]["dag_monitor_budget"]["failed_checks"], 1)

    def test_safe_exclusion_correction_allows_the_requested_next_tool(self):
        first = plan()
        first["nodes"][0].update(status="pending", evidence_ids=[], conclusion="")
        first["current_node_id"] = "group"
        second = plan()
        second["nodes"][0].update(evidence_ids=["observation_1"], excluded_target_ids=[2])
        responses = [action() | {"focus_ids": [1, 3], "monitor": first},
                     action() | {"monitor": second},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        result, calls = self.run_ground(responses, 2, repairs=0)
        self.assertEqual(result["predicted_id"], 1)
        state = json.loads(calls[2].args[0][1]["content"][0]["text"])
        self.assertEqual(state["supporting_evidence_objects"]["observation_1"], [1, 3])
        self.assertEqual(state["supporting_evidence_objects"]["observation_2"], [1, 2, 3])
        self.assertEqual(state["discarded_exclusions"][0]["target_ids"], [2])
        self.assertEqual(state["active_target_ids"], [1, 2])
        budget = result["active_trace"]["dag_monitor_budget"]
        self.assertEqual(budget["successful_tool_calls"], 2)
        self.assertEqual(budget["failed_checks"], 0)

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

    def test_category_guard_routes_finish_back_to_legal_inspection(self):
        pending = plan()
        responses = [action("finish", 1) | {"monitor": self.verified_plan("initial")},
                     action("inspect") | {"monitor": pending},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        with patch("eviground.rendering.inspection.render_action",
                   return_value={"image": "inspected.png", "marked_ids": [1, 2, 3]}):
            result, calls = self.run_ground(responses, 1, alternative_category=True)
        self.assertEqual(result["predicted_id"], 1)
        state = json.loads(calls[1].args[0][1]["content"][0]["text"])
        self.assertEqual(state["required_inspection_ids"], [1])
        self.assertEqual(state["task_dag"]["current_node_id"], "compare")
        self.assertEqual(state["ready_node_ids"], ["compare"])
        self.assertEqual(state["task_dag"]["nodes"][-1]["allowed_actions"], ["finish"])
        self.assertEqual(state["task_dag"]["nodes"][-1]["status"], "pending")
        after_inspect = json.loads(calls[2].args[0][1]["content"][0]["text"])
        self.assertEqual(after_inspect["required_inspection_ids"], [])
        budget = result["active_trace"]["dag_monitor_budget"]
        self.assertEqual(budget["successful_tool_calls"], 1)
        self.assertEqual(budget["failed_checks"], 1)
        self.assertEqual(budget["decision_calls"], 3)

    def test_category_reopening_does_not_treat_requested_focus_as_visible(self):
        responses = [action("finish", 1) | {"monitor": self.verified_plan("initial")},
                     action("inspect") | {"monitor": plan()},
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        with patch("eviground.rendering.inspection.render_action",
                   return_value={"image": "inspected.png", "marked_ids": [2, 3]}):
            result, calls = self.run_ground(responses, 1, alternative_category=True)
        self.assertIsNone(result["predicted_id"])
        state = json.loads(calls[2].args[0][1]["content"][0]["text"])
        self.assertEqual(state["required_inspection_ids"], [1])
        self.assertIn("visibly inspected", result["scene_evidence"]["unresolved"][0])
