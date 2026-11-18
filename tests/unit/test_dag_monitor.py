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
            if target is not None else [],
            "unresolved": ["The middle comparison remains unverified"]
            if kind == "finish" and target is None else []}


class DagMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2},
                                  ["inspect", "measure", "finish"])

    def test_final_dependency_feedback_names_missing_nodes_atomically(self):
        self.monitor.update(plan(), action(), [])
        before, history = deepcopy(self.monitor.plan), deepcopy(self.monitor.history)
        invalid = plan()
        invalid['nodes'][-1]['depends_on'] = ['group']
        with self.assertRaisesRegex(ValueError, r"missing node IDs \['compare'\]") as raised:
            self.monitor.update(invalid, action(), [])
        self.assertNotIn("'group'", str(raised.exception))
        self.assertNotIn("'verify'", str(raised.exception))
        self.assertEqual(self.monitor.plan, before)
        self.assertEqual(self.monitor.history, history)

    def test_compact_schema_blocks_observed_invalid_graphs_before_decoding(self):
        from jsonschema import Draft202012Validator, ValidationError, validate

        schema = self.monitor.schema(compact=True)
        Draft202012Validator.check_schema(schema)
        p = plan()
        mapping = {n["id"]: f"n{i}" for i, n in enumerate(p["nodes"])}
        mapping["verify"] = "n11"
        for n in p["nodes"]:
            n["id"] = mapping[n["id"]]
            n["depends_on"] = [mapping[d] for d in n["depends_on"]]
        p["current_node_id"] = mapping[p["current_node_id"]]
        wire = deepcopy(p)
        wire["requirement_nodes"] = {r["id"]: [n["id"] for n in wire["nodes"]
                                                if r["id"] in n["condition_ids"]]
                                     for r in self.monitor.requirements}
        for n in wire["nodes"]:
            del n["condition_ids"]
        wire["verification"] = wire["nodes"].pop()
        validate(wire, schema)
        self.assertEqual(self.monitor.encode(p), wire)
        self.assertEqual(self.monitor.decode(wire), p)
        self.assertEqual(self.monitor.encode(wire), wire)
        state = {"observations": [], "task_dag": p,
                 "previous_actions": [{"action": "measure", "monitor": p}]}
        compact = react.compact_state(state, self.monitor)
        validate(compact["task_dag"], schema)
        restored = deepcopy(compact["previous_actions"][0]["monitor"])
        for key, value in restored.items():
            if isinstance(value, dict) and set(value) == {"same_as_task_dag"}:
                restored[key] = compact["task_dag"][value["same_as_task_dag"]]
        validate(restored, schema)
        self.assertEqual(compact["task_dag"], wire)
        self.assertEqual(state["task_dag"], p)
        self.monitor.update(p, action(), [])
        for variant in ("no_tasks", "text_dependency", "condition_as_node", "verification_tool",
                        "missing_requirement", "missing_verification", "pending_exclusion",
                        "empty_quotes", "placeholder_quote"):
            invalid = deepcopy(wire)
            if variant == "no_tasks":
                invalid["nodes"] = []
            elif variant == "text_dependency":
                invalid["nodes"][1]["depends_on"] = ["bind reference"]
            elif variant == "condition_as_node":
                invalid["nodes"][0]["id"] = "q0"
            elif variant == "verification_tool":
                invalid["verification"]["allowed_actions"] = ["inspect", "measure", "finish"]
            elif variant == "missing_requirement":
                del invalid["requirement_nodes"]["q0"]
            elif variant == "missing_verification":
                del invalid["verification"]
            elif variant == "pending_exclusion":
                invalid["nodes"][1]["excluded_target_ids"] = [1]
            else:
                invalid["nodes"][1]["source_quotes"] = [] if variant == "empty_quotes" else ["<exact description substring>"]
            with self.subTest(variant=variant), self.assertRaises(ValidationError):
                validate(invalid, schema)
        absent = deepcopy(wire)
        absent["requirement_nodes"]["q0"] = ["n10"]
        with self.assertRaisesRegex(ValueError, "absent DAG node"):
            self.monitor.decode(absent)
        no_tools = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2}, ["finish"])
        Draft202012Validator.check_schema(no_tools.schema(compact=True))

    def test_compact_requirements_distinguish_prose_from_node_dependencies(self):
        requirement = {"id": "q1", "dependencies": ["bind reference"]}
        state = {"observations": [], "monitor_requirements": [requirement]}
        compact = react.compact_state(state)
        self.assertEqual(compact["monitor_requirements"],
                         [{"id": "q1", "prerequisite_descriptions": ["bind reference"]}])
        self.assertIn("dependencies", requirement)

    def test_compact_example_keeps_full_description_check_at_verification(self):
        from eviground.grounding.dag_monitor import COMPACT_INSTRUCTIONS
        from jsonschema import validate

        start = COMPACT_INSTRUCTIONS.index('{"nodes":')
        end = COMPACT_INSTRUCTIONS.index('\nReplace quoted placeholders', start)
        example = json.loads(COMPACT_INSTRUCTIONS[start:end].replace(
            '<exact description substring>', QUERY))
        validate(example, self.monitor.schema(compact=True))
        self.assertEqual(example['requirement_nodes']['q0'], ['n11'])
        self.assertEqual(example['nodes'][0]['depends_on'], [])
        self.assertEqual(example['verification']['depends_on'], ['n0'])
        event = self.monitor.update(self.monitor.decode(example), action(), [])
        final = next(n for n in event['plan']['nodes'] if n['kind'] == 'verification')
        self.assertEqual(final['condition_ids'], ['q0'])
        self.assertEqual(self.monitor.ready_ids(), ['n0'])

    def test_compact_set_bounds_block_observed_repeated_id_generation(self):
        from jsonschema import ValidationError, validate

        schema = self.monitor.schema(compact=True)
        branches = schema["properties"]["nodes"]["items"]["anyOf"]
        branches += schema["properties"]["verification"]["anyOf"]
        for branch in branches:
            properties = branch["properties"]
            for field, valid, repeated in (
                ("excluded_target_ids", [1, 2], [1, 2] * 2000),
                ("bound_object_ids", [1, 2, 3], [1, 2, 3] * 2000),
                ("depends_on", [f"n{i}" for i in range(11)], ["n0"] * 12),
            ):
                with self.subTest(kind=properties["kind"], field=field):
                    if field == "excluded_target_ids" and properties["status"]["enum"] == ["pending"]:
                        valid = []
                    validate(valid, properties[field])
                    with self.assertRaises(ValidationError):
                        validate(repeated, properties[field])
        validate([f"n{i}" for i in range(12)], schema["properties"]["reopen_node_ids"])
        with self.assertRaises(ValidationError):
            validate(["n0"] * 13, schema["properties"]["reopen_node_ids"])
        # Standard protocol remains available without compact wire limits.
        self.assertNotIn("maxItems", self.monitor.schema()["properties"]["nodes"])

    def test_compact_completed_nodes_only_cite_current_evidence(self):
        from jsonschema import ValidationError, validate

        schema = self.monitor.schema(compact=True, evidence_ids={"initial", "observation_1"})
        completed = schema["properties"]["nodes"]["items"]["anyOf"][1]["properties"]
        validate(["initial", "observation_1"], completed["evidence_ids"])
        with self.assertRaises(ValidationError):
            validate(["observation_2"], completed["evidence_ids"])

    def test_compact_action_sets_reject_duplicates_without_order_assumptions(self):
        from jsonschema import ValidationError, validate

        pending, completed = self.monitor.schema(compact=True)["properties"]["nodes"]["items"]["anyOf"]
        for actions in (["inspect"], ["measure"], ["inspect", "measure"], ["measure", "inspect"]):
            validate(actions, pending["properties"]["allowed_actions"])
        validate([], completed["properties"]["allowed_actions"])
        for actions in ([], ["measure", "measure"], ["finish"]):
            with self.subTest(actions=actions), self.assertRaises(ValidationError):
                validate(actions, pending["properties"]["allowed_actions"])

    def test_action_schema_emits_monitor_first_and_shares_its_contract(self):
        from jsonschema import Draft202012Validator, validate

        base = {"type": "object", "properties": {
            "action": {"enum": ["finish"]}, "focus_ids": {"type": "array", "items": {"type": "integer"}},
            "context": {}, "azimuth_deg": {}, "elevation_deg": {}, "frame": {},
            "predicted_id": {"enum": [1, None]},
            "unresolved": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "object_ids": {"type": "array", "items": {"type": "integer"}}}}},
            "monitor": {"type": "object", "properties": {"sentinel": {"const": True}},
                        "required": ["sentinel"], "additionalProperties": False}},
            "required": ["action", "focus_ids", "monitor", "unresolved"]}
        before = deepcopy(base)
        schema = react.constrained_action_schema(base, ["finish"])
        Draft202012Validator.check_schema(schema)
        for variant in schema["anyOf"]:
            self.assertEqual(next(iter(variant["properties"])), "monitor")
            self.assertEqual(variant["required"][0], "monitor")
            self.assertEqual(variant["properties"]["monitor"], {"$ref": "#/$defs/monitor"})
        self.assertEqual(schema["$defs"]["monitor"], base["properties"]["monitor"])
        validate({"monitor": {"sentinel": True}, "action": "finish", "focus_ids": [],
                  "predicted_id": None, "unresolved": ["The reference is not visible"]}, schema)
        self.assertEqual(base, before)

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
        from eviground.evaluation.costs import revision_metrics

        self.assertEqual(revision_metrics(dict(dag_monitor_history=self.monitor.history)),
                         dict(dag_reopened=True, candidates_restored=True, revision_triggered=True))
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
    def test_dialogue_retains_actions_evidence_and_error_without_repeating_images(self):
        p = plan()
        mapping = {"group": "n0", "compare": "n1", "verify": "n11"}
        for n in p["nodes"]:
            n["id"] = mapping[n["id"]]
            n["depends_on"] = [mapping[d] for d in n["depends_on"]]
        p["current_node_id"] = "n1"
        monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2}, ["inspect", "measure", "finish"])
        first = action() | {"monitor": monitor.encode(p)}
        finished = deepcopy(p)
        for n in finished["nodes"][1:]:
            n.update(status="completed", evidence_ids=["observation_1"], conclusion="comparison supports chair 1")
        finished["nodes"][1]["excluded_target_ids"] = [2]
        finished["current_node_id"] = "n11"
        responses = [deepcopy(first), deepcopy(first),
                     action("finish", 1) | {"monitor": monitor.encode(finished)}]
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact", "EVIGROUND_DIALOGUE_HISTORY": "1"}):
            result, calls = self.run_ground(responses, 2, repairs=1)
        self.assertEqual(result["predicted_id"], 1)
        messages = calls[-1].args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(json.loads(messages[2]["content"]), responses[0])
        self.assertNotIn("previous_actions", json.loads(messages[1]["content"][0]["text"]))
        feedbacks = [json.loads(m["content"][-1]["text"]) for m in messages[1:] if m["role"] == "user"]
        self.assertEqual([len(f["observations"]) for f in feedbacks], [0, 1, 1])
        self.assertIn("already acquired", feedbacks[-1]["observations"][0]["error"])
        self.assertEqual(feedbacks[-1]["supporting_evidence_ids"], ["initial", "observation_1"])
        self.assertEqual(feedbacks[-1]["remaining_tool_calls"], 1)
        self.assertEqual(feedbacks[-1]["remaining_plan_repairs"], 0)
        self.assertEqual(sum("image" in part for m in messages[1:] if m["role"] == "user"
                             for part in m["content"]), 2)
        self.assertEqual(len(result["active_trace"]["react_observations"]), 2)

    def test_dialogue_switch_is_strict_and_audited(self):
        with patch.dict(os.environ, {"EVIGROUND_DIALOGUE_HISTORY": "0"}):
            before = settings()
        with patch.dict(os.environ, {"EVIGROUND_DIALOGUE_HISTORY": "1"}):
            self.assertTrue(settings()["dialogue_history_enabled"])
            self.assertNotEqual(before, settings())
        with patch.dict(os.environ, {"EVIGROUND_DIALOGUE_HISTORY": "true"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                settings()

    def test_compact_next_turn_ends_with_current_evidence_and_matching_dag_wire(self):
        from jsonschema import validate

        monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2},
                             ["inspect", "measure", "finish"])
        p = plan()
        mapping = {"group": "n0", "compare": "n1", "verify": "n11"}
        for n in p["nodes"]:
            n["id"] = mapping[n["id"]]
            n["depends_on"] = [mapping[d] for d in n["depends_on"]]
        p["current_node_id"] = "n1"
        finished = deepcopy(p)
        for n in finished["nodes"][1:]:
            n.update(status="completed", evidence_ids=["observation_1"], conclusion="middle comparison supports chair 1")
        finished["nodes"][1]["excluded_target_ids"] = [2]
        finished["current_node_id"] = "n11"
        responses = [action() | {"monitor": monitor.encode(p)},
                     action("finish", 1) | {"monitor": monitor.encode(finished)}]
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact"}):
            result, calls = self.run_ground(responses, 1)
        self.assertEqual(result["predicted_id"], 1)
        content = calls[1].args[0][1]["content"]
        static = json.loads(content[0]["text"])
        feedback = json.loads(content[-1]["text"])
        self.assertNotIn("observations", static)
        self.assertIn("image", content[-2])
        self.assertEqual(feedback["observations"][0]["evidence_id"], "observation_1")
        self.assertEqual(feedback["remaining_tool_calls"], 0)
        from tests.unit.test_prompt_tables import expand
        validate(expand(feedback["task_dag"]), monitor.schema(compact=True))
        history_monitor = expand(static["previous_actions"][0]["monitor"])
        for key, value in history_monitor.items():
            if isinstance(value, dict) and set(value) == {"same_as_task_dag"}:
                history_monitor[key] = expand(feedback["task_dag"])[value["same_as_task_dag"]]
        validate(history_monitor, monitor.schema(compact=True))

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

    def test_no_repeat_schema_bans_only_successfully_acquired_observations(self):
        from jsonschema import Draft202012Validator

        bad = plan()
        bad["nodes"][0]["evidence_ids"] = ["future"]
        requested = action() | {"monitor": plan()}
        responses = [action() | {"monitor": bad}, requested,
                     action("finish", 1) | {"monitor": self.verified_plan("observation_2")}]
        with patch.dict(os.environ, {"EVIGROUND_ACTION_NO_REPEAT": "1",
                                     "EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            result, calls = self.run_ground(responses, 2)
        self.assertEqual(result["predicted_id"], 1)
        first = calls[0].kwargs["response_schema"]["schema"]
        retry = calls[1].kwargs["response_schema"]["schema"]
        after_acquisition = calls[2].kwargs["response_schema"]["schema"]
        self.assertNotIn("allOf", first)
        self.assertNotIn("allOf", retry)
        self.assertEqual(len(after_acquisition["allOf"]), 1)
        self.assertFalse(Draft202012Validator(after_acquisition).is_valid(requested))
        finish = responses[-1]
        self.assertTrue(Draft202012Validator(after_acquisition).is_valid(finish))
        state = json.loads(calls[2].args[0][1]["content"][0]["text"])
        self.assertEqual(state["unavailable_evidence_actions"], [react.evidence_action_key(requested)])
        self.assertTrue(result["active_trace"]["action_no_repeat"])

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

    def test_production_budget_allows_ten_tools_six_repairs_and_one_finish(self):
        invalid = action() | {'focus_ids': [], 'monitor': plan()}
        inspections = [action('inspect') | {'azimuth_deg': i * 30, 'monitor': plan()}
                       for i in range(10)]
        finish = action('finish', 1) | {'monitor': self.verified_plan('observation_16')}
        with patch('eviground.rendering.inspection.render_action',
                   return_value=dict(image='inspection.png', marked_ids=[1, 2, 3])):
            result, calls = self.run_ground([invalid] * 6 + inspections + [finish], 10, repairs=6)
        budget = result['active_trace']['react_budget']
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(len(calls), 17)
        self.assertEqual(budget['max_decision_calls'], 17)
        self.assertEqual(budget['successful_tool_calls'], 10)
        self.assertEqual(budget['failed_checks'], 6)
        self.assertEqual(budget['repair_calls'], 6)
        schema = calls[-1].kwargs['response_schema']['schema']
        self.assertEqual(schema['properties']['action']['enum'], ['finish'])

    def test_seventh_rejected_reply_does_not_allow_a_seventh_correction(self):
        invalid = action('finish', 1) | {'monitor': plan()}
        result, calls = self.run_ground([invalid] * 8, 10, repairs=6)
        budget = result['active_trace']['react_budget']
        self.assertEqual(len(calls), 7)
        self.assertEqual(budget['failed_checks'], 7)
        self.assertEqual(budget['repair_calls'], 6)
        self.assertIsNone(result['predicted_id'])

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
