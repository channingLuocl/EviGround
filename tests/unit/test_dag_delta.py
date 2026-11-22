"""Delta serialization must preserve the full DAG's evidence and rollback rules."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError, validate
import numpy as np

from eviground.grounding import react
from eviground.grounding.dag_monitor import DagMonitor, NODE_STATE_FIELDS
from eviground.grounding.evidence_followup import EvidenceFollowup
from eviground.grounding.settings import dag_wire_format, settings
from tests.helpers import parsed, response
from tests.unit.test_dag_monitor import QUERY, action, plan
from tests.unit.test_prompt_tables import expand


def compact_ids(p):
    p = deepcopy(p)
    ids = {"group": "n0", "compare": "n1", "verify": "n11"}
    for n in p["nodes"]:
        n["id"] = ids[n["id"]]
        n["depends_on"] = [ids[i] for i in n["depends_on"]]
    p["current_node_id"] = ids[p["current_node_id"]]
    p["reopen_node_ids"] = [ids[i] for i in p["reopen_node_ids"]]
    return p


def delta(p, changed):
    return {"node_updates": [{k: deepcopy(n[k]) for k in NODE_STATE_FIELDS}
                             for n in sorted(p["nodes"], key=lambda n: n["id"] != "n11")
                             if n["id"] in changed],
            **{k: deepcopy(p[k]) for k in ("current_node_id", "reopen_node_ids", "revision_reason")}}


class DagDeltaTests(unittest.TestCase):
    def setUp(self):
        self.monitor = self.new_monitor()
        self.start = compact_ids(plan())
        self.start['nodes'][-1]['depends_on'] = ['n0', 'n1']
        self.monitor.update(self.start, action(), [])

    def new_monitor(self):
        return DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2}, ["inspect", "measure", "finish"])

    def finished(self, evidence="observation_1"):
        p = deepcopy(self.start)
        for n in p["nodes"][1:]:
            n.update(status="completed", evidence_ids=[evidence], conclusion="comparison supports chair 1")
        p["nodes"][1]["excluded_target_ids"] = [2]
        p["current_node_id"] = "n11"
        return p

    def observations(self, **overrides):
        return [{"tool": "measure", "evidence_id": "observation_1", "focus_ids": [1, 2, 3], **overrides}]

    def test_delta_is_exact_full_snapshot_without_mutation(self):
        p = self.finished()
        proposal = delta(p, {"n1", "n11"})
        before, accepted = deepcopy(proposal), deepcopy(self.monitor.plan)
        schema = self.monitor.delta_schema({"initial", "observation_1"})
        Draft202012Validator.check_schema(schema)
        validate(proposal, schema)
        decoded = self.monitor.decode(proposal)
        self.assertEqual(decoded, p)
        self.assertEqual(proposal, before)
        self.assertEqual(self.monitor.plan, accepted)
        reference = self.new_monitor()
        reference.update(self.start, action(), [])
        expected = reference.update(p, action("finish", 1), self.observations())
        actual = self.monitor.update(decoded, action("finish", 1), self.observations())
        self.assertEqual(actual, expected)

    def test_empty_updates_preserve_pending_and_cannot_finish(self):
        p = delta(self.start, set())
        self.assertEqual(self.monitor.decode(p), self.start)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.monitor.update(self.monitor.decode(p), action("finish", 1), [])

    def test_delta_requires_prior_plan(self):
        with self.assertRaisesRegex(ValueError, "accepted task DAG"):
            self.new_monitor().decode(delta(self.start, set()))

    def compiled_schema(self):
        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        return calls[1].kwargs["response_schema"]["schema"]

    def test_action_compilation_rejects_captured_terminal_failures(self):
        schema = self.compiled_schema()
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        valid = action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}
        validate(valid, schema)
        for change in ("missing_verification", "pending_verification", "wrong_stage",
                       "tool_at_verification", "unresolved", "future_evidence"):
            value = deepcopy(valid)
            m = value["monitor"]
            if change == "missing_verification":
                m["node_updates"].pop(0)
            elif change == "pending_verification":
                m["node_updates"][0].update(status="pending", evidence_ids=[], conclusion="")
            elif change == "wrong_stage":
                m["current_node_id"] = "n1"
            elif change == "tool_at_verification":
                value.update(action="inspect", context="scene", focus_ids=[1, 2],
                             predicted_id=None, evidence=[])
            elif change == "unresolved":
                value["unresolved"] = ["a condition remains"]
            else:
                m["node_updates"][0]["evidence_ids"] = ["observation_99"]
            with self.subTest(change=change):
                self.assertFalse(validator.is_valid(value))

    def test_compact_action_bounds_repeated_unresolved_and_evidence(self):
        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        for index, call in enumerate(calls):
            schema = call.kwargs["response_schema"]["schema"]
            state = json.loads(call.args[0][1]["content"][0]["text"])
            limit = len(state["monitor_requirements"])
            proposal = action() | {"monitor": self.monitor.encode(self.start)
                                   if index == 0 else delta(self.start, {"n1"})}
            proposal["unresolved"] = [QUERY] * (limit + 1)
            self.assertFalse(Draft202012Validator(schema).is_valid(proposal))
            proposal["unresolved"] = [QUERY]
            validate(proposal, schema)
        finish = action("finish", 1) | {
            "monitor": delta(self.finished(), {"n1", "n11"})}
        finish["evidence"] *= 5
        self.assertFalse(Draft202012Validator(schema).is_valid(finish))
        finish["evidence"] = finish["evidence"][:4]
        validate(finish, schema)

    def test_delta_selects_action_branch_before_emitting_state_updates(self):
        # Otherwise starting with n0 prunes the non-null finish branch, whose
        # explicit verification checkpoint must start with n11.
        for branch in self.compiled_schema()["anyOf"]:
            self.assertEqual(list(branch["properties"])[:4],
                             ["reason", "action", "predicted_id", "monitor"])
            self.assertEqual(branch["required"][:3], ["reason", "action", "predicted_id"])

    def test_replanning_masks_self_dependencies_without_imposing_id_order(self):
        schema = self.compiled_schema()
        validator = Draft202012Validator(schema)
        proposal = action('inspect') | {'context': 'scene',
                    'monitor': {'replace_plan': self.monitor.encode(self.start)}}
        validator.validate(proposal)
        for index in range(len(proposal['monitor']['replace_plan']['nodes'])):
            invalid = deepcopy(proposal)
            task = invalid['monitor']['replace_plan']['nodes'][index]
            task['depends_on'] = [task['id']]
            self.assertFalse(validator.is_valid(invalid), task['id'])
        # Replanning is allowed to change dependency order, rather than being
        # silently restricted to the initial topological numbering convention.
        reverse = deepcopy(self.start)
        reverse['nodes'][0]['depends_on'] = ['n1']
        reverse['nodes'][1]['depends_on'] = []
        reverse['nodes'][1].update(status='completed', evidence_ids=['initial'],
                                   conclusion='initial geometry supports this comparison')
        reverse['current_node_id'] = 'n0'
        proposal['monitor']['replace_plan'] = self.monitor.encode(reverse)
        validator.validate(proposal)
        # A full valid replacement can still be accepted by the original Monitor.
        reverse['nodes'][0].update(status='pending', evidence_ids=[], conclusion='')
        reverse['reopen_node_ids'] = ['n0']
        reverse['current_node_id'] = 'n1'
        proposal['monitor']['replace_plan'] = self.monitor.encode(reverse)
        validator.validate(proposal)
        self.monitor.update(self.monitor.decode(proposal['monitor']), proposal,
                            self.observations())
        self.assertEqual(self.monitor.plan['current_node_id'], 'n1')

    def test_evidence_bound_preserves_every_distinct_known_id(self):
        schema = self.compiled_schema()
        value = action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}
        value["evidence"][0]["object_ids"] = [1, 2, 3]
        validate(value, schema)
        value["evidence"][0]["object_ids"].append(1)
        self.assertFalse(Draft202012Validator(schema).is_valid(value))
        for branch in schema["anyOf"]:
            ids = branch["properties"]["evidence"]["items"]["properties"]["object_ids"]
            self.assertEqual(ids["maxItems"], len(ids["items"]["enum"]))

    def test_measurement_reference_instructions_are_absent_without_references(self):
        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        for index, call in enumerate(calls):
            self.assertNotIn(react.OBSERVATION_REFERENCE_INSTRUCTIONS, call.args[0][0]["content"])
            if index == 0:
                self.assertNotIn(react.HISTORY_REFERENCE_INSTRUCTIONS, call.args[0][0]["content"])

    def test_history_references_expand_to_every_original_plan_and_keep_current_dag(self):
        first = self.monitor.encode(self.start)
        changed = self.monitor.encode(self.finished())
        state = {"observations": [], "task_dag": changed,
                 "previous_actions": [{"monitor": p, "reason": str(i)}
                                      for i, p in enumerate([first, first, changed, first])]}
        original = deepcopy(state)
        compact = react.compact_state(state, self.monitor)
        self.assertEqual(compact["previous_actions"][1]["monitor"]["nodes"], {"same_as_action": 0})
        self.assertEqual(compact["previous_actions"][3]["monitor"]["nodes"], {"same_as_action": 0})
        self.assertEqual(compact["previous_actions"][2]["monitor"]["nodes"],
                         {"same_as_task_dag": "nodes"})
        expanded = []
        for action_value in compact["previous_actions"]:
            restored = deepcopy(action_value)
            for key, value in restored["monitor"].items():
                if isinstance(value, dict) and set(value) == {"same_as_action"}:
                    restored["monitor"][key] = deepcopy(expanded[value["same_as_action"]]["monitor"][key])
                elif isinstance(value, dict) and set(value) == {"same_as_task_dag"}:
                    restored["monitor"][key] = deepcopy(compact["task_dag"][value["same_as_task_dag"]])
            expanded.append(restored)
        self.assertEqual(expanded, original["previous_actions"])
        self.assertEqual(compact["task_dag"], original["task_dag"])
        self.assertEqual(state, original)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(state)))

    def test_repeated_history_supplies_reference_instructions_after_acquisition(self):
        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action() | {"focus_ids": [1, 3], "monitor": delta(self.start, set())},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        self.assertNotIn(react.HISTORY_REFERENCE_INSTRUCTIONS, calls[0].args[0][0]["content"])
        self.assertIn(react.HISTORY_REFERENCE_INSTRUCTIONS, calls[2].args[0][0]["content"])

    def test_inspection_camera_schema_matches_renderer_bounds(self):
        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        for index, monitor in enumerate([self.monitor.encode(self.start), delta(self.start, {"n1"})]):
            validator = Draft202012Validator(calls[index].kwargs["response_schema"]["schema"])
            proposal = action("inspect") | {"context": "scene", "monitor": monitor}
            for azimuth, elevation, expected in [
                    (None, None, True), (0, 10, True), (359.999, 75, True),
                    (-1, 30, False), (360, 30, False), (0, 0, False),
                    (0, 9.999, False), (0, 75.001, False)]:
                with self.subTest(turn=index, azimuth=azimuth, elevation=elevation):
                    proposal.update(azimuth_deg=azimuth, elevation_deg=elevation)
                    self.assertEqual(validator.is_valid(proposal), expected)

    def test_finish_delta_requires_every_pending_task_update(self):
        schema = self.compiled_schema()
        valid = action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}
        validate(valid, schema)
        missing = deepcopy(valid)
        missing["monitor"]["node_updates"] = [missing["monitor"]["node_updates"][0]]
        # Captured keyed/brief output completed verification but left its comparison
        # pending. Reject the omission during decoding, not after another API call.
        self.assertFalse(Draft202012Validator(schema).is_valid(missing))
        self.assertEqual(valid["monitor"]["node_updates"][1]["id"], "n1")

    def test_finish_delta_can_omit_already_completed_tasks(self):
        full = self.finished()
        self.monitor.update(full, action("finish", 1), self.observations())
        proposal = delta(full, {"n11"})
        before = deepcopy(proposal)
        shared = {}
        schema = react.constrained_delta_monitor_schema(
            self.monitor.delta_schema({"initial", "observation_1"}),
            "finish", shared, selected=True, pending_node_ids=[])
        validate(proposal, {**schema, "$defs": shared})
        self.monitor.update(self.monitor.decode(proposal), action("finish", 1), self.observations())
        self.assertEqual(proposal, before)

    def test_tool_delta_cannot_complete_its_current_stage(self):
        schema = self.compiled_schema()
        valid = action() | {"monitor": delta(self.start, {"n1"})}
        validate(valid, schema)
        decoded = self.monitor.decode(valid["monitor"])
        self.monitor.update(decoded, valid, [])
        invalid = deepcopy(valid)
        invalid["monitor"]["node_updates"][0].update(
            status="completed", evidence_ids=["observation_1"],
            conclusion="The measured height selects chair 1.", excluded_target_ids=[2])
        # Captured online failure: a supported comparison still cannot request
        # another tool from the same node after declaring that node completed.
        with self.assertRaises(ValidationError):
            validate(invalid, schema)

    def test_delta_and_acquired_action_mask_preserve_supported_finish(self):
        from eviground.api.client import request_options
        from eviground.grounding.react import evidence_action_key

        with patch.dict(os.environ, {"EVIGROUND_ACTION_NO_REPEAT": "1",
                                    "EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            schema = self.compiled_schema()
            options = request_options(4000, None, True,
                dict(name="react_action", strict=True, schema=schema))
        acquired = schema["x-eviground-acquired-actions"]
        self.assertEqual(acquired, [evidence_action_key(action())])
        valid = action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}
        validate(valid, schema)
        self.assertEqual(self.monitor.update(self.monitor.decode(valid["monitor"]), valid,
                                             self.observations())["active_target_ids"], [1])
        invalid = deepcopy(valid)
        invalid["monitor"]["current_node_id"] = "n1"
        self.assertFalse(Draft202012Validator(schema).is_valid(invalid))
        # Changing DAG state cannot make a previously acquired tool observation new.
        repeated = action() | {"monitor": delta(self.start, {"n1"})}
        self.assertFalse(Draft202012Validator(schema).is_valid(repeated))
        self.assertIn("guided_grammar", options)
        self.assertNotIn("response_format", options)

    def test_tool_delta_cannot_complete_verification_with_pending_current_stage(self):
        schema = self.compiled_schema()
        invalid = action() | {"monitor": delta(self.start, {"n1"})}
        verification = delta(self.finished(), {"n11"})["node_updates"][0]
        invalid["monitor"]["node_updates"].append(verification)
        with self.assertRaises(ValidationError):
            validate(invalid, schema)

    def test_action_compilation_preserves_abstention_replacement_and_rollback(self):
        schema = self.compiled_schema()
        abstention = action("finish") | {"monitor": delta(self.start, set())}
        abstention["monitor"]["current_node_id"] = "n11"
        validate(abstention, schema)
        for missing in ([], [""], ["  "]):
            unexplained = {**abstention, "unresolved": missing}
            self.assertFalse(Draft202012Validator(schema).is_valid(unexplained))
        self.monitor.update(self.monitor.decode(abstention["monitor"]), abstention, [])
        reopened = deepcopy(self.start)
        reopened["nodes"][0].update(status="pending", evidence_ids=[], conclusion="")
        reopened["current_node_id"] = "n0"
        rollback = action() | {"monitor": delta(reopened, {"n0"})}
        rollback["monitor"]["reopen_node_ids"] = ["n0"]
        validate(rollback, schema)
        replacement = action("finish", 1) | {"monitor": {
            "replace_plan": self.monitor.encode(self.finished())}}
        validate(replacement, schema)
        replacement["monitor"]["replace_plan"]["verification"].update(
            status="pending", evidence_ids=[], conclusion="")
        with self.assertRaises(ValidationError):
            validate(replacement, schema)

    def test_compilation_does_not_fill_exclusions_or_bypass_full_monitor(self):
        schema = self.compiled_schema()
        unsupported = self.finished()
        unsupported["nodes"][1]["excluded_target_ids"] = []
        proposal = action("finish", 1) | {"monitor": delta(unsupported, {"n1", "n11"})}
        before = deepcopy(proposal)
        validate(proposal, schema)
        self.assertEqual(proposal, before)
        with self.assertRaisesRegex(ValueError, "one active target"):
            self.monitor.update(self.monitor.decode(proposal["monitor"]), proposal,
                                self.observations())

    def test_invalid_delta_fields_are_rejected(self):
        valid = delta(self.finished(), {"n1", "n11"})
        for change in ("unknown", "duplicate", "missing", "immutable", "pending_evidence", "empty_conclusion"):
            p = deepcopy(valid)
            n = p["node_updates"][0]
            if change == "unknown":
                n["id"] = "n10"
            elif change == "duplicate":
                p["node_updates"][1] = deepcopy(n)
            elif change == "missing":
                del n["bound_object_ids"]
            elif change == "immutable":
                n["goal"] = "silently replace a task"
            elif change == "pending_evidence":
                n["status"] = "pending"
            else:
                n["conclusion"] = ""
            with self.subTest(change=change), self.assertRaises((ValidationError, ValueError)):
                self.monitor.decode(p)

    def test_evidence_checks_remain_in_schema_and_monitor(self):
        future = delta(self.finished("observation_99"), {"n1", "n11"})
        with self.assertRaises(ValidationError):
            validate(future, self.monitor.delta_schema({"initial", "observation_1"}))
        for observations in ([], self.observations(error="failed"),
                             self.observations(no_evidence_acquired=True)):
            p = delta(self.finished(), {"n1", "n11"})
            with self.subTest(observations=observations), self.assertRaisesRegex(ValueError, "failed evidence"):
                self.monitor.update(self.monitor.decode(p), action("finish", 1), observations)

    def test_replacement_can_add_remove_and_reassign_tasks(self):
        p = deepcopy(self.start)
        added = deepcopy(p["nodes"][1])
        added.update(id="n2", goal="independent check", depends_on=["n1"])
        p["nodes"].insert(-1, added)
        p["nodes"][-1]["depends_on"] = ["n2"]
        # Requirement assignment remains a planner decision, including q0 at verification.
        p["nodes"][0]["condition_ids"] = []
        p["nodes"][1]["condition_ids"] = []
        p["nodes"][2]["condition_ids"] = []
        p["current_node_id"] = "n0"
        wire = {"replace_plan": self.monitor.encode(p)}
        self.assertEqual(self.monitor.decode(wire), p)
        self.monitor.update(self.monitor.decode(wire), action(), [])
        p["nodes"] = [n for n in p["nodes"] if n["id"] != "n2"]
        p["nodes"][-1]["depends_on"] = ["n1"]
        p["current_node_id"] = "n1"
        self.assertEqual(self.monitor.decode({"replace_plan": self.monitor.encode(p)}), p)
        event = self.monitor.update(self.monitor.decode({"replace_plan": self.monitor.encode(p)}), action(), [])
        self.assertEqual({n["id"] for n in event["plan"]["nodes"]}, {"n0", "n1", "n11"})

    def test_rollback_resets_descendants_and_restores_candidates_identically(self):
        full = self.finished()
        self.monitor.update(full, action("finish", 1), self.observations())
        reference = self.new_monitor()
        reference.update(self.start, action(), [])
        reference.update(full, action("finish", 1), self.observations())
        reopened = deepcopy(self.monitor.plan)
        reopened["nodes"][0].update(status="pending", evidence_ids=[], conclusion="", excluded_target_ids=[])
        reopened.update(current_node_id="n0", reopen_node_ids=["n0"], revision_reason="recheck the binding")
        expected = reference.update(reopened, action(), self.observations())
        actual = self.monitor.update(self.monitor.decode(delta(reopened, {"n0"})), action(), self.observations())
        self.assertEqual(actual, expected)
        self.assertEqual(actual["reactivated_target_ids"], [2])
        self.assertEqual(actual["invalidated_node_ids"], ["n0", "n1", "n11"])
        self.assertTrue(all(n["status"] == "pending" and not n["excluded_target_ids"]
                            and not n["evidence_ids"] and not n["conclusion"] for n in self.monitor.plan["nodes"]))

    def run_ground(self, answers, category=False, repairs=0, expanded_target_pool=False):
        objects = [{"bbox_id": i, "target": "chair", "bbox_3d": [i, 0, .5, 1, 1, 1],
                    "category_hypotheses": [{"category": "chair", "rank": 1, "gap_from_top1": 0}]}
                   for i in (1, 2, 3)]
        if category:
            objects[0]["category_hypotheses"] = [{"category": "stool", "rank": 1, "gap_from_top1": 0},
                                                 {"category": "chair", "rank": 2, "gap_from_top1": .2}]
        trace = {"steps": [], "internal_usage": {"render": 0, "compute": 0, "visual_check": 0}}
        env = {"EVIGROUND_DAG_MONITOR_ENABLED": "1", "EVIGROUND_DAG_WIRE_FORMAT": "delta",
               "EVIGROUND_PROMPT_PROFILE": "compact", "EVIGROUND_DIALOGUE_HISTORY": "0",
               "EVIGROUND_POLICY_ANNEALING_ENABLED": "1", "EVIGROUND_DAG_REPAIR_LIMIT": str(repairs),
               "EVIGROUND_CATEGORY_INSPECTION_REQUIRED": "1"}
        with (tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env),
              patch("eviground.rendering.overview.scene_views", return_value=[{"image": "a.png"}, {"image": "b.png"}]),
              patch.object(react, "image_part", side_effect=lambda p: {"image": p}),
              patch.object(react, "chat", side_effect=[a if isinstance(a, Exception) else response(a)
                                                       for a in answers]) as chat):
            result = react.ground(QUERY, parsed(), objects, objects[:2], np.zeros((3, 3)),
                                  np.zeros((3, 3)), np.arange(3), Path(out), [], trace, 10,
                                  expanded_target_pool=expanded_target_pool)
        return result, chat.call_args_list

    def test_rejected_finish_runs_joint_measurement_without_accepting_the_plan(self):
        def require_measurement(plan, observations, action):
            if not any(row["tool"] == "measure" for row in observations):
                raise EvidenceFollowup("Measure alternatives together", focus_ids=[1, 2, 3])

        premature = action("finish", 1) | {"monitor": self.monitor.encode(self.finished("initial"))}
        verified = action("finish", 1) | {"monitor": self.monitor.encode(self.finished("observation_1"))}
        with patch.object(DagMonitor, "check_exclusion_evidence", side_effect=require_measurement):
            result, calls = self.run_ground([premature, verified], repairs=1)
        self.assertEqual(result["predicted_id"], 1)
        feedback = json.loads(calls[1].args[0][1]["content"][-1]["text"])
        self.assertEqual(feedback["active_target_ids"], [1, 2])
        measured = expand(feedback["observations"])[-1]
        self.assertEqual(measured["tool"], "measure")
        self.assertEqual(measured["evidence_followup"]["focus_ids"], [1, 2, 3])

    def test_tentative_conflict_gets_one_joint_scene_view_before_another_answer(self):
        gap = EvidenceFollowup("Conflicting views of both candidates", focus_ids=[1, 2, 3], tool="inspect")
        verified = action("finish", 1) | {"monitor": self.monitor.encode(self.finished("observation_1"))}
        view = {"image": "joint.png", "marked_ids": [1, 2, 3], "context": "scene"}
        with patch("eviground.rendering.inspection.render_action", return_value=view) as render:
            result, calls = self.run_ground([gap, verified], repairs=1)
        self.assertEqual(result["predicted_id"], 1)
        render.assert_called_once()
        self.assertEqual(render.call_args.args[0]["focus_ids"], [1, 2, 3])
        self.assertIn({"image": "joint.png"}, calls[1].args[0][1]["content"])

    def test_rejected_delta_survives_feedback_and_can_be_repaired(self):
        duplicate = delta(self.start, {"n1"})
        duplicate["node_updates"].append(deepcopy(duplicate["node_updates"][0]))
        replacement = {"replace_plan": self.monitor.encode(self.start)}
        replacement["replace_plan"]["requirement_nodes"]["q0"] = ["n10"]
        for invalid, error in ((duplicate, "Duplicate node IDs"),
                               (replacement, "absent DAG node")):
            with self.subTest(error=error):
                before = deepcopy(invalid)
                answers = [action() | {"monitor": self.monitor.encode(self.start)},
                           action() | {"monitor": invalid},
                           action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}]
                result, calls = self.run_ground(answers, repairs=1)
                self.assertEqual(result["predicted_id"], 1)
                state = json.loads(calls[2].args[0][1]["content"][0]["text"])
                feedback = json.loads(calls[2].args[0][1]["content"][-1]["text"])
                self.assertEqual(expand(state["previous_actions"])[-1]["monitor"], before)
                self.assertEqual(expand(feedback["task_dag"]), self.monitor.encode(self.start))
                self.assertEqual(feedback["remaining_tool_calls"], 9)
                self.assertEqual(feedback["remaining_plan_repairs"], 0)
                self.assertIn(error, feedback["observations"][-1]["error"])

    def test_rejected_focus_group_is_masked_on_the_next_controller_request(self):
        invalid = action('measure') | {'focus_ids': [1, 2, 1, 2],
                                      'monitor': delta(self.start, {'n1'})}
        answers = [action() | {'monitor': self.monitor.encode(self.start)}, invalid,
                   action('finish', 1) | {'monitor': delta(self.finished(), {'n1', 'n11'})}]
        with patch.dict(os.environ, {'EVIGROUND_ACTION_NO_REPEAT': '1',
                                     'EVIGROUND_GUIDED_DECODING_BACKEND': 'guidance'}):
            result, calls = self.run_ground(answers, repairs=1)
        before = calls[1].kwargs['response_schema']['schema']
        after = calls[2].kwargs['response_schema']['schema']
        self.assertTrue(Draft202012Validator(before).is_valid(invalid))
        self.assertFalse(Draft202012Validator(after).is_valid(invalid))
        self.assertEqual(after['x-eviground-invalid-focus-groups'], [[1, 1, 2, 2]])
        self.assertEqual(invalid['focus_ids'], [1, 2, 1, 2])
        self.assertEqual(result['predicted_id'], 1)
        trace = result['active_trace']
        self.assertEqual(trace['react_budget']['successful_tool_calls'], 1)
        self.assertEqual(trace['react_budget']['failed_checks'], 1)
        self.assertEqual(trace['react_budget']['decision_calls'], 3)
        self.assertIn('distinct known focus_ids', trace['steps'][1]['observation']['error'])

    def test_production_path_first_full_then_delta_keeps_full_internal_state(self):
        result, calls = self.run_ground([action() | {"monitor": self.monitor.encode(self.start)},
                                        action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        self.assertEqual(result["predicted_id"], 1)
        trace = result["active_trace"]
        self.assertEqual(trace["dag_wire_format"], "delta")
        self.assertEqual(trace["react_tool_budget"], 10)
        self.assertTrue(trace["policy_annealing_enabled"])
        self.assertEqual(len(trace["policy_annealing_history"]), 2)
        self.assertEqual(trace["dag_monitor_budget"]["successful_tool_calls"], 1)
        self.assertEqual(trace["react_actions"][-1]["monitor"], self.finished())
        first = calls[0].kwargs["response_schema"]["schema"]["$defs"]["monitor"]
        second = calls[1].kwargs["response_schema"]["schema"]["$defs"]["monitor_tool"]
        self.assertTrue(all("verification" in branch["required"] for branch in first["anyOf"]))
        self.assertIn("node_updates", second["anyOf"][0]["required"])
        feedback = json.loads(calls[1].args[0][1]["content"][-1]["text"])
        self.assertEqual(feedback["remaining_tool_calls"], 9)
        self.assertEqual(expand(feedback["task_dag"]), self.monitor.encode(self.start))
        validate(action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})},
                 calls[1].kwargs["response_schema"]["schema"])

    def test_delta_turn_does_not_request_a_full_pending_snapshot(self):
        from eviground.grounding.dag_monitor import COMPACT_INSTRUCTIONS

        _, calls = self.run_ground([
            action() | {"monitor": self.monitor.encode(self.start)},
            action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}])
        first, later = [call.args[0][0]["content"] for call in calls]
        self.assertIn(COMPACT_INSTRUCTIONS, first)
        self.assertNotIn("Return monitor as a complete task DAG on every action", later)
        self.assertNotIn("Wire-shape example ONLY", later)
        self.assertNotIn("complete-DAG output\nshape above", later)
        # Only obsolete output-shape text is removed, not the planning rules.
        for paragraph in COMPACT_INSTRUCTIONS.split("\n\n")[1:6]:
            self.assertIn(paragraph, later)
        self.assertIn("update for EVERY task that is still pending", later)
        self.assertIn('return monitor={"replace_plan":', later)
        self.assertIn("Round-aware policy constraint", later)

    def test_delta_cannot_bypass_category_inspection(self):
        answers = [action() | {"monitor": self.monitor.encode(self.start)},
                   action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}]
        result, _ = self.run_ground(answers, category=True)
        self.assertIsNone(result["predicted_id"])
        self.assertIn("alternative-only", result["scene_evidence"]["unresolved"][0])

    def test_expanded_target_pool_requires_focused_category_inspection(self):
        answers = [action() | {"monitor": self.monitor.encode(self.start)},
                   action("finish", 1) | {"monitor": delta(self.finished(), {"n1", "n11"})}]
        result, _ = self.run_ground(answers, expanded_target_pool=True)
        self.assertIsNone(result["predicted_id"])
        self.assertIn("no classifier category match", result["scene_evidence"]["unresolved"][0])

    def test_opt_in_is_recorded_and_invalid_configuration_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(dag_wire_format(), "snapshot")
            self.assertEqual(settings()["dag_wire_format"], "snapshot")
        with patch.dict(os.environ, {"EVIGROUND_DAG_WIRE_FORMAT": "delta"}):
            self.assertEqual(settings()["dag_wire_format"], "delta")
        with patch.dict(os.environ, {"EVIGROUND_DAG_WIRE_FORMAT": "wrong"}):
            with self.assertRaises(ValueError):
                settings()

    def test_delta_rejects_standard_prompts_and_disabled_monitor(self):
        for profile, enabled in (("standard", "1"), ("compact", "0")):
            with (patch.dict(os.environ, {"EVIGROUND_DAG_WIRE_FORMAT": "delta",
                                          "EVIGROUND_PROMPT_PROFILE": profile,
                                          "EVIGROUND_DAG_MONITOR_ENABLED": enabled}),
                  self.assertRaisesRegex(ValueError, "requires compact")):
                react.ground(QUERY, parsed(), [{"bbox_id": 1}], [{"bbox_id": 1}],
                             None, None, None, Path("unused"), [], {}, 10)


if __name__ == "__main__":
    unittest.main()
