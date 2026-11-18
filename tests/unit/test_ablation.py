"""Ablation conditions change tool access, guard behavior, and provenance."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding import react
from eviground.grounding.settings import evidence_policy, settings
from tests.helpers import parsed, response


class AblationTests(unittest.TestCase):
    def test_inspection_defaults_and_explicit_angles_share_one_observation(self):
        from copy import deepcopy

        objects = [{"bbox_id": 1, "target": "chair", "bbox_3d": [0, 0, 1, 1, 1, 1]}]
        inspect = {"action": "inspect", "reason": "compare appearance", "focus_ids": [1],
                   "context": "scene", "azimuth_deg": None, "elevation_deg": None,
                   "frame": None, "predicted_id": None, "evidence": [], "unresolved": []}
        finish = {**inspect, "action": "finish", "focus_ids": [], "context": None,
                  "predicted_id": None, "unresolved": ["appearance remains uncertain"]}
        initial = [{"image": "initial.png", "kind": "overview"}]
        rendered = {"image": "inspection.png", "marked_ids": [1]}
        env = {"EVIGROUND_DAG_MONITOR_ENABLED": "0", "EVIGROUND_DAG_WIRE_FORMAT": "snapshot",
               "EVIGROUND_DIALOGUE_HISTORY": "0", "EVIGROUND_PROMPT_PROFILE": "standard",
               "EVIGROUND_POLICY_ANNEALING_ENABLED": "0", "EVIGROUND_SUCCESSFUL_TOOL_BUDGET": "1",
               "EVIGROUND_DAG_REPAIR_LIMIT": "6", "EVIGROUND_INSPECT_ENABLED": "1",
               "EVIGROUND_MEASURE_ENABLED": "1", "EVIGROUND_CATEGORY_INSPECTION_REQUIRED": "0"}
        for explicit_first in (False, True):
            with self.subTest(explicit_first=explicit_first):
                default = deepcopy(inspect)
                explicit = {**inspect, "azimuth_deg": 315.0, "elevation_deg": 30.0}
                distinct = {**inspect, "azimuth_deg": 90, "elevation_deg": 30}
                actions = [explicit, default] if explicit_first else [default, explicit]
                original = deepcopy(actions)
                trace = {"steps": [], "internal_usage": {"render": 0, "compute": 0, "visual_check": 0}}
                with (tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env),
                      patch("eviground.rendering.overview.scene_views", return_value=initial),
                      patch("eviground.rendering.inspection.render_action", return_value=rendered) as render,
                      patch.object(react, "image_part", side_effect=lambda p: {"image": p}),
                      patch.object(react, "chat", side_effect=[response(a) for a in [*actions, distinct, finish]])):
                    react.ground("the chair", parsed(), objects, objects,
                                 np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1),
                                 Path(out), [], trace, 10)
                self.assertEqual(render.call_count, 2)
                self.assertEqual(trace["react_budget"]["successful_tool_calls"], 2)
                observations = trace["react_observations"]
                self.assertIn("already acquired", observations[1]["error"])
                self.assertTrue(observations[1]["no_evidence_acquired"])
                self.assertNotIn("error", observations[2])
                self.assertEqual(actions, original)

    def test_compact_action_contract_rejects_inconsistent_fields_and_empty_choice_evidence(self):
        import copy
        import jsonschema

        _, calls = self.run_finish(1, 1, 0)
        base = calls[0].kwargs["response_schema"]["schema"]
        schema = react.constrained_action_schema(base, ["inspect", "measure", "finish"])
        action = {"action": "inspect", "reason": "Compare appearance", "focus_ids": [1],
                  "context": "isolated", "azimuth_deg": None, "elevation_deg": None,
                  "frame": None, "predicted_id": None, "evidence": [], "unresolved": []}
        jsonschema.validate(action, schema)
        invalid = copy.deepcopy(action)
        invalid["predicted_id"] = 1
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
        action.update(action="finish", focus_ids=[], context=None, predicted_id=1)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(action, schema)
        action["evidence"] = [{"source_quote": "the chair", "object_ids": [1], "observation": "visible chair"}]
        jsonschema.validate(action, schema)
        action.update(predicted_id=None, evidence=[])
        jsonschema.validate(action, schema)
        finish_only = react.constrained_action_schema(base, ["finish"])
        action.update(action="inspect", focus_ids=[1], context="scene")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(action, finish_only)

    def test_abstention_evidence_contract_matches_scene_validator(self):
        import copy
        import jsonschema
        from eviground.rendering.overview import validate_decision

        _, calls = self.run_finish(1, 1, 0)
        base = calls[0].kwargs["response_schema"]["schema"]
        action = {"action": "finish", "reason": "Reference remains uncertain",
                  "focus_ids": [], "context": None, "azimuth_deg": None,
                  "elevation_deg": None, "frame": None, "predicted_id": None,
                  "evidence": [], "unresolved": ["reference identity"]}
        original = copy.deepcopy(base)
        for schema in (base, react.constrained_action_schema(base, ["finish"])):
            with self.subTest(constrained="anyOf" in schema):
                jsonschema.validate(action, schema)
                validate_decision(action, "the chair", [1], [1])
                evidence = {"source_quote": "the chair", "object_ids": [1],
                            "observation": "Chair visible; reference unresolved"}
                supported = {**action, "evidence": [evidence]}
                jsonschema.validate(supported, schema)
                validate_decision(supported, "the chair", [1], [1])
                for ids in ([], [99]):
                    invalid = {**action, "evidence": [{**evidence, "object_ids": ids}]}
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(invalid, schema)
                    with self.assertRaisesRegex(ValueError, "Scene evidence requires"):
                        validate_decision(invalid, "the chair", [1], [1])
        self.assertEqual(base, original)

    def test_evidence_id_arrays_are_bounded_by_known_object_count(self):
        import jsonschema

        _, calls = self.run_finish(1, 1, 0)
        base = calls[0].kwargs["response_schema"]["schema"]
        evidence = {"source_quote": "the chair", "object_ids": [1],
                    "observation": "Chair visible; reference unresolved"}
        action = {"action": "finish", "reason": "Reference remains uncertain",
                  "focus_ids": [], "context": None, "azimuth_deg": None,
                  "elevation_deg": None, "frame": None, "predicted_id": None,
                  "evidence": [evidence], "unresolved": ["reference identity"]}
        for schema in (base, react.constrained_action_schema(base, ["finish"])):
            jsonschema.validate(action, schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**action, "evidence": [{**evidence, "object_ids": [1] * 40}]}, schema)

    def run_finish(self, inspect_enabled, measure_enabled, guard, returned="finish"):
        query = "the chair"
        objects = [{"bbox_id": 1, "target": "stool", "bbox_3d": [0, 0, 1, 1, 1, 1],
                    "category_hypotheses": [
                        {"category": "stool", "rank": 1, "gap_from_top1": 0},
                        {"category": "chair", "rank": 2, "gap_from_top1": .2}]}]
        action = {"action": returned, "reason": "visible evidence supports chair",
                  "focus_ids": [], "context": None, "azimuth_deg": None,
                  "elevation_deg": None, "frame": None, "predicted_id": 1,
                  "evidence": [{"source_quote": query, "object_ids": [1],
                                "observation": "visible chair"}], "unresolved": []}
        trace = {"steps": [], "internal_usage": {"render": 0, "compute": 0, "visual_check": 0}}
        env = {"EVIGROUND_INSPECT_ENABLED": str(inspect_enabled),
               "EVIGROUND_MEASURE_ENABLED": str(measure_enabled),
               "EVIGROUND_CATEGORY_INSPECTION_REQUIRED": str(guard)}
        views = [{"image": "initial.png", "kind": "overview"},
                 {"image": "top.png", "kind": "top"}]
        with (tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env),
              patch("eviground.rendering.overview.scene_views", return_value=views),
              patch("eviground.rendering.inspection.render_action") as render,
              patch.object(react, "measure") as measure,
              patch.object(react, "image_part", side_effect=lambda p: {"image": p}),
              patch.object(react, "chat", return_value=response(action)) as chat):
            result = react.ground(query, parsed(), objects, objects,
                                  np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1),
                                  Path(out), [], trace, 5)
            render.assert_not_called()
            measure.assert_not_called()
            return result, chat.call_args_list

    def test_each_disabled_condition_finishes_without_guard_or_tool(self):
        for inspect_enabled, measure_enabled, guard in ((0, 1, 1), (1, 0, 0),
                                                       (1, 1, 0), (0, 0, 1)):
            with self.subTest(inspect=inspect_enabled, measure=measure_enabled):
                result, calls = self.run_finish(inspect_enabled, measure_enabled, guard)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result["predicted_id"], 1)
                available = [n for n, enabled in (("inspect", inspect_enabled),
                             ("measure", measure_enabled)) if enabled] + ["finish"]
                schema = calls[0].kwargs["response_schema"]["schema"]
                self.assertEqual(schema["properties"]["focus_ids"]["maxItems"], 8)
                self.assertEqual(schema["properties"]["action"]["enum"], available)
                self.assertEqual(result["active_trace"]["react_available_actions"], available)
                prompt = calls[0].args[0][0]["content"]
                self.assertNotIn("A finish without that inspection", prompt)
                for tool, enabled in (("inspect", inspect_enabled), ("measure", measure_enabled)):
                    self.assertEqual(f"- {tool}:" in prompt, bool(enabled))

    def test_disabled_action_cannot_execute_even_if_response_violates_schema(self):
        with self.assertRaisesRegex(ValueError, "unavailable action"):
            self.run_finish(0, 1, 0, returned="inspect")

    def test_full_prompt_is_unchanged_and_switches_enter_identity(self):
        env = {k: "1" for k in ("EVIGROUND_INSPECT_ENABLED", "EVIGROUND_MEASURE_ENABLED",
                               "EVIGROUND_CATEGORY_INSPECTION_REQUIRED")}
        with patch.dict(os.environ, env):
            self.assertEqual(react.planner_instructions(evidence_policy()), react.INSTRUCTIONS)
            full = settings()
        with patch.dict(os.environ, {**env, "EVIGROUND_MEASURE_ENABLED": "0"}):
            self.assertNotEqual(settings(), full)
        with patch.dict(os.environ, {**env, "EVIGROUND_INSPECT_ENABLED": "false"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                evidence_policy()
