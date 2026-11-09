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
