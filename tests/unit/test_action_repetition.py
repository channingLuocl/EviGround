"""Repeated observations cannot be disguised by order or unused parameters."""
from copy import deepcopy
from itertools import combinations, permutations, product
import json
import os
import re
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from eviground.api.response_schemas import array, enum, nullable, obj
from eviground.grounding.action_repetition import exclude_acquired_actions, no_repeat_grammar
from eviground.grounding.react import evidence_action_key


class EvidenceActionKeyTests(unittest.TestCase):
    def test_inspection_normalizes_defaults_order_and_non_tool_fields(self):
        original = dict(action="inspect", focus_ids=[2, 1], context="isolated",
                        azimuth_deg=None, elevation_deg=None, reason="first", monitor={})
        repeat = {**original, "focus_ids": [1, 2], "azimuth_deg": 315.0,
                  "elevation_deg": 30, "reason": "different", "monitor": {"revised": True}}
        untouched = deepcopy(original)
        self.assertEqual(evidence_action_key(original), evidence_action_key(repeat))
        self.assertEqual(original, untouched)

    def test_distinct_camera_context_or_group_remains_available(self):
        original = dict(action="inspect", focus_ids=[1, 2], context="isolated",
                        azimuth_deg=None, elevation_deg=None)
        for change in ({"azimuth_deg": 316}, {"elevation_deg": 31},
                       {"context": "scene"}, {"focus_ids": [1, 3]}):
            with self.subTest(change=change):
                self.assertNotEqual(evidence_action_key(original), evidence_action_key({**original, **change}))

    def test_float_tail_is_the_same_camera(self):
        original = dict(action="inspect", focus_ids=[1, 2], context="isolated",
                        azimuth_deg=None, elevation_deg=None)
        self.assertEqual(evidence_action_key(original), evidence_action_key(
            {**original, "elevation_deg": 30.000000000000004}))

    def test_measurement_normalizes_groups_and_ignores_camera_fields(self):
        original = dict(action="measure", focus_ids=[3, 1, 2],
                        frame={"origin_ids": [2, 1], "facing_ids": [4, 3]})
        repeat = {**original, "focus_ids": [1, 2, 3], "context": "scene", "azimuth_deg": 60,
                  "frame": {"origin_ids": [1, 2], "facing_ids": [3, 4]}}
        self.assertEqual(evidence_action_key(original), evidence_action_key(repeat))

    def test_different_observer_frame_and_frameless_measurement_are_distinct(self):
        original = dict(action="measure", focus_ids=[1, 2], frame=None)
        framed = {**original, "frame": {"origin_ids": [], "facing_ids": [3]}}
        self.assertNotEqual(evidence_action_key(original), evidence_action_key(framed))
        self.assertNotEqual(evidence_action_key(framed), evidence_action_key(
            {**framed, "frame": {"origin_ids": [3], "facing_ids": [4]}}))


class ActionExclusionSchemaTests(unittest.TestCase):
    def setUp(self):
        self.known = [1, 2, 3, 4]
        self.schema = obj({
            "action": enum("inspect", "measure", "finish"),
            "focus_ids": {**array({"type": "integer", "enum": self.known}), "maxItems": 4},
            "context": nullable(enum("scene", "isolated")),
            "azimuth_deg": nullable({"type": "number", "minimum": 0, "exclusiveMaximum": 360}),
            "elevation_deg": nullable({"type": "number", "minimum": 10, "maximum": 75}),
            "frame": nullable(obj({"origin_ids": array({"type": "integer", "enum": self.known}),
                                   "facing_ids": array({"type": "integer", "enum": self.known})})),
            "reason": {"type": "string"},
            "monitor": obj({"revision": {"type": "integer"}}),
        })
        self.inspect = dict(action="inspect", focus_ids=[1, 2], context="isolated",
                            azimuth_deg=None, elevation_deg=None, frame=None,
                            reason="compare", monitor={"revision": 1})
        self.measure = {**self.inspect, "action": "measure", "context": None}

    def validator(self, *acquired):
        schema = exclude_acquired_actions(self.schema, [evidence_action_key(a) for a in acquired],
                                          self.known, ["inspect", "measure", "finish"], max_focus=4)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    def test_all_id_orders_and_camera_aliases_are_banned(self):
        validator = self.validator(self.inspect)
        for ids in permutations([1, 2]):
            for azimuth in (None, 315, 315.0):
                for elevation in (None, 30, 30.0):
                    repeat = {**self.inspect, "focus_ids": list(ids), "azimuth_deg": azimuth,
                              "elevation_deg": elevation, "reason": "changed",
                              "monitor": {"revision": 2}}
                    self.assertFalse(validator.is_valid(repeat), repeat)
        self.assertFalse(validator.is_valid({**self.inspect, "elevation_deg": 30.000000000000004}))
        self.assertFalse(validator.is_valid({**self.inspect, "elevation_deg": 29.999999999999996}))

    def test_grammar_masks_float_tails_without_blocking_a_new_pose(self):
        branch = deepcopy(self.schema)
        branch["properties"]["action"] = enum("inspect")
        schema = exclude_acquired_actions({"anyOf": [branch]},
            [evidence_action_key(self.inspect)], self.known, ["inspect"])
        terminal = next(line for line in no_repeat_grammar(schema).splitlines()
                        if line.startswith("IDENTITY_0:"))
        blocked = terminal.split(" & ~/", 1)[1][:-1]
        for angle, expected in [(30.000000000000004, True), (29.999999999999996, True),
                                (30, True), (30.000001, False), (30.01, False)]:
            value = {k: self.inspect[k] for k in
                     ("action", "focus_ids", "context", "azimuth_deg", "elevation_deg", "frame")}
            value["elevation_deg"] = angle
            self.assertEqual(re.fullmatch(blocked, json.dumps(value, separators=(',', ':'))[1:-1])
                             is not None, expected, angle)

    def test_exhaustive_groups_match_canonical_observation_identity(self):
        validator = self.validator(self.inspect)
        for size in range(1, 5):
            for group in combinations(self.known, size):
                for order in permutations(group):
                    value = {**self.inspect, "focus_ids": list(order)}
                    self.assertEqual(validator.is_valid(value), set(group) != {1, 2}, value)

    def test_distinct_context_pose_action_and_frame_remain_available(self):
        validator = self.validator(self.inspect, self.measure)
        for change in ({"context": "scene"}, {"azimuth_deg": 316}, {"elevation_deg": 31}):
            self.assertTrue(validator.is_valid({**self.inspect, **change}), change)
        self.assertTrue(validator.is_valid({**self.inspect, "action": "finish", "focus_ids": []}))
        framed = {**self.measure, "frame": {"origin_ids": [1], "facing_ids": [2]}}
        self.assertTrue(validator.is_valid(framed))
        validator = self.validator(framed)
        self.assertFalse(validator.is_valid(framed))
        for change in ({"origin_ids": [], "facing_ids": [2]},
                       {"origin_ids": [1], "facing_ids": [3]}):
            self.assertTrue(validator.is_valid({**framed, "frame": change}))
        self.assertTrue(validator.is_valid(self.measure))

    def test_fractional_camera_bypass_is_blocked_before_and_after_acquisition(self):
        for acquired in ([], [evidence_action_key(self.inspect)]):
            schema = exclude_acquired_actions(self.schema, acquired, self.known,
                                              ["inspect", "measure", "finish"])
            validator = Draft202012Validator(schema)
            for angle in (30.0000035, 30.00000253689325, 30.01):
                self.assertFalse(validator.is_valid({**self.inspect, "elevation_deg": angle}))
            self.assertTrue(validator.is_valid({**self.inspect, "elevation_deg": 31}))

    def test_grammar_disallows_micro_angle_changes_and_retains_whole_degrees(self):
        branch = deepcopy(self.schema)
        branch["properties"]["action"] = enum("inspect")
        schema = exclude_acquired_actions({"anyOf": [branch]},
            [evidence_action_key(self.inspect)], self.known, ["inspect"])
        terminal = next(line for line in no_repeat_grammar(schema).splitlines()
                        if line.startswith("IDENTITY_0:"))
        allowed = terminal.split(": /", 1)[1].split("/ & ~/", 1)[0]
        for angle, expected in ((30.0000035, False), (30.01, False), (31, True), (31.0, True)):
            value = {k: self.inspect[k] for k in
                     ("action", "focus_ids", "context", "azimuth_deg", "elevation_deg", "frame")}
            value["elevation_deg"] = angle
            self.assertEqual(re.fullmatch(allowed, json.dumps(value, separators=(',', ':'))[1:-1])
                             is not None, expected, angle)

    def test_rejected_duplicate_groups_preserve_every_distinct_id_order(self):
        for action_name in ('inspect', 'measure'):
            branch = deepcopy(self.schema)
            branch['properties']['action'] = enum(action_name)
            schema = exclude_acquired_actions({'anyOf': [branch]}, [], self.known,
                                              [action_name], invalid_focus_groups=[[1, 2, 1]])
            self.assertIn('x-eviground-acquired-actions', schema)
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema)
            terminal = next(line for line in no_repeat_grammar(schema).splitlines()
                            if line.startswith('IDENTITY_0:'))
            blocked = terminal.split(' & ~/', 1)[1][:-1]
            for size in range(1, 5):
                for ids in product(self.known, repeat=size):
                    value = {**self.inspect, 'action': action_name, 'focus_ids': list(ids)}
                    if action_name == 'measure':
                        value['context'] = None
                    identity = {key: value[key] for key in ('action', 'focus_ids', 'context',
                                'azimuth_deg', 'elevation_deg', 'frame')}
                    text = json.dumps(identity, separators=(',', ':'))[1:-1]
                    rejected_group = len(ids) == 3 and set(ids) <= {1, 2}
                    self.assertEqual(validator.is_valid(value), not rejected_group, ids)
                    self.assertEqual(re.fullmatch(blocked, text) is not None, rejected_group, ids)
                    if len(set(ids)) == len(ids):
                        self.assertTrue(validator.is_valid(value), ids)

    def test_framed_measurement_order_cannot_evade_exclusion(self):
        framed = {**self.measure, "frame": {"origin_ids": [1, 2], "facing_ids": [3, 4]}}
        validator = self.validator(framed)
        for origin in permutations([1, 2]):
            for facing in permutations([3, 4]):
                self.assertFalse(validator.is_valid({**framed, "frame": {
                    "origin_ids": list(origin), "facing_ids": list(facing)}}))

    def test_duplicate_mask_preserves_numeric_prefixes_and_whitespace(self):
        known = [-1, 1, 10, 20]
        branch = deepcopy(self.schema)
        branch['properties']['action'] = enum('measure')
        branch['properties']['focus_ids']['items']['enum'] = known
        schema = exclude_acquired_actions({'anyOf': [branch]}, [], known, ['measure'])
        self.assertNotIn('x-eviground-acquired-actions', schema)
        schema = exclude_acquired_actions({'anyOf': [branch]}, [], known, ['measure'],
                                          invalid_focus_groups=[[1, 10, 1]])
        terminal = next(line for line in no_repeat_grammar(schema).splitlines()
                        if line.startswith('IDENTITY_0:'))
        blocked = terminal.split(' & ~/', 1)[1][:-1]
        for ids in ([1, 10], [-1, 1, 10, 20], [10, 1, 10], [-1, 1, -1]):
            value = {key: self.measure[key] for key in ('action', 'focus_ids', 'context',
                     'azimuth_deg', 'elevation_deg', 'frame')}
            value['focus_ids'] = ids
            for text in (json.dumps(value), json.dumps(value, indent=2)):
                self.assertEqual(re.fullmatch(blocked, text[1:-1].strip()) is not None,
                                 len(ids) == 3 and set(ids) <= {1, 10}, (ids, text))

    def test_invalid_group_masks_cannot_remove_distinct_or_unknown_ids(self):
        for ids in ([], [1, 2], [1, 99, 1], [1, True, 1]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'repeated known IDs'):
                exclude_acquired_actions(self.schema, [], self.known, ['inspect'],
                                          invalid_focus_groups=[ids])

    def test_native_constraints_and_input_schema_are_preserved(self):
        original = deepcopy(self.schema)
        validator = self.validator(self.inspect)
        self.assertFalse(validator.is_valid({**self.inspect, "azimuth_deg": 360}))
        self.assertFalse(validator.is_valid({**self.inspect, "focus_ids": [99]}))
        self.assertFalse(validator.is_valid({**self.inspect, "context": "scene", "extra": True}))
        self.assertEqual(self.schema, original)
        self.assertIs(exclude_acquired_actions(self.schema, [], self.known, []), self.schema)

    def test_wire_uses_one_guidance_transport_and_rejects_other_backends(self):
        from eviground.api.client import APIError, request_options

        branch = deepcopy(self.schema)
        branch["properties"]["action"] = enum("inspect")
        schema = exclude_acquired_actions({"anyOf": [branch]},
            [evidence_action_key(self.inspect)], self.known, ["inspect"])
        response_schema = {"name": "react_action", "strict": True, "schema": schema}
        with patch.dict(os.environ, {"EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            options = request_options(100, None, True, response_schema)
        self.assertIn("guided_grammar", options)
        self.assertNotIn("response_format", options)
        self.assertEqual(options["guided_decoding_backend"], "guidance")
        with patch.dict(os.environ, {"EVIGROUND_GUIDED_DECODING_BACKEND": ""}):
            with self.assertRaisesRegex(APIError, "requires the Guidance"):
                request_options(100, None, True, response_schema)

    def test_first_request_retains_native_schema_until_an_exclusion_is_known(self):
        from eviground.api import client

        branch = deepcopy(self.schema)
        branch['properties']['action'] = enum('inspect')
        guarded = exclude_acquired_actions({'anyOf': [branch]}, [], self.known, ['inspect'])
        first = {'name': 'react_action', 'strict': True, 'schema': guarded}
        self.assertNotIn('allOf', guarded)
        self.assertNotIn('x-eviground-acquired-actions', guarded)
        messages = [{'role': 'system', 'content': 'Use the scene evidence.'}]
        with patch.dict(os.environ, {'EVIGROUND_GUIDED_DECODING_BACKEND': 'guidance',
                                    'EVIGROUND_PROMPT_PROFILE': 'compact'}):
            options = client.request_options(4000, None, True, first)
            self.assertIn('response_format', options)
            guarded = exclude_acquired_actions({'anyOf': [branch]}, [], self.known, ['inspect'],
                                              invalid_focus_groups=[[1, 2, 1]])
            schema = {'name': 'react_action', 'strict': True, 'schema': guarded}
            options = client.request_options(4000, None, True, schema)
            result = client._schema_messages(messages, schema)
        self.assertIn('guided_grammar', options)
        self.assertNotIn('response_format', options)
        self.assertNotIn('Output JSON Schema:', result[0]['content'])

    def test_grammar_masks_frames_after_sorted_history_round_trip(self):
        # The controller persists signatures with sort_keys=True, which reverses
        # the frame property order relative to the model's wire representation.
        branch = deepcopy(self.schema)
        branch["properties"]["action"] = enum("measure")
        framed = {**self.measure, "frame": {"origin_ids": [1, 2], "facing_ids": [3, 4]}}
        acquired = [evidence_action_key(framed)]
        original = exclude_acquired_actions({"anyOf": [branch]}, acquired, self.known, ["measure"])
        persisted = deepcopy(original)
        persisted["x-eviground-acquired-actions"] = json.loads(json.dumps(acquired, sort_keys=True))
        self.assertEqual(no_repeat_grammar(original), no_repeat_grammar(persisted))
        grammar = no_repeat_grammar(persisted)
        terminal = next(line for line in grammar.splitlines() if line.startswith("IDENTITY_0:"))
        blocked = terminal.split(" & ~/", 1)[1][:-1]
        for focus in permutations([1, 2]):
            for origin in permutations([1, 2]):
                for facing in permutations([3, 4]):
                    prefix = dict(action="measure", focus_ids=list(focus), context=None,
                        azimuth_deg=None, elevation_deg=None,
                        frame={"origin_ids": list(origin), "facing_ids": list(facing)})
                    wire = json.dumps(prefix, separators=(",", ":"))[1:-1]
                    self.assertIsNotNone(re.fullmatch(blocked, wire), prefix)
        prefix["frame"] = {"origin_ids": [1], "facing_ids": [3, 4]}
        self.assertIsNone(re.fullmatch(blocked, json.dumps(prefix, separators=(",", ":"))[1:-1]))

    def test_prompt_omits_expanded_masks_but_preserves_feedback_and_decoder(self):
        from eviground.api import client

        branch = deepcopy(self.schema)
        branch["properties"]["action"] = enum("inspect")
        base = {"anyOf": [branch]}
        acquired = [evidence_action_key(self.inspect)]
        guarded = exclude_acquired_actions(base, acquired, self.known, ["inspect"])
        original = deepcopy(guarded)
        schema = dict(name="react_action", strict=True, schema=guarded)
        messages = [{"role": "system", "content": "Do not repeat acquired observations."},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(
                        {"unavailable_evidence_actions": acquired})}]}]
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact",
                                    "EVIGROUND_GUIDED_DECODING_BACKEND": "guidance"}):
            result = client._schema_messages(messages, schema)
            options = client.request_options(4000, None, True, schema)
        self.assertNotIn("Output JSON Schema:", result[0]["content"])
        self.assertTrue(result[0]["content"].startswith(messages[0]["content"]))
        self.assertEqual(result[-1], messages[-1])
        self.assertEqual(guarded, original)
        self.assertEqual(options["guided_grammar"], no_repeat_grammar(client._wire_schema(guarded)))
        self.assertFalse(Draft202012Validator(guarded).is_valid(self.inspect))
        self.assertEqual(options["max_completion_tokens"] if "max_completion_tokens" in options
                         else options["max_tokens"], 4000)
        guarded["allOf"].append({"type": "object"})
        with patch.dict(os.environ, {"EVIGROUND_PROMPT_PROFILE": "compact"}):
            with self.assertRaisesRegex(ValueError, "unrelated schema intersections"):
                client._schema_messages(messages, schema)


if __name__ == "__main__":
    unittest.main()
