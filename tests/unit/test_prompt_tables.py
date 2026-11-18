"""Prompt compression must retain every value, list order, and original input."""
from copy import deepcopy
import json
import unittest

from eviground.grounding.react import compact_prompt_tables, compact_state


def expand(value):
    if isinstance(value, list):
        return [expand(child) for child in value]
    if not isinstance(value, dict):
        return value
    if set(value) in ({"columns", "rows"}, {"columns", "rows", "constants"}):
        return [{**expand(value.get("constants", {})),
                 **{key: expand(child) for key, child in zip(value["columns"], row)}}
                for row in value["rows"]]
    return {key: expand(child) for key, child in value.items()}


class PromptTableTests(unittest.TestCase):
    def test_nested_arrays_round_trip_with_different_key_order(self):
        original = [{"object_identifier": i, "category_hypotheses": [
            {"category_name": "chair", "rank": 1},
            {"rank": 2, "category_name": "table"}]} for i in (3, 1, 7)]
        before = deepcopy(original)
        encoded = compact_prompt_tables(original)
        self.assertEqual(expand(encoded), original)
        self.assertEqual(original, before)
        self.assertLess(len(json.dumps(encoded)), len(json.dumps(original)))

    def test_short_heterogeneous_empty_and_scalar_values_are_preserved(self):
        for value in ([], [{}], [{"x": 1}, {"y": 2}], [1, None, True, "中文"],
                      [{"a": 1}, {"a": 2}], {"nullable": None}):
            self.assertEqual(compact_prompt_tables(value), value)

    def test_small_repeated_fields_use_reference_only_when_shorter(self):
        mapping = {"q0": ["n11"], "q1": ["n0"], "q2": ["n1"], "q3": ["n2"]}
        state = {"observations": [], "previous_actions": [
            {"monitor": {"requirement_nodes": mapping, "node_updates": []}},
            {"monitor": {"requirement_nodes": mapping, "node_updates": []}}]}
        original = deepcopy(state)
        encoded = compact_state(state)
        self.assertEqual(encoded["previous_actions"][1]["monitor"]["requirement_nodes"],
                         {"same_as_action": 0})
        self.assertEqual(encoded["previous_actions"][1]["monitor"]["node_updates"], [])
        self.assertEqual(state, original)

    def test_rejected_initial_plan_history_without_accepted_dag(self):
        state = {"task_dag": None, "observations": [], "previous_actions": [
            {"monitor": {"nodes": [{"id": "n0", "goal": "still unresolved"}]}}]}
        self.assertEqual(compact_state(state), state)

    def test_constant_columns_retain_order_and_different_nested_values(self):
        value = [{"source_quotes": ["the chair beside the entrance"],
                  "status": "pending", "id": i} for i in range(11)]
        encoded = compact_prompt_tables(value)
        self.assertIn("constants", encoded)
        self.assertEqual(expand(encoded), value)

    def test_constant_detection_distinguishes_boolean_integer_and_float(self):
        value = [{"value": child, "nested": [child], "id": i}
                 for i, child in enumerate((True, 1, 1.0))]
        restored = expand(compact_prompt_tables(value))
        self.assertEqual(json.dumps(restored, sort_keys=True), json.dumps(value, sort_keys=True))

    def test_view_references_resolve_exactly_without_reusing_changed_camera(self):
        views = [{"at": [1.1234567891234567, 2.1234567891234567, 3.1234567891234567],
                  "elevation_deg": angle, "provenance": "observed CPU point-cloud view"}
                 for angle in (30, 31, 32)]
        state = {"observations": [{"evidence_id": f"observation_{i + 1}", "views": [view]}
                                  for i, view in enumerate(views)]}
        original = deepcopy(state)
        encoded = compact_state(state)
        self.assertIn("same_as_view", encoded["observations"][1]["views"][0]["at"])
        restored = []
        for observation in encoded["observations"]:
            view = deepcopy(observation["views"][0])
            for key, value in view.items():
                if isinstance(value, dict) and set(value) == {"same_as_view"}:
                    evidence_id, index = value["same_as_view"]
                    self.assertEqual(index, 0)
                    view[key] = deepcopy(restored[int(evidence_id.split("_")[-1]) - 1][key])
            restored.append(view)
        self.assertEqual(restored, views)
        self.assertEqual(state, original)


if __name__ == "__main__":
    unittest.main()
