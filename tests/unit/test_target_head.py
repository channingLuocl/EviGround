"""Source hints preserve a topic without overriding explicit target changes."""

import unittest
from eviground.grounding.bindings import target_head_hint


class TargetHeadTests(unittest.TestCase):
    def test_copular_topic_is_not_its_later_reference(self):
        query = "the window is in the back of the room. there are desks in front of it."
        self.assertEqual(target_head_hint(query, ["window", "desk"]),
                         {"category": "window", "source_quote": "window"})

    def test_quantified_existential_topic_retains_exact_quote(self):
        for continuation in ("it has a microwave in the middle of it.",
                             "there is a microwave in the middle of them."):
            query = "there is a set of bottom kitchen cabinets in the room. " + continuation
            self.assertEqual(target_head_hint(query, ["kitchen cabinet", "microwave"]),
                             {"category": "kitchen cabinet", "source_quote": "set of bottom kitchen cabinets"})

    def test_explicit_later_target_change_does_not_get_a_first_noun_hint(self):
        queries = ["there is a large painting on the wall. this the long skinny table under it.",
                   "there is a sofa in the room. select the table behind it."]
        for query in queries:
            with self.subTest(query=query):
                self.assertIsNone(target_head_hint(query, ["painting", "sofa", "table"]))

    def test_hint_does_not_invent_a_category_absent_from_hypotheses(self):
        self.assertIsNone(target_head_hint("the window is behind the desk", ["desk"]))
