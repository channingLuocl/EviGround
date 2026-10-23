"""Quoted noun phrases cannot transfer reference properties to a target."""

import unittest
from eviground.query.validation import _source_attribute_bindings


def parsed(attrs, unhandled=()):
    return {
        "Target": ["trash can"],
        "Anchor": ["trash can", "trash can"],
        "Relations": [
            {
                "target_category": "trash can",
                "anchor_categories": ["trash can", "trash can"],
                "relation": "between",
                "phrase": "in between another trash can and a smaller trash can with a blue bag",
            }
        ],
        "Attributes": attrs,
        "Observer": {},
        "Unhandled": list(unhandled),
    }


QUERY = "The trash can that is in between another trash can and a smaller trash can with a blue bag."


class EmbeddedNounRoleTests(unittest.TestCase):

    def test_direct_connectors_allow_optional_quoted_determiner(self):
        data = {"Target": ["chair"], "Anchor": [], "Observer": {}}
        for connector in ("with", "is", "has", "that is", "that has"):
            for article in ("", "a ", "an ", "the "):
                query = "the chair " + connector + " " + article + "blue cushion"
                for phrase in (
                    "blue cushion",
                    connector + " " + article + "blue cushion",
                ):
                    with self.subTest(query=query, phrase=phrase):
                        self.assertEqual(
                            _source_attribute_bindings(query, data, phrase),
                            {("chair", "target")},
                        )

    def test_postnoun_quote_does_not_cross_a_modeled_head(self):
        data = {"Target": ["table"], "Anchor": ["chair"], "Observer": {}}
        query = "the table with a blue chair"
        self.assertEqual(
            _source_attribute_bindings(query, data, "blue chair"),
            {("chair", "reference")},
        )
        self.assertEqual(
            _source_attribute_bindings(query, data, "with a blue chair"), set()
        )
        self.assertEqual(
            _source_attribute_bindings(
                "the table near the chair with a blue cushion", data, "blue cushion"
            ),
            {("chair", "reference")},
        )
        self.assertEqual(
            _source_attribute_bindings(
                "the table with a blue cushion near a box",
                data,
                "with a blue cushion near a box",
            ),
            set(),
        )

    def test_repeated_postnoun_property_keeps_both_language_roles(self):
        data = {"Target": ["chair"], "Anchor": ["chair"], "Observer": {}}
        query = "the chair with a blue cushion above the chair with a blue cushion"
        self.assertEqual(
            _source_attribute_bindings(query, data, "blue cushion"),
            {("chair", "target"), ("chair", "reference")},
        )

    def test_postnoun_attachment_respects_observer_and_selection(self):
        data = {
            "Target": ["chair"],
            "Anchor": ["chair"],
            "Observer": {
                "facing": {
                    "category": "chair",
                    "phrase": "facing a chair with a blue cushion",
                }
            },
        }
        self.assertEqual(
            _source_attribute_bindings(
                "Facing a chair with a blue cushion, select the chair.",
                data,
                "blue cushion",
            ),
            {("chair", "reference")},
        )
        self.assertEqual(
            _source_attribute_bindings(
                "Facing a chair, select the chair with a blue cushion.",
                data,
                "blue cushion",
            ),
            {("chair", "target")},
        )

    def test_repeated_phrase_does_not_select_one_language_role(self):
        query = "The red trash can above the red trash can."
        data = {
            "Target": ["trash can"],
            "Anchor": ["trash can"],
            "Relations": [],
            "Observer": {},
        }
        self.assertEqual(
            _source_attribute_bindings(query, data, "red trash can"),
            {("trash can", "target"), ("trash can", "reference")},
        )

    def test_relation_quote_is_not_a_single_head_noun_attribute(self):
        data = {
            "Target": ["chair"],
            "Anchor": ["table"],
            "Relations": [],
            "Observer": {},
        }
        self.assertEqual(
            _source_attribute_bindings(
                "the table near a red chair", data, "near a red chair"
            ),
            set(),
        )
        self.assertEqual(
            _source_attribute_bindings(
                "the red chair above a table", data, "red chair above a table"
            ),
            set(),
        )
        self.assertEqual(
            _source_attribute_bindings(
                "the red chair near a blue box", data, "red chair near a blue box"
            ),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
