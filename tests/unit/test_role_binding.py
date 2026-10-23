"""Entry regressions based on the first real module acceptance traces."""

import unittest
from eviground.query.validation import validate_structure
from eviground.query.query_rewrite import rewrite_query


def obj(oid, category, z, height=1.0):
    return {
        "bbox_id": oid,
        "target": category,
        "bbox_3d": [0.0, 0.0, z, 1.0, 1.0, height],
    }


def raw(target, anchors, relations=(), attrs=()):
    return {
        "Target": [target],
        "Anchor": list(anchors),
        "Relations": list(relations),
        "Attributes": list(attrs),
        "Observer": {},
        "Unhandled": [],
    }


class SourceRoleBindingTests(unittest.TestCase):

    def test_unqualified_shared_color_stays_missing(self):
        query = "shelf with a shelf above it; one is red"
        data = validate_structure(
            query, raw("shelf", ["shelf"], [], [{"category": "shelf", "phrase": "red"}])
        )
        self.assertEqual(data["Attributes"], [])
        self.assertIn(
            "attribute target/reference role is ambiguous: red",
            data["diagnostic_missing"],
        )

    def test_repeated_color_same_category_requires_both_roles(self):
        query = "The red shelf above the red shelf."
        data = validate_structure(
            query,
            raw(
                "shelf",
                ["shelf"],
                [
                    {
                        "target_category": "shelf",
                        "anchor_category": "shelf",
                        "relation": "above",
                        "phrase": "above the red shelf",
                    }
                ],
                [{"category": "shelf", "role": "target", "phrase": "red"}],
            ),
        )
        self.assertIn(
            "appearance condition role not represented: red (reference shelf)",
            data["diagnostic_missing"],
        )

    def test_parser_cannot_invent_comparison_scope_metadata(self):
        data = validate_structure(
            "the smaller of the two cabinets",
            raw(
                "cabinet",
                [],
                [],
                [
                    {
                        "category": "cabinet",
                        "phrase": "smaller",
                        "geometry_attribute": "smallest",
                        "comparison_scope": "full_category",
                        "scope_phrase": "the smaller of the two cabinets",
                    }
                ],
            ),
        )
        self.assertNotIn("geometry_attribute", data["Attributes"][0])
        self.assertEqual(data["Attributes"][0]["phrase"], "smaller")
        self.assertEqual(data["Attributes"][0]["comparison_scope"], "explicit_pair")
        self.assertEqual(data["Attributes"][0]["required_member_count"], 2)
        self.assertEqual(data["diagnostic_missing"], [])

    def test_unquoted_pair_metadata_cannot_restrict_comparison_members(self):
        query = "the smaller cabinet"
        data = validate_structure(
            query,
            raw(
                "cabinet",
                [],
                [],
                [
                    {
                        "category": "cabinet",
                        "phrase": "smaller",
                        "geometry_attribute": "smallest",
                        "comparison_scope": "explicit_pair",
                        "required_member_count": 2,
                        "scope_phrase": "the smaller of the two cabinets",
                    }
                ],
            ),
        )
        (attribute,) = data["Attributes"]
        self.assertEqual(attribute["phrase"], "smaller")
        self.assertNotIn("geometry_attribute", attribute)
        self.assertNotIn("comparison_scope", attribute)
        self.assertNotIn("required_member_count", attribute)
        (condition,) = rewrite_query(query, data)["constraints"][0]["size_conditions"]
        self.assertEqual(condition["scope_kind"], "full_category")
        self.assertEqual(condition["original_quote"], "smaller")
        self.assertNotIn("required_member_count", condition)


class RelationScopeTests(unittest.TestCase):

    def test_live_parser_shape_recovers_correct_one_from_explicit_facing_group(self):
        query = "When standing in the middle of the room facing the windows the correct one is on the right."
        data = raw(
            "",
            ["room"],
            [
                {
                    "target_category": "window",
                    "anchor_category": "room",
                    "anchor_categories": ["room"],
                    "relation": "right",
                    "scope": "relative_anchor",
                    "phrase": "the correct one is on the right",
                }
            ],
        )
        data["Target"] = []
        data["Observer"] = {
            "origin": {
                "kind": "room_center",
                "category": "",
                "phrase": "standing in the middle of the room",
            },
            "facing": {
                "category": "windows",
                "group": False,
                "phrase": "facing the windows",
            },
        }
        data["Unhandled"] = [query]
        data = validate_structure(query, data)
        self.assertEqual(data["Target"], ["window"])
        self.assertEqual(data["Anchor"], [])
        self.assertEqual(data["Relations"][0]["scope"], "side")
        self.assertEqual(
            data["target_coreferences"],
            [{"phrase": "the correct one", "category": "window"}],
        )
        self.assertEqual(data["Unhandled"], [])
        self.assertFalse(data["diagnostic_missing"])

    def test_complete_sentence_recovery_keeps_extra_condition_missing(self):
        query = "When standing in the middle of the room facing the windows the correct one is on the right, not the left one."
        data = raw(
            "",
            ["room"],
            [
                {
                    "target_category": "window",
                    "anchor_category": "room",
                    "anchor_categories": ["room"],
                    "relation": "right",
                    "scope": "relative_anchor",
                    "phrase": "the correct one is on the right",
                }
            ],
        )
        data["Target"] = []
        data["Observer"] = {
            "origin": {
                "kind": "room_center",
                "category": "",
                "phrase": "standing in the middle of the room",
            },
            "facing": {
                "category": "windows",
                "group": False,
                "phrase": "facing the windows",
            },
        }
        data["Unhandled"] = [query]
        data = validate_structure(query, data)
        self.assertEqual(data["Target"], [])
        self.assertEqual(data["Unhandled"], [query])
        self.assertTrue(
            any(
                (
                    item.startswith("unhandled condition:")
                    for item in data["diagnostic_missing"]
                )
            )
        )

    def test_plain_side_quote_can_use_explicit_observer(self):
        query = "When standing in the middle of the room facing the windows the correct one is on the right."
        data = raw(
            "window",
            [],
            [
                {
                    "target_category": "window",
                    "anchor_category": "",
                    "relation": "right",
                    "scope": "relative_anchor",
                    "phrase": "the correct one is on the right",
                }
            ],
        )
        data["Unhandled"] = ["the correct one"]
        data["Observer"] = {
            "origin": {
                "kind": "room_center",
                "phrase": "standing in the middle of the room",
            },
            "facing": {
                "category": "windows",
                "group": False,
                "phrase": "facing the windows",
            },
        }
        data = validate_structure(query, data)
        self.assertEqual(data["Relations"][0]["scope"], "side")
        self.assertTrue(data["Observer"]["facing"]["group"])
        self.assertEqual(
            data["target_coreferences"],
            [{"phrase": "the correct one", "category": "window"}],
        )
        self.assertEqual(data["Unhandled"], [])
        coreference = next(
            (
                record
                for record in data["condition_records"]
                if record["condition_type"] == "target_coreference"
            )
        )
        self.assertEqual(coreference["target_role"], {"category": "window"})
        self.assertEqual(coreference["binding_status"], "bound")
        self.assertFalse(data["diagnostic_missing"])

    def test_left_window_keeps_condition_with_missing_observer(self):
        data = validate_structure(
            "The left window.",
            raw(
                "window",
                [],
                [
                    {
                        "target_category": "window",
                        "anchor_category": "",
                        "relation": "left",
                        "scope": "",
                        "phrase": "The left window",
                    }
                ],
            ),
        )
        self.assertEqual(data["Relations"][0]["scope"], "side")
        self.assertIn(
            "observer origin or facing direction is unspecified",
            data["diagnostic_missing"],
        )

    def test_target_coreference_stays_missing_without_observer_frame(self):
        query = "The correct one is on the right."
        data = raw(
            "window",
            [],
            [
                {
                    "target_category": "window",
                    "anchor_category": "",
                    "relation": "right",
                    "scope": "side",
                    "phrase": query,
                }
            ],
        )
        data["Unhandled"] = ["the correct one"]
        data = validate_structure(query, data)
        self.assertEqual(data["target_coreferences"], [])
        self.assertEqual(data["Unhandled"], ["the correct one"])
        self.assertTrue(data["diagnostic_missing"])

    def test_missing_relative_anchor_and_extreme_cannot_become_side(self):
        for quote in ("left of the cabinet", "on the far left", "on the leftmost side"):
            with self.subTest(quote=quote):
                data = validate_structure(
                    "window " + quote,
                    raw(
                        "window",
                        [],
                        [
                            {
                                "target_category": "window",
                                "anchor_category": "",
                                "relation": "left",
                                "scope": "",
                                "phrase": quote,
                            }
                        ],
                    ),
                )
                self.assertEqual(data["Relations"], [])
                self.assertTrue(data["diagnostic_missing"])


if __name__ == "__main__":
    unittest.main()
