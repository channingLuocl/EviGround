import unittest
from eviground.categories.retrieval import (
    canonical_category,
    match_categories,
    match_retrieval_categories,
    retrieval_category,
    ALIASES,
)


class EvidenceTests(unittest.TestCase):

    def test_aliases_and_no_partial_word_match(self):
        objects = [
            {"bbox_id": 1, "target": "tv"},
            {"bbox_id": 2, "target": "doorframe"},
            {"bbox_id": 3, "target": "door"},
            {"bbox_id": 4, "target": "trash can"},
        ]
        for name in ["TV", "television", "screen"]:
            self.assertEqual(
                [o["bbox_id"] for o in match_categories([name], objects)], [1]
            )
        self.assertEqual(canonical_category("trashcan"), "trash can")
        self.assertEqual(
            [o["bbox_id"] for o in match_categories(["front doors"], objects)], [3]
        )
        self.assertEqual(match_categories(["blackboard"], objects), [])

    def test_retrieval_strips_appearance_but_not_original_query(self):
        objects = [{"bbox_id": 1, "target": "tv"}, {"bbox_id": 2, "target": "door"}]
        self.assertEqual(
            match_retrieval_categories(["large black screen"], objects), [objects[0]]
        )
        self.assertEqual(
            match_retrieval_categories(["front doors"], objects), [objects[1]]
        )

    def test_category_normalization_is_idempotent_for_supported_aliases(self):
        names = list(ALIASES) + ["double doors", "carts", "photocopier", "photocopiers"]
        for name in names:
            with self.subTest(name=name):
                normalized = canonical_category(name)
                self.assertEqual(canonical_category(normalized), normalized)
                retrieved = retrieval_category("large black " + name)
                self.assertEqual(retrieval_category(retrieved), retrieved)
        self.assertEqual(canonical_category("double doors"), "door")
        self.assertEqual(canonical_category("carts"), "cart")
        for name in ("photocopier", "photocopiers", "copiers"):
            self.assertEqual(canonical_category(name), "copier")

    def test_category_aliases_do_not_merge_distinct_categories_or_strip_plural_suffixes(
        self,
    ):
        for name in ("printer", "desk", "glass", "cactus", "doorframe", "cartwheel"):
            with self.subTest(name=name):
                self.assertEqual(canonical_category(name), name)
        objects = [
            {"bbox_id": 1, "target": "copier"},
            {"bbox_id": 2, "target": "table"},
            {"bbox_id": 3, "target": "cartwheel"},
        ]
        for name in ("printer", "desk", "cart"):
            self.assertEqual(match_categories(name, objects), [])


if __name__ == "__main__":
    unittest.main()
