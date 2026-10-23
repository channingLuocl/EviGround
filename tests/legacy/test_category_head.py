import unittest

from eviground.grounding.evidence import match_categories, match_retrieval_categories


class CategoryHeadTests(unittest.TestCase):
    def test_whole_object_does_not_match_its_named_part(self):
        for category, part in [('bathroom stall', 'bathroom stall door'),
                               ('table', 'table lamp'), ('window', 'window frame')]:
            with self.subTest(category=category):
                objects = [{'bbox_id': 1, 'target': category}, {'bbox_id': 2, 'target': part}]
                self.assertEqual([o['bbox_id'] for o in match_categories(category, objects)], [1])
                self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(category, objects)], [1])
                self.assertEqual([o['bbox_id'] for o in match_categories(part, objects)], [2])
                # No exact whole label is not permission to substitute its part.
                self.assertEqual(match_categories(category, objects[1:]), [])

    def test_compound_subtypes_and_explicit_parts_keep_their_head(self):
        objects = [{'bbox_id': 1, 'target': 'coffee table'},
                   {'bbox_id': 2, 'target': 'kitchen cabinets'},
                   {'bbox_id': 3, 'target': 'bathroom stall door'},
                   {'bbox_id': 4, 'target': 'table lamp'}]
        for category, expected in [('table', [1]), ('cabinet', [2]), ('door', [3]),
                                    ('lamp', [4]), ('bathroom stall door', [3])]:
            with self.subTest(category=category):
                self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(category, objects)], expected)


if __name__ == '__main__':
    unittest.main()
