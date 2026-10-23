"""Regression contracts for rank-preserving synonym retrieval, without API calls."""
import os
import unittest
from unittest.mock import patch

from eviground.grounding.evidence import canonical_category, match_retrieval_categories, retrieval_category
from eviground.grounding.r8_grounding import query_evidence
from tests.legacy.test_r8_grounding import prediction
from eviground.grounding.v4_policy import settings


class R8RetrievalTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1',
            'EVIGROUND_V7_4': '1', 'EVIGROUND_V9': '1', 'EVIGROUND_V9_SOURCE': 'r8',
            'EVIGROUND_R8_GROUNDING_REVISION': 'r10'})
        env.start(); self.addCleanup(env.stop)

    def test_version_and_legacy_opt_out(self):
        self.assertEqual(settings()['version'], 'predicted-category-grounding-v9-r8-r10')
        with patch.dict(os.environ, {'EVIGROUND_R8_GROUNDING_REVISION': 'r5'}):
            self.assertEqual(canonical_category('sofa'), 'sofa')
            self.assertEqual(match_retrieval_categories(['sofa'], [prediction(1, ['couch'])]), [])

    def test_equivalent_category_names_and_modifiers(self):
        for name, expected in [('sofa', 'couch'), ('brown sofa', 'couch'),
                ('bookcases', 'bookshelf'), ('wooden cupboard', 'cabinet'),
                ('filing cabinets', 'file cabinet'), ('painting', 'picture'),
                ('artwork', 'picture'), ('picture frames', 'picture')]:
            with self.subTest(name=name):
                self.assertEqual([o['bbox_id'] for o in match_retrieval_categories(
                    [name], [prediction(7, [expected])])], [7])

    def test_synonyms_preserve_original_nouns_and_comparison_scopes(self):
        from eviground.geometry.size_semantics import size_specs
        self.assertEqual(canonical_category('sofa'), 'sofa')
        self.assertEqual(retrieval_category('brown sofa'), 'sofa')
        for noun in ('bookcase', 'sofa', 'cupboard'):
            query = 'the taller of the two ' + noun + 's'
            comparison = size_specs('taller', noun, query=query)[0]
            self.assertEqual(comparison['scope_kind'], 'explicit_pair')
            self.assertEqual(comparison['required_member_count'], 2)

    def test_distinct_types_and_compound_heads_do_not_merge(self):
        for name, other in [('sofa chair', 'couch'), ('doorframe', 'door'),
                ('blackboard', 'whiteboard'), ('pillow', 'cushion'), ('desk', 'table'),
                ('seat', 'chair'), ('frame', 'picture')]:
            with self.subTest(name=name):
                self.assertEqual(match_retrieval_categories([name], [prediction(1, [other])]), [])

    def test_alternative_and_reference_preserve_rank_and_raw_labels(self):
        target = prediction(1, ['chair', 'couch', 'bed'])
        anchor = prediction(2, ['cabinet', 'bookshelf', 'table'])
        data = query_evidence({'Target': ['sofa'], 'Anchor': ['bookcase']},
                              [target, anchor], [target])
        binding = data['target_category_bindings'][0]
        self.assertFalse(binding['primary_match'])
        self.assertEqual([(h['category'], h['rank']) for h in binding['matched_hypotheses']],
                         [('couch', 2)])
        self.assertEqual(data['numeric_reference_distances'][0]['reference_id'], 2)
        self.assertEqual([h['category'] for h in target['category_hypotheses']],
                         ['chair', 'couch', 'bed'])


if __name__ == '__main__':
    unittest.main()
