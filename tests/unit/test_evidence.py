import unittest
from seeground.grounding.evidence import (canonical_category, match_categories, distance_evidence,
                      match_retrieval_categories, distance_rankings, propose_category_checks,
                      distance_pair_metrics, retrieval_category, ALIASES)
from seeground.grounding.evidence import validate_distance_relations


class EvidenceTests(unittest.TestCase):
    def test_aliases_and_no_partial_word_match(self):
        objects=[{'bbox_id':1,'target':'tv'},{'bbox_id':2,'target':'doorframe'},
                 {'bbox_id':3,'target':'door'},{'bbox_id':4,'target':'trash can'}]
        for name in ['TV','television','screen']:
            self.assertEqual([o['bbox_id'] for o in match_categories([name],objects)],[1])
        self.assertEqual(canonical_category('trashcan'),'trash can')
        self.assertEqual([o['bbox_id'] for o in match_categories(['front doors'],objects)],[3])
        self.assertEqual(match_categories(['blackboard'],objects),[])

    def test_distances_preserve_each_reference(self):
        targets=[{'bbox_id':1,'bbox_3d':[0,0,0,2,2,2]},
                 {'bbox_id':2,'bbox_3d':[3,4,0,2,2,2]}]
        evidence=distance_evidence(targets,[targets[0]])
        self.assertEqual(len(evidence['pairs']),1)
        self.assertEqual(evidence['pairs'][0]['center_distance_m'],5.)
        self.assertAlmostEqual(evidence['pairs'][0]['aabb_gap_m'],5**.5,places=4)
        self.assertEqual(distance_evidence(targets,[])['pairs'],[])

    def test_retrieval_strips_appearance_but_not_original_query(self):
        objects=[{'bbox_id':1,'target':'tv'},{'bbox_id':2,'target':'door'}]
        self.assertEqual(match_retrieval_categories(['large black screen'],objects),[objects[0]])
        self.assertEqual(match_retrieval_categories(['front doors'],objects),[objects[1]])

    def test_category_normalization_is_idempotent_for_supported_aliases(self):
        names = list(ALIASES) + ['double doors', 'carts', 'photocopier', 'photocopiers']
        for name in names:
            with self.subTest(name=name):
                normalized = canonical_category(name)
                self.assertEqual(canonical_category(normalized), normalized)
                retrieved = retrieval_category('large black ' + name)
                self.assertEqual(retrieval_category(retrieved), retrieved)
        self.assertEqual(canonical_category('double doors'), 'door')
        self.assertEqual(canonical_category('carts'), 'cart')
        for name in ('photocopier', 'photocopiers', 'copiers'):
            self.assertEqual(canonical_category(name), 'copier')

    def test_category_aliases_match_saved_real_query_snapshots(self):
        # Category-only snapshots from the 11970, 3311 and 29925 real runs.
        cases = [
            ('double doors', [{'bbox_id': 30, 'target': 'doors'},
                              {'bbox_id': 31, 'target': 'doorframe'}], [30]),
            ('photocopier', [{'bbox_id': 4, 'target': 'copier'},
                            {'bbox_id': 25, 'target': 'picture'}], [4]),
            ('carts', [{'bbox_id': oid, 'target': 'cart'} for oid in (7, 8, 9)],
             [7, 8, 9]),
            ('curtains', [{'bbox_id': 7, 'target': 'curtain'},
                          {'bbox_id': 9, 'target': 'curtain'},
                          {'bbox_id': 10, 'target': 'shelf'}], [7, 9]),
        ]
        for raw_name, objects, expected in cases:
            for name in (raw_name, canonical_category(raw_name), retrieval_category(raw_name)):
                with self.subTest(raw_name=raw_name, name=name):
                    self.assertEqual([obj['bbox_id'] for obj in match_categories(name, objects)],
                                     expected)
                    self.assertEqual([obj['bbox_id'] for obj in
                                      match_retrieval_categories(name, objects)], expected)
            normalized_objects = [{**obj, 'target': canonical_category(obj['target'])}
                                  for obj in objects]
            self.assertEqual([obj['bbox_id'] for obj in
                              match_categories(raw_name, normalized_objects)], expected)

    def test_category_aliases_do_not_merge_distinct_categories_or_strip_plural_suffixes(self):
        for name in ('printer', 'desk', 'glass', 'cactus', 'doorframe', 'cartwheel'):
            with self.subTest(name=name):
                self.assertEqual(canonical_category(name), name)
        objects = [{'bbox_id': 1, 'target': 'copier'},
                   {'bbox_id': 2, 'target': 'table'},
                   {'bbox_id': 3, 'target': 'cartwheel'}]
        for name in ('printer', 'desk', 'cart'):
            self.assertEqual(match_categories(name, objects), [])

    def test_rankings_keep_references_separate_and_ties(self):
        targets=[{'bbox_id':1,'target':'chair','bbox_3d':[1,0,0,1,1,1]},
                 {'bbox_id':2,'target':'chair','bbox_3d':[-1,0,0,1,1,1]},
                 {'bbox_id':3,'target':'chair','bbox_3d':[4,0,0,1,1,1]}]
        anchors=[{'bbox_id':10,'target':'table','bbox_3d':[0,0,0,1,1,1]},
                 {'bbox_id':11,'target':'table','bbox_3d':[5,0,0,1,1,1]}]
        query='The chair nearest the table.'
        parsed={'Relations':[{'target_category':'chair','anchor_category':'table',
                             'relation':'nearest','phrase':query}]}
        ranks=distance_rankings(query,parsed,targets,anchors)['rankings']
        self.assertEqual(ranks[0]['best_ids'],[1,2])
        self.assertEqual(ranks[1]['best_ids'],[3])
        parsed['Relations'][0].update(relation='farthest',phrase='The chair farthest from the table.')
        ranks=distance_rankings('The chair farthest from the table.',parsed,targets,anchors)['rankings']
        self.assertEqual(ranks[0]['best_ids'],[3])
        self.assertEqual(ranks[1]['best_ids'],[2])

    def test_distance_metrics_are_named_and_keep_surface_horizontal_separate(self):
        target = {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [3, 4, 2, 2, 2, 2]}
        anchor = {'bbox_id': 9, 'target': 'table', 'bbox_3d': [0, 0, 0, 2, 2, 2]}
        values = distance_pair_metrics(target, anchor)
        self.assertAlmostEqual(values['center'], 5.3851648, places=5)
        self.assertAlmostEqual(values['horizontal_center'], 5., places=5)
        self.assertAlmostEqual(values['aabb_gap'], values['surface_gap'], places=8)
        evidence = distance_evidence([target], [anchor])
        pair = evidence['pairs'][0]
        self.assertEqual(pair['metric_status']['viewpoint'], 'unknown')
        self.assertIn('horizontal_center', evidence['metric_definitions'])
        self.assertIn('surface_gap', evidence['metric_definitions'])

    def test_ranking_uses_requested_metric_and_explicit_viewpoint(self):
        targets = [
            {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [0, 0, 10, 1, 1, 1]},
            {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [3, 0, 0, 1, 1, 1]},
        ]
        anchor = {'bbox_id': 9, 'target': 'table', 'bbox_3d': [0, 0, 0, 1, 1, 1]}
        query = 'The chair nearest the table.'
        parsed = {'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                                 'relation': 'nearest', 'metric': 'horizontal_center',
                                 'phrase': query}]}
        result = distance_rankings(query, parsed, targets, [anchor])
        self.assertEqual(result['rankings'][0]['metric'], 'horizontal_center')
        self.assertEqual(result['rankings'][0]['best_ids'], [1])
        parsed['Relations'][0]['metric'] = 'viewpoint'
        missing = distance_rankings(query, parsed, targets, [anchor])
        self.assertFalse(missing['rankings'])
        self.assertIn('explicit numeric observer origin', missing['ignored'][0]['reason'])
        with_origin = distance_rankings(query, parsed, targets, [anchor], viewpoint=[0, 0, 0])
        self.assertEqual(with_origin['rankings'][0]['metric'], 'viewpoint')
        self.assertEqual(with_origin['rankings'][0]['best_ids'], [2])

    def test_near_equal_distances_remain_tied_without_id_preference(self):
        targets = [
            {'bbox_id': 8, 'target': 'chair', 'bbox_3d': [1., 0, 0, 1, 1, 1]},
            {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [1.006, 0, 0, 1, 1, 1]},
        ]
        anchor = {'bbox_id': 9, 'target': 'table', 'bbox_3d': [0, 0, 0, 1, 1, 1]}
        query = 'The chair nearest the table.'
        parsed = {'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                                 'relation': 'nearest', 'phrase': query}]}
        ranking = distance_rankings(query, parsed, targets, [anchor])['rankings'][0]
        self.assertEqual(ranking['best_ids'], [2, 8])
        self.assertTrue(ranking['near_equal'])
        self.assertEqual(ranking['exact_best_ids'], [8])

    def test_near_and_far_do_not_become_extreme_rankings(self):
        target = {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [0, 0, 0, 1, 1, 1]}
        anchor = {'bbox_id': 9, 'target': 'table', 'bbox_3d': [1, 0, 0, 1, 1, 1]}
        for relation_name in ('near', 'far'):
            query = 'The chair ' + relation_name + ' the table.'
            parsed = {'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                                     'relation': relation_name, 'phrase': query}]}
            result = distance_rankings(query, parsed, [target], [anchor])
            self.assertEqual(result['rankings'], [])
            self.assertIn('no calibrated distance threshold', result['ignored'][0]['reason'])

    def test_explicit_instance_bindings_do_not_cross_rank_anchors(self):
        targets = [
            {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [0, 0, 0, 1, 1, 1]},
            {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [10, 0, 0, 1, 1, 1]},
        ]
        anchors = [
            {'bbox_id': 10, 'target': 'table', 'bbox_3d': [1, 0, 0, 1, 1, 1]},
            {'bbox_id': 11, 'target': 'table', 'bbox_3d': [9, 0, 0, 1, 1, 1]},
        ]
        query = 'The chair nearest the table.'
        parsed = {'Relations': [{'target_category': 'chair',
                                 'anchor_category': 'table',
                                 'relation': 'nearest',
                                 'target_id': 2,
                                 'anchor_id': 10,
                                 'phrase': query}]}
        result = distance_rankings(query, parsed, targets, anchors)
        self.assertEqual(len(result['rankings']), 1)
        ranking = result['rankings'][0]
        self.assertEqual(ranking['anchor_id'], 10)
        self.assertEqual(ranking['best_ids'], [2])
        self.assertEqual([item['target_id'] for item in ranking['ordered_targets']], [2])

    def test_plane_metric_requires_valid_finite_wall_segment(self):
        target = {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [2, 2, 1, 1, 1, 1]}
        wall = {'bbox_id': 9, 'target': 'wall', 'bbox_3d': [0, 0, 1, 10, 0.2, 3],
                'wall_geometry': {'valid': True, 'origin_xy_m': [0, 0],
                                  'normal_xy': [0, 1], 'tangent_xy': [1, 0],
                                  'tangent_extent_m': [0, 4]}}
        values = distance_pair_metrics(target, wall)
        self.assertEqual(values['plane'], 2.)
        query = 'The chair nearest the wall.'
        parsed = {'Relations': [{'target_category': 'chair',
                                 'anchor_category': 'wall',
                                 'relation': 'nearest',
                                 'metric': 'plane',
                                 'phrase': query}]}
        ranking = distance_rankings(query, parsed, [target], [wall])['rankings'][0]
        self.assertEqual(ranking['metric'], 'plane')
        self.assertEqual(ranking['best_ids'], [1])
        outside = {**target, 'bbox_id': 2, 'bbox_3d': [8, 2, 1, 1, 1, 1]}
        ignored = distance_rankings(query, parsed, [outside], [wall])
        self.assertEqual(ignored['rankings'][0]['comparison_status'], 'unknown')
        self.assertEqual(ignored['rankings'][0]['unknown_target_ids'], [2])
        self.assertEqual(ignored['rankings'][0]['best_ids'], [])
        self.assertIn('no finite target distances', ignored['ignored'][0]['reason'])

    def test_unstated_or_negated_superlative_never_ranked(self):
        chair={'bbox_id':1,'target':'chair','bbox_3d':[0,0,0,1,1,1]}
        table={'bbox_id':2,'target':'table','bbox_3d':[1,0,0,1,1,1]}
        for query,phrase in [('chair by the table','chair by the table'),
                             ('chair by the table','chair nearest the table'),
                             ('chair not nearest the table','not nearest the table')]:
            parsed={'Relations':[{'target_category':'chair','anchor_category':'table',
                                 'relation':'nearest','phrase':phrase}]}
            result=distance_rankings(query,parsed,[chair],[table])
            self.assertEqual(result['rankings'],[])
            self.assertTrue(result['ignored'])
            validated=validate_distance_relations(query,parsed)
            self.assertEqual(validated['Relations'][0]['relation'],'other')
            self.assertEqual(parsed['Relations'][0]['relation'],'nearest')

    def test_proposal_keeps_labels_then_checks_neighbors_with_budget(self):
        objects=[{'bbox_id':10,'target':'tv','bbox_3d':[0,0,0,1,1,1]},
                 {'bbox_id':50,'target':'water cooler','bbox_3d':[1,0,0,1,1,1]},
                 {'bbox_id':60,'target':'chair','bbox_3d':[3,0,0,1,1,1]},
                 {'bbox_id':70,'target':'trash can','bbox_3d':[6,0,0,1,1,1]}]
        proposed=propose_category_checks(objects,[objects[3]],[objects[0]],2)
        self.assertEqual([o['bbox_id'] for o in proposed],[70,50])
        self.assertEqual(objects[1]['target'],'water cooler')


if __name__=='__main__':unittest.main()
