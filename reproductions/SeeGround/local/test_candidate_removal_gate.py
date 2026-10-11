import unittest

from active_diagnostic import apply_computed_evidence
from diagnostic import candidate_removal_gate, diagnose, validate_structure


class CandidateRemovalGateTests(unittest.TestCase):
    def test_unknown_never_removes_and_records_why(self):
        row = candidate_removal_gate(
            {'candidate_id': 7, 'status': 'unknown', 'reason': 'scope is unresolved'},
            source_quote='small book', target_role='target',
            role_binding_status='unknown', comparison_scope_ids=[7, 8],
            measurement_source='box measurements',
            explicit_fail_criterion='would fail if scope were bound')
        self.assertFalse(row['removal_allowed'])
        self.assertIn('not an explicit fail', row['removal_blocked_reason'])
        self.assertEqual(row['source_quote'], 'small book')
        self.assertEqual(row['comparison_scope_ids'], [7, 8])

    def test_fail_requires_bound_role_scope_source_and_criterion(self):
        incomplete = candidate_removal_gate(
            {'candidate_id': 7, 'status': 'fail'},
            source_quote='largest shelf', target_role='target',
            role_binding_status='bound', comparison_scope_ids=[],
            measurement_source='annotated_box')
        self.assertFalse(incomplete['removal_allowed'])

        complete = candidate_removal_gate(
            {'candidate_id': 7, 'status': 'fail'},
            source_quote='largest shelf', target_role='target',
            role_binding_status='bound', comparison_scope_ids=[7, 8],
            comparison_scope_status='bound',
            measurement_source='annotated_box',
            explicit_fail_criterion='reliable measurement contradicts largest')
        self.assertTrue(complete['removal_allowed'])
        self.assertEqual(complete['target_role'], 'target')

    def test_reference_fail_cannot_remove_target_in_shared_update(self):
        diagnosis = {'candidate_ids': [7, 8], 'stable_candidate_ids': []}
        update = apply_computed_evidence(diagnosis, [{
            'candidate_id': 7, 'status': 'fail', 'source_quote': 'largest shelf',
            'target_role': 'reference', 'reference_role': 'reference',
            'role_binding_status': 'bound', 'comparison_scope_status': 'bound',
            'comparison_scope_ids': [7, 8], 'measurement_source': 'annotated_box',
            'explicit_fail_criterion': 'reference 7 volume is below reference 8',
        }], 'size')
        self.assertEqual(update['removed_ids'], [])
        self.assertEqual(diagnosis['candidate_ids'], [7, 8])

    def test_active_update_uses_gate_instead_of_status_only(self):
        diagnosis = {'candidate_ids': [1, 2], 'stable_candidate_ids': []}
        result = apply_computed_evidence(
            diagnosis,
            [
                {'candidate_id': 1, 'status': 'fail', 'reason': 'unknown role'},
                {'candidate_id': 2, 'status': 'unknown', 'reason': 'missing scope'},
            ],
            'size')
        self.assertEqual(result['removed_ids'], [])
        self.assertEqual(diagnosis['candidate_ids'], [1, 2])
        audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit']
        self.assertEqual(len(audit), 2)
        self.assertTrue(all(not item['removal_allowed'] for item in audit))

    def test_duplicate_candidate_proofs_are_conflict_and_never_remove(self):
        diagnosis={'candidate_ids':[1,2],'stable_candidate_ids':[]}
        proof={'candidate_id':1,'status':'fail','source_quote':'largest shelf',
               'target_role':'target','role_binding_status':'bound',
               'comparison_scope_status':'bound','comparison_scope_ids':[1,2],
               'measurement_source':'annotated_box','explicit_fail_criterion':'volume 1 < volume 2'}
        update=apply_computed_evidence(diagnosis,[{**proof,'status':'unknown'},proof],'size')
        self.assertEqual(update['removed_ids'],[])
        self.assertEqual(diagnosis['candidate_ids'],[1,2])
        self.assertTrue(all(item['status']=='unknown' for item in
                            diagnosis['active_computed_evidence'][0]['candidate_removal_audit']))

    def test_empty_size_category_pool_keeps_candidate(self):
        parsed = validate_structure(
            'the smallest book',
            {'Target': ['book'], 'Anchor': [], 'Relations': [],
             'Attributes': [{'category': 'book', 'role': 'target', 'phrase': 'smallest'}],
             'Observer': {}, 'Unhandled': []})
        objects = [{'bbox_id': 1, 'target': 'table', 'bbox_3d': [0, 0, 0, 1, 1, 1]}]
        result = diagnose(parsed, objects, objects, [0, 0, 0], trusted_categories=False)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertTrue(any('scope' in reason
                            for reason in result['missing']))

    def test_relation_replay_cannot_bypass_unfinished_reference_or_category_binding(self):
        proof = {'candidate_id': 1, 'status': 'fail', 'source_quote': 'above the largest shelf',
                 'target_role': 'target', 'role_binding_status': 'bound',
                 'comparison_scope_status': 'bound', 'comparison_scope_ids': [1, 2, 7],
                 'measurement_source': 'annotated_box geometry',
                 'explicit_fail_criterion': 'candidate 1 is below reference 7'}
        for dimension in ('position', 'distance'):
            for blocker in (
                    {'pending_size_conditions': [{'role': 'reference', 'attribute': 'largest'}]},
                    {'geometry': {'size_constraints': [{'role': 'reference',
                        'candidate_evidence': [{'candidate_id': 7, 'status': 'unknown'}]}]}},
                    {'visual_needs': [{'id': 7, 'role': 'reference'}]},
                    {'geometry': {'target_role_binding_status': 'unknown'}},
                    {'geometry': {'reference_binding_status': 'unknown'}},
                    {'geometry': {'target_instance_binding_status': 'unknown'}}):
                with self.subTest(dimension=dimension, blocker=blocker):
                    diagnosis = {'candidate_ids': [1, 2], 'stable_candidate_ids': [], **blocker}
                    update = apply_computed_evidence(diagnosis, [proof], dimension)
                    self.assertEqual(update['removed_ids'], [])
                    self.assertEqual(update['after'], [1, 2])
                    audit = diagnosis['active_computed_evidence'][0]['candidate_removal_audit'][0]
                    self.assertFalse(audit['removal_allowed'])
                    self.assertEqual(audit['status'], 'unknown')
                    self.assertTrue(audit['binding_missing'])
                    self.assertEqual(diagnosis['status'], 'missing')
                    singleton = {'candidate_ids': [1], 'stable_candidate_ids': [1], **blocker}
                    apply_computed_evidence(singleton, [proof], dimension)
                    self.assertEqual(singleton['candidate_ids'], [1])
                    self.assertEqual(singleton['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
