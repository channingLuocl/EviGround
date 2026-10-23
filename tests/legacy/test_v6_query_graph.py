"""Offline query obligations, reference alternatives and candidate recall."""
import json
import os
import unittest
from unittest.mock import patch

from eviground.grounding.diagnostic import diagnose, size_condition_scope, size_conditions, validate_structure
from eviground.query.query_graph import local_graph
from eviground.grounding.react_decision import planner_state
from eviground.grounding.v4_policy import settings
from eviground.geometry.v4_relations import bind_tasks


def parsed(query, attributes=(), relations=(), unhandled=()):
    return validate_structure(query, {'Target': ['chair'], 'Anchor': ['table'],
        'Attributes': list(attributes), 'Relations': list(relations),
        'Observer': {}, 'Unhandled': list(unhandled)})


def near(phrase):
    return {'target_category': 'chair', 'anchor_category': 'table', 'relation': 'near', 'phrase': phrase}


def obj(oid, category, x):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [x, 0, .5, 1, 1, 1]}


class V6QueryGraphTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V5': '1',
            'EVIGROUND_V6': '1', 'EVIGROUND_SYNTAX_BACKEND': 'spacy'})
        env.start()
        self.addCleanup(env.stop)

    def test_opt_in_and_v5_baseline(self):
        self.assertEqual(settings()['version'], 'relative-size-v6')
        with patch.dict(os.environ, {'EVIGROUND_V6': '0'}):
            self.assertEqual(settings()['version'], 'relative-size-v5')

    def test_negative_relation_keeps_positive_size_and_reference_description(self):
        query = 'the smaller chair not near the white table'
        p = parsed(query, [{'category': 'chair', 'phrase': 'smaller'},
                           {'category': 'table', 'role': 'reference', 'phrase': 'white'}],
                   [near('near the white table')])
        self.assertEqual(p['Relations'], [])
        self.assertEqual([a['phrase'] for a in p['Attributes']], ['smaller', 'white'])
        conditions = p['query_structure']['conditions']
        self.assertEqual(next(c for c in conditions if c.get('source_quote') == 'white')['polarity'], 'positive')
        self.assertTrue(any(c['polarity'] == 'exclude' for c in conditions))

    def test_independent_positive_relation_survives_exclusion(self):
        p = parsed('the chair near the table, not the white one',
                   [{'category': 'chair', 'phrase': 'white'}], [near('near the table')])
        self.assertEqual(len(p['Relations']), 1)
        self.assertEqual(p['Attributes'], [])
        self.assertTrue(any(c['source_quote'] == 'white' and c['parse_status'] == 'unknown'
                            for c in p['query_structure']['conditions']))

    def test_negative_size_never_executes_as_positive_or_removes_candidates(self):
        p = parsed('the chair that is not smaller', [{'category': 'chair', 'phrase': 'smaller'}])
        self.assertEqual(size_conditions(p, 'chair'), [])
        objects = [obj(1, 'chair', 0), obj(2, 'chair', 3)]
        d = diagnose(p, objects, objects, [0, 0, 0], trusted_categories=True)
        self.assertEqual(d['candidate_ids'], [1, 2])
        self.assertNotEqual(d['status'], 'resolved')

    def test_repeated_quote_and_disabled_syntax_keep_unknown(self):
        for query in ('the white chair near the white table', 'the chair not near the table'):
            with patch.dict(os.environ, {'EVIGROUND_SYNTAX_BACKEND': 'none'}):
                p = parsed(query, [{'category': 'chair', 'phrase': 'white'}], [near('near the table')])
            if 'not' in query:
                self.assertEqual(p['Relations'], [])
            self.assertTrue(any(c['parse_status'] == 'unknown' for c in p['query_structure']['conditions']))

    def test_unrepresented_modifier_has_unknown_obligation(self):
        p = parsed('the striped chair near the table', relations=[near('near the table')])
        self.assertIn('striped', p['query_structure']['uncovered_modifiers'])
        self.assertIn('query modifier not represented: striped', p['diagnostic_missing'])

    def test_reference_description_and_all_instances_retained(self):
        query = 'the chair near the white table'
        p = parsed(query, [{'category': 'table', 'role': 'reference', 'phrase': 'white'}],
                   [near('near the white table')])
        chairs = [obj(1, 'chair', 0), obj(2, 'chair', 3)]
        objects = chairs + [obj(3, 'table', 0), obj(4, 'table', 4), obj(5, 'lamp', 20)]
        graph = local_graph(p, objects, chairs, {'candidate_ids': [1, 2]})
        binding = next(b for b in graph['reference_bindings'] if b['category'] == 'table')
        self.assertEqual(binding['candidate_ids'], [3, 4])
        self.assertTrue(binding['description_condition_ids'])
        self.assertEqual({n['bbox_id'] for n in graph['nodes']}, {1, 2, 3, 4})
        row = bind_tasks(query, p, objects, chairs)[0]['geometry_evidence']
        self.assertEqual(row['anchor_ids'], [3, 4])
        self.assertEqual(row['status'], 'unknown')

    def test_counted_size_scope_is_not_redefined_by_live_ids_or_graph(self):
        query = 'the smaller of the two chairs near the table'
        p = parsed(query, [{'category': 'chair', 'phrase': 'smaller'}], [near('near the table')])
        chairs = [obj(i, 'chair', i) for i in (1, 2, 3)]
        objects = chairs + [obj(4, 'table', 0)]
        graph = local_graph(p, objects, chairs, {'candidate_ids': [1]})
        self.assertEqual(graph['initial_target_ids'], [1, 2, 3])
        spec = size_conditions(p, 'chair')[0]
        self.assertEqual(spec['required_member_count'], 2)
        _, binding = size_condition_scope(p, objects, 'chair', spec)
        self.assertEqual(binding['status'], 'unknown')

    def test_planner_summary_preserves_unknown_quotes_and_bindings(self):
        p = parsed('the chair near the table, not the white one', relations=[near('near the table')])
        objects = [obj(1, 'chair', 0), obj(2, 'table', 0), obj(3, 'table', 4)]
        graph = local_graph(p, objects, objects[:1], {'candidate_ids': [1]})
        compact, audit = planner_state({'parsed_query': p, 'query_graph': graph, 'history': [],
                                        'observation': {}, 'evidence_state': {'missing': []}})
        payload = json.dumps(compact)
        self.assertIn('not the white one', payload)
        self.assertEqual(compact['query_graph']['reference_bindings'], graph['reference_bindings'])
        self.assertLess(audit['input_units'], 32768 - 2048)


if __name__ == '__main__':
    unittest.main()
