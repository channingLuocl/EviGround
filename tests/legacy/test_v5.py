"""Behavior checks for v5; all network behavior is mocked."""
from eviground.paths import ROOT
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eviground.grounding.v4_policy import backtrack_events, settings


class V5Tests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V5': '1',
            'EVIGROUND_API_PROVIDER': 'qwen', 'EVIGROUND_REACT_BACKEND': 'qwen'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_version_and_opt_out(self):
        self.assertEqual(settings()['version'], 'relative-size-v5')
        with patch.dict(os.environ, {'EVIGROUND_V5': '0'}):
            self.assertEqual(settings()['version'], 'relative-size-v4')

    def test_supporting_deleted_alternative_does_not_backtrack(self):
        event = {'event_id': 0, 'candidate_id': 18, 'source_quotes': ['round'], 'invalidated': False}
        problem = {'supported': True, 'trigger': 'final_review', 'candidate_ids': [18],
            'source_quotes': ['round'], 'findings': [{'finding': 'confirms_exclusion',
                'candidate_id': 18, 'event_id': 0, 'source_quote': 'round', 'validated': True}]}
        self.assertEqual(backtrack_events([event], problem), [])
        problem['findings'][0]['finding'] = 'refutes_exclusion'
        self.assertEqual(backtrack_events([event], problem), [event])
        self.assertEqual(backtrack_events([event], problem, used=True), [])

    def test_old_undirected_report_cannot_restore(self):
        self.assertEqual(backtrack_events([{'event_id': 0, 'candidate_id': 8,
            'source_quotes': ['on the floor'], 'invalidated': False}],
            {'supported': True, 'trigger': 'final_review', 'candidate_ids': [8],
             'source_quotes': ['on the floor']}), [])

    def test_floor_outliers_do_not_destroy_tabletop_measurement(self):
        from eviground.geometry.object_geometry import tabletop_measurement
        rng = np.random.default_rng(13)
        floor = np.column_stack((rng.uniform(-.6, .6, (500, 2)), rng.normal(0, .002, 500)))
        floor[:60, 2] = rng.uniform(.15, .35, 60)
        top = np.column_stack((rng.uniform(-.5, .5, (500, 2)), rng.normal(.75, .002, 500)))
        obj = {'bbox_id': 1, 'target': 'desk', 'bbox_3d': [0, 0, .4, 1, 1, .8]}
        result = tabletop_measurement(obj, top, [{'bbox_id': 2, 'target': 'floor'}],
            floor, np.full(len(floor), 2))
        self.assertAlmostEqual(result['tabletop_height_m'], .75, delta=.015)
        self.assertGreater(result['tabletop_evidence']['floor_plane']['outlier_fraction'], .05)

    def test_front_area_does_not_depend_on_screen_projection(self):
        from eviground.geometry.object_geometry import _oriented_measurements
        x, z = np.meshgrid(np.linspace(-.4, .4, 30), np.linspace(0, 1.2, 30))
        points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        metrics, quality, details = _oriented_measurements({'target': 'mirror'}, points)
        self.assertAlmostEqual(metrics['body_plane_area_m2'], .96, delta=.01)
        self.assertEqual(quality['body_plane_area_m2']['status'], 'usable')

    def test_compact_state_keeps_all_conditions_and_unknown_members(self):
        from eviground.grounding.react_decision import planner_state
        state = {'query': 'small chair near the door', 'parsed_query': {'Target': ['chair'],
            'Relations': [{'relation': 'near', 'phrase': 'near the door'}], 'Unhandled': ['unsupported'],
            'condition_records': [{'source_quote': 'small'}, {'source_quote': 'near the door'}]},
            'remaining_candidate_ids': [1, 2], 'observation': {'details': {
                'size_constraints': [{'unknown_measurement_ids': [2], 'reason': 'unmeasured'}],
                'comparison_metrics': [{'id': 1, 'junk': 'x' * 100000}]}},
            'history': [{'observation': {'details': {'junk': 'x' * 100000}}}] * 10}
        compact, audit = planner_state(state)
        rendered = json.dumps(compact)
        self.assertIn('near the door', rendered)
        self.assertIn('unknown_measurement_ids', rendered)
        self.assertLess(len(rendered), 5000)
        self.assertEqual(audit['method'], 'utf8_bytes_upper_bound')

    def test_batch_progress_only_emits_two_result_fields(self):
        from io import StringIO
        from eviground.evaluation.batch_progress import BatchProgress
        stdout, stderr = StringIO(), StringIO()
        progress = BatchProgress(2, stdout=stdout, stderr=stderr)
        progress.update({'predicted_id': 3, 'correct': True, 'target_id': 3, 'missing': ['x']})
        progress.update({'predicted_id': None, 'correct': False, 'error': 'x'})
        self.assertEqual([set(json.loads(line)) for line in stdout.getvalue().splitlines()],
                         [{'predicted_id', 'correct'}] * 2)
        self.assertIn('2/2', stderr.getvalue())

    def test_real_saved_confirmations_never_restore(self):
        base = Path('/Volumes/KESU/论文/.worktrees/relative-size-evidence-v4/reproductions/SeeGround/local_outputs/v4_full300')
        for row in (14770, 18106):
            path = base / f'q{row:05d}/prediction.json'
            if not path.exists():
                self.skipTest('saved v4 review unavailable')
            trace = json.loads(path.read_text())['active_trace']
            backtrack = trace['backtracks'][0]
            self.assertEqual(backtrack_events(trace['deletion_provenance'], backtrack['problem']), [])

    def test_directional_report_requires_source_bound_visible_proof(self):
        from eviground.rendering.v4_visual import validate_findings
        view = {'focus_ids': [1, 2], 'visible_instance_pixels': {'1': 100, '2': 100}}
        payload = {'selected_id': 1, 'deletions': [{'event_id': 0, 'candidate_id': 2,
                    'source_quotes': ['round'], 'invalidated': False}]}
        finding = {'finding': 'refutes_exclusion', 'candidate_id': 2, 'event_id': 0,
            'source_quote': 'round', 'target_role': 'target', 'predicate_status': 'pass',
            'evidence_type': 'visible_geometry', 'reason': 'curved perimeter', 'observation_ids': [2]}
        report = {'findings': [finding], 'observations': [{'id': 2, 'reason': 'curved perimeter',
            'condition_checks': [{'source_quote': 'round', 'status': 'pass', 'reason': 'continuous curved perimeter'}]}]}
        checked = validate_findings(report, 'round table', view, payload)
        self.assertEqual(len(checked), 1)
        report['observations'][0]['condition_checks'] = []
        self.assertEqual(validate_findings(report, 'round table', view, payload), [])

    def test_ambiguous_attachment_computes_full_scope_without_deleting_unknown_subset(self):
        from eviground.grounding.active_diagnostic import size_candidate_evidence
        from eviground.grounding.diagnostic import validate_structure
        from eviground.geometry.object_geometry import point_cloud_measurements
        objects = [{'bbox_id': i, 'target': 'shelf', 'bbox_3d': [3*i, 0, .5, extent, extent, 1]}
                   for i, extent in [(1, 1), (2, .4)]]
        rng = np.random.default_rng(9)
        xyz = np.vstack([rng.uniform(-.5, .5, (1000, 3)) * o['bbox_3d'][3:] + o['bbox_3d'][:3] for o in objects])
        ids = np.repeat([1, 2], 1000)
        parsed = validate_structure('the largest shelf by the door', {'Target': ['shelf'],
            'Anchor': ['door'], 'Relations': [{'target_category': 'shelf', 'anchor_category': 'door',
                'relation': 'near', 'phrase': 'by the door'}], 'Attributes': [{'category': 'shelf', 'phrase': 'largest'}],
            'Observer': {}, 'Unhandled': []})
        records, constraints, decision = size_candidate_evidence(parsed, objects,
            point_cloud_measurements(objects, xyz, ids), objects=objects)
        self.assertEqual(len(constraints[0]['scope_hypotheses']), 2)
        self.assertEqual(constraints[0]['scope_hypotheses'][0]['scope_status'], 'bound')
        self.assertEqual(constraints[0]['scope_hypotheses'][1]['scope_status'], 'unknown')
        self.assertFalse(any(r['status'] == 'fail' for r in records))

    def test_temporary_retry_budget_is_shared_across_calls(self):
        import requests
        from types import SimpleNamespace
        from eviground.api.qwen_api import chat, QwenAPIError
        from eviground.grounding.v5_runtime import sample_recovery
        response = SimpleNamespace(ok=True, json=lambda: {'choices': [{'finish_reason': 'stop',
            'message': {'content': '{}'}}]})
        token = sample_recovery.set({'used': False, 'attempts': []})
        try:
            with patch.dict(os.environ, {'EVIGROUND_MODEL': 'mock', 'EVIGROUND_BASE_URL': 'https://example.invalid/v1',
                        'EVIGROUND_API_KEY': 'test-only'}), patch('eviground.api.qwen_api.requests.post',
                    side_effect=[requests.Timeout(), response, requests.Timeout()]) as post:
                result = chat([], json_mode=True)
                self.assertEqual(result['_baseline']['extra_attempts'], 1)
                with self.assertRaises(QwenAPIError):
                    chat([], json_mode=True)
                self.assertEqual(post.call_count, 3)
        finally:
            sample_recovery.reset(token)

    def test_review_failure_preserves_prediction_and_resumes_only_review(self):
        from eviground.grounding.real_scene import run_query
        from eviground.api.qwen_api import QwenAPIError
        from tests.legacy.test_real_scene import RealSceneTests, reply
        parsed = {'Target': ['water cooler'], 'Anchor': [], 'Relations': [],
                  'Observer': {}, 'Attributes': [], 'Unhandled': []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RealSceneTests().scene(root)
            (root / 'objects_gt.json').write_text((root / 'objects.json').read_text())
            diagnosis = {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'resolved', 'missing': [],
                'geometry': {}, 'pending_size_conditions': [], 'hypotheses': [{'candidate_ids': [1]}]}
            failure = QwenAPIError('mock 502', diagnostic={'failure_kind': 'http_5xx', 'http_status': 502})
            with patch('eviground.grounding.real_scene.chat', return_value=reply(json.dumps(parsed))), patch('eviground.grounding.active_diagnostic.diagnose', return_value=diagnosis), \
                 patch('eviground.grounding.active_diagnostic.chat', side_effect=[reply('{"action":"finish","predicted_id":1,"reason":"verified"}'), failure]):
                result = run_query(root, 'water cooler', root / 'out', mode='active_diagnostic', object_table='gt')
            self.assertEqual(result['predicted_id'], 1)
            self.assertTrue(result['review_failed'])
            self.assertTrue((root / 'out/prediction.json').exists())
            with patch('eviground.grounding.real_scene.chat') as parser, patch('eviground.grounding.active_diagnostic.diagnose', return_value=diagnosis), \
                 patch('eviground.grounding.active_diagnostic.chat', return_value=reply('{"status":"consistent","reason":"visible"}')) as visual:
                resumed = run_query(root, 'water cooler', root / 'out', mode='active_diagnostic', object_table='gt')
            parser.assert_not_called()
            self.assertEqual(visual.call_count, 1)
            self.assertEqual(resumed['predicted_id'], 1)
            self.assertFalse(resumed['review_failed'])

    def test_inference_stage_resume_reuses_parser_and_successful_decisions(self):
        from eviground.grounding.real_scene import run_query
        from eviground.api.qwen_api import QwenAPIError
        from tests.legacy.test_real_scene import RealSceneTests, reply
        parsed = {'Target': ['water cooler'], 'Anchor': [], 'Relations': [],
                  'Observer': {}, 'Attributes': [], 'Unhandled': []}
        diagnosis = {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'resolved', 'missing': [],
            'geometry': {}, 'pending_size_conditions': [], 'hypotheses': [{'candidate_ids': [1]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RealSceneTests().scene(root)
            (root / 'objects_gt.json').write_text((root / 'objects.json').read_text())
            failure = QwenAPIError('mock timeout', diagnostic={'failure_kind': 'timeout'})
            with patch('eviground.grounding.real_scene.chat', return_value=reply(json.dumps(parsed))), \
                 patch('eviground.grounding.active_diagnostic.diagnose', return_value=diagnosis), \
                 patch('eviground.grounding.active_diagnostic.chat', side_effect=failure):
                with self.assertRaises(QwenAPIError):
                    run_query(root, 'water cooler', root / 'out', mode='active_diagnostic', object_table='gt')
            checkpoint = json.loads((root / 'out/pending_inference.json').read_text())
            self.assertEqual(checkpoint['stage'], 'planner')
            with patch('eviground.grounding.real_scene.chat') as parser, patch('eviground.grounding.active_diagnostic.diagnose', return_value=diagnosis), \
                 patch('eviground.grounding.active_diagnostic.chat', side_effect=[
                    reply('{"action":"finish","predicted_id":1,"reason":"verified"}'),
                    reply('{"status":"consistent","reason":"visible"}')]):
                result = run_query(root, 'water cooler', root / 'out', mode='active_diagnostic', object_table='gt')
            parser.assert_not_called()
            self.assertEqual(result['predicted_id'], 1)
            self.assertFalse((root / 'out/pending_inference.json').exists())

    def test_no_retry_for_parameter_error_and_two_attempts_for_temporary_failure(self):
        from types import SimpleNamespace
        from eviground.api.qwen_api import chat, QwenAPIError
        from eviground.grounding.v5_runtime import sample_recovery
        for status, expected in [(400, 1), (401, 1), (502, 2)]:
            token = sample_recovery.set({'used': False, 'attempts': []})
            try:
                response = SimpleNamespace(ok=False, status_code=status, text='invalid parameter', headers={})
                with patch.dict(os.environ, {'EVIGROUND_MODEL': 'mock', 'EVIGROUND_BASE_URL': 'https://example.invalid/v1',
                            'EVIGROUND_API_KEY': 'test-only'}), patch('eviground.api.qwen_api.requests.post', return_value=response) as post:
                    with self.assertRaises(QwenAPIError):
                        chat([], json_mode=True)
                    self.assertEqual(post.call_count, expected)
            finally:
                sample_recovery.reset(token)

    def test_persisted_request_cache_never_reissues_completed_api(self):
        from types import SimpleNamespace
        from eviground.api.qwen_api import chat
        from eviground.grounding.v5_runtime import sample_recovery, save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'api_recovery.json')
            budget = {'used': False, 'attempts': [], 'completed_requests': {},
                      'checkpoint_identity': 'fixed-input', 'api_checkpoint': path}
            token = sample_recovery.set(budget)
            try:
                response = SimpleNamespace(ok=True, json=lambda: {'choices': [{'finish_reason': 'stop',
                    'message': {'content': '{}'}}]})
                with patch.dict(os.environ, {'EVIGROUND_MODEL': 'mock', 'EVIGROUND_BASE_URL': 'https://example.invalid/v1',
                            'EVIGROUND_API_KEY': 'test-only'}), patch('eviground.api.qwen_api.requests.post', return_value=response) as post:
                    chat([], json_mode=True)
                    budget['resuming'] = True
                    replay = chat([], json_mode=True)
                    self.assertTrue(replay['_baseline']['cache_hit'])
                    self.assertEqual(post.call_count, 1)
            finally:
                sample_recovery.reset(token)

    def test_supported_floor_levels_remain_unknown_and_floor_absence_never_uses_min_z(self):
        from eviground.geometry.object_geometry import horizontal_plane, tabletop_measurement
        x, y = np.meshgrid(np.linspace(-.5, .5, 15), np.linspace(-.5, .5, 15))
        floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        second = floor + [0, 0, .2]
        self.assertEqual(horizontal_plane(np.vstack((floor, second)))['status'], 'unknown')
        table = {'bbox_id': 1, 'target': 'table', 'bbox_3d': [0, 0, .75, 1, 1, .1]}
        top = floor + [0, 0, .75]
        measured = tabletop_measurement(table, top, [table], top, np.ones(len(top), int))
        self.assertIsNone(measured['tabletop_height_m'])

    def test_support_phrase_preserves_reference_on_target_direction(self):
        from eviground.geometry.v4_relations import relation_tasks, geometric_relation
        query = 'the smaller chair with a backpack on it'
        parsed = {'Target': ['chair'], 'Anchor': ['backpack'], 'Unhandled': ['with a backpack on it']}
        task = relation_tasks(query, parsed)[0]
        self.assertEqual(task['support_subject'], 'reference')
        chair = {'bbox_id': 1, 'bbox_3d': [0, 0, .5, 1, 1, 1]}
        bag = {'bbox_id': 2, 'bbox_3d': [0, 0, 1.1, .3, .3, .2]}
        self.assertEqual(geometric_relation(task, chair, [bag])['status'], 'pass')
        task.pop('support_subject')
        self.assertEqual(geometric_relation(task, chair, [bag])['status'], 'fail')

    def test_regular_diagnosis_has_both_scope_hypotheses(self):
        from eviground.grounding.diagnostic import diagnose, validate_structure
        objects = [{'bbox_id': i, 'target': 'shelf', 'bbox_3d': [i, 0, 1, i, 1, 1]} for i in (1, 2)]
        query = 'the largest shelf by the door'
        parsed = validate_structure(query, {'Target': ['shelf'], 'Anchor': ['door'],
            'Relations': [{'target_category': 'shelf', 'anchor_category': 'door', 'relation': 'near', 'phrase': 'by the door'}],
            'Attributes': [{'category': 'shelf', 'phrase': 'largest'}], 'Observer': {}, 'Unhandled': []})
        diagnosis = diagnose(parsed, objects, objects, np.zeros(3), trusted_categories=True,
                             apply_size_constraints=False, query=query)
        constraint = diagnosis['geometry']['size_constraints'][0]
        self.assertEqual(len(constraint['scope_hypotheses']), 2)
        self.assertEqual(constraint['scope_hypotheses'][0]['scope_status'], 'bound')

    def test_saved_review_confirmation_keeps_active_deletion_and_unused_opportunity(self):
        from copy import deepcopy
        base = Path('/Volumes/KESU/论文/.worktrees/relative-size-evidence-v4/reproductions/SeeGround/local_outputs/v4_full300')
        for row in (14770, 18106):
            saved = json.loads((base / f'q{row:05d}/prediction.json').read_text())
            trace = saved['active_trace']
            old = trace['backtracks'][0]['problem']
            events = deepcopy(trace['deletion_provenance'])
            for event in events:
                event['invalidated'] = False
            finding = old['evidence'][0]
            event = next(e for e in events if e['candidate_id'] in finding['candidate_ids']
                         and finding['source_quote'] in e['source_quotes'])
            problem = {'supported': True, 'trigger': 'final_review', 'findings': [{
                'finding': 'confirms_exclusion', 'candidate_id': event['candidate_id'],
                'event_id': event['event_id'], 'source_quote': finding['source_quote'],
                'reason': finding['reason'], 'validated': True}]}
            self.assertEqual(backtrack_events(events, problem, used=False), [])
            self.assertFalse(event['invalidated'])

    def test_real_context20637_preserves_condition_quotes_under_budget(self):
        from eviground.grounding.active_diagnostic import evidence_state
        from eviground.grounding.react_decision import planner_state
        path = Path('/Volumes/KESU/论文/.worktrees/relative-size-evidence-v4/reproductions/SeeGround/local_outputs/v4_full300/q20637/run.json')
        if not path.exists():
            self.skipTest('saved context fixture unavailable')
        saved = json.loads(path.read_text())
        trace = saved['active_diagnostic']
        diagnosis = trace['diagnosis_after_step']
        compact, audit = planner_state({'query': saved['query'], 'parsed_query': saved['parsed_query'],
            'history': trace['steps'], 'observation': trace['steps'][-1]['observation'],
            'evidence_state': evidence_state(diagnosis), 'deletion_events': trace['deletion_provenance']},
            instruction_bytes=5000)
        self.assertLess(audit['input_units'], 32768 - 2048)
        self.assertEqual(compact['evidence_state']['missing'], diagnosis['missing'])
        self.assertEqual(compact['parsed_query']['condition_records'], saved['parsed_query']['condition_records'])
        current = {oid for p in compact['observation']['candidate_evidence'] for oid in p['candidate_ids']}
        prior = {p['candidate_id'] for p in trace['steps'][-1]['observation']['candidate_evidence']}
        self.assertEqual(current, prior)

    def test_context_error_retries_only_after_state_changes(self):
        from types import SimpleNamespace
        from eviground.api.qwen_api import chat, QwenAPIError
        from eviground.grounding.v5_runtime import sample_recovery
        failure = SimpleNamespace(ok=False, status_code=400, text='maximum context length exceeded', headers={})
        success = SimpleNamespace(ok=True, json=lambda: {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({'evidence_state': {},
            'history': [{'action': {'action': 'evidence'}, 'remaining_candidate_ids': [1]}] * 3})}]}]
        token = sample_recovery.set({'used': False, 'attempts': []})
        try:
            with patch.dict(os.environ, {'EVIGROUND_MODEL': 'mock', 'EVIGROUND_BASE_URL': 'https://example.invalid/v1',
                        'EVIGROUND_API_KEY': 'test-only'}), patch('eviground.api.qwen_api.requests.post', side_effect=[failure, success]) as post:
                chat(messages, json_mode=True)
                self.assertEqual(post.call_count, 2)
                self.assertNotEqual(post.call_args_list[0].kwargs['json']['messages'],
                                    post.call_args_list[1].kwargs['json']['messages'])
        finally:
            sample_recovery.reset(token)

    def test_jev_and_visual_calls_share_single_recovery(self):
        from eviground.api.decision_api import decide, DecisionAPIError
        from eviground.api.qwen_api import chat, QwenAPIError
        from eviground.grounding.v5_runtime import sample_recovery
        import requests
        token = sample_recovery.set({'used': False, 'attempts': []})
        try:
            good = {'model': 'mock', 'answers': {}, 'validation_warnings': [], 'usage': {}, 'latency_s': 0}
            with patch('eviground.api.decision_api._decide_once', side_effect=[
                DecisionAPIError('timeout', {}, diagnostic={'failure_kind': 'timeout'}), good]) as planner:
                self.assertEqual(len(decide('jev', {}, {})['api_attempts']), 2)
                self.assertEqual(planner.call_count, 2)
            with patch.dict(os.environ, {'EVIGROUND_MODEL': 'mock', 'EVIGROUND_BASE_URL': 'https://example.invalid/v1',
                        'EVIGROUND_API_KEY': 'test-only'}), patch('eviground.api.qwen_api.requests.post', side_effect=requests.Timeout()) as visual:
                with self.assertRaises(QwenAPIError):
                    chat([], json_mode=True)
                self.assertEqual(visual.call_count, 1)
        finally:
            sample_recovery.reset(token)

    def test_selected_target_refutation_revokes_acceptance_without_restoring(self):
        from tests.legacy.test_v4 import V4Tests
        diagnosis = {'candidate_ids': [1], 'stable_candidate_ids': [1], 'status': 'resolved',
                     'missing': [], 'geometry': {}, 'hypotheses': [{'candidate_ids': [1]}], 'pending_size_conditions': []}
        report = {'status': 'contradiction', 'reason': 'selected requirement refuted', 'findings': [{
            'finding': 'contradicts_selected', 'candidate_id': 1, 'source_quote': 'tall',
            'target_role': 'target', 'predicate_status': 'fail', 'evidence_type': 'visible_geometry',
            'observation_ids': [1], 'reason': 'visible shorter structure'}], 'observations': [{
            'id': 1, 'reason': 'visible shorter structure', 'condition_checks': [{
                'source_quote': 'tall', 'status': 'fail', 'reason': 'visible shorter structure'}]}]}
        result = V4Tests().controller([{'action': 'finish', 'predicted_id': 1, 'reason': 'supported'}], [report], diagnosis)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['backtrack_count'], 0)

    def test_true_refutation_restores_once_with_source_event(self):
        from tests.legacy.test_v4 import V4Tests
        report = {'status': 'contradiction', 'reason': 'specific deletion refuted', 'findings': [{
            'finding': 'refutes_exclusion', 'candidate_id': 2, 'event_id': 0, 'source_quote': 'tall',
            'target_role': 'target', 'predicate_status': 'pass', 'evidence_type': 'visible_geometry',
            'observation_ids': [2], 'reason': 'visible standing body'}], 'observations': [{
            'id': 2, 'reason': 'visible standing body', 'condition_checks': [{
                'source_quote': 'tall', 'status': 'pass', 'reason': 'visible standing body'}]}]}
        result = V4Tests().controller([{'action': 'stop', 'reason': 'unknown'},
            {'action': 'stop', 'reason': 'still unknown'}], [report])
        self.assertEqual(result['backtrack_count'], 1)
        self.assertEqual(result['active_trace']['backtracks'][0]['restored_ids'], [2])
        self.assertIsNone(result['predicted_id'])

    def test_identical_visual_request_uses_cached_report_at_exhausted_budget(self):
        from copy import deepcopy
        from eviground.grounding.active_diagnostic import execute_evidence
        from tests.legacy.test_v4 import box, cloud, parsed
        objects = [box(17, 'table'), box(18, 'table', (2, 0, .5), (2, 2, 1))]
        xyz = np.concatenate([cloud(o, surfaces=True) for o in objects])
        ids = np.concatenate([np.full(len(cloud(o, surfaces=True)), o['bbox_id']) for o in objects])
        p = parsed('table', 'smaller')
        diagnosis = {'candidate_ids': [17, 18], 'size_comparison_pool_ids': [17, 18], 'geometry': {}}
        report = {'status': 'unknown', 'reason': 'top only', 'observations': [{'id': 17,
            'completeness': 'unknown', 'reason': 'top only', 'visible_surfaces': ['top']}]}
        calls = []
        with tempfile.TemporaryDirectory() as out, patch('eviground.grounding.active_diagnostic.chat', return_value={
            '_content': json.dumps(report), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}) as visual:
            request = {'action': 'evidence', 'dimension': 'size', 'reason': 'measure', 'inspect': True}
            execute_evidence(request, 'the smaller table', p, diagnosis, objects, objects, xyz,
                np.full(xyz.shape, 100, dtype=np.uint8), ids, Path(out), 1, calls)
            p['_v4_budget']['size'] = 2
            result = execute_evidence(request, 'the smaller table', p, diagnosis, objects, objects, xyz,
                np.full(xyz.shape, 100, dtype=np.uint8), ids, Path(out), 2, calls)
            self.assertEqual(visual.call_count, 1)
            self.assertTrue(result['details']['visual_inspection']['cache_hit'])

    def test_live_failed_context_fixtures_keep_obligations_and_fit_budget(self):
        from eviground.grounding.active_diagnostic import evidence_state, INSTRUCTIONS
        from eviground.grounding.react_decision import planner_state
        for row in (3157, 4545, 5902, 10211):
            path = ROOT / 'local_outputs/v5_development_68_r2' / f'q{row:05d}/pending_inference.json'
            if not path.exists():
                path = ROOT / 'local_outputs/v5_development_68' / f'q{row:05d}/pending_inference.json'
            if not path.exists():
                self.skipTest('live context fixture unavailable')
            saved = json.loads(path.read_text())
            diagnosis = saved['diagnosis']
            state = {'query': saved['parsed_query']['source_query'], 'parsed_query': saved['parsed_query'],
                'observation': saved['runtime']['observation'], 'history': saved['active_trace']['steps'],
                'evidence_state': evidence_state(diagnosis),
                'deletion_events': saved['active_trace']['deletion_provenance']}
            compact, audit = planner_state(state, instruction_bytes=len(INSTRUCTIONS.encode()) + 2000)
            self.assertLess(audit['input_units'], 32768 - 2048)
            self.assertEqual(compact['parsed_query']['condition_records'], saved['parsed_query']['condition_records'])
            self.assertEqual(compact['evidence_state']['missing'], diagnosis['missing'])
            prior = saved['runtime']['observation'].get('candidate_evidence', [])
            expected = {p['candidate_id'] for p in prior}
            observed = {oid for p in compact['observation'].get('candidate_evidence', []) for oid in p['candidate_ids']}
            self.assertEqual(expected, observed)


if __name__ == '__main__':
    unittest.main()
