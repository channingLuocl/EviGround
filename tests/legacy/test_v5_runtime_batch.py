"""Lossless controller summaries and batch recovery; no live requests."""
from eviground.paths import ROOT
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from eviground.grounding.react_decision import planner_state
from eviground.experiments.run_size_validation import main, run_case


class V5RuntimeBatchTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V5': '1', 'EVIGROUND_V4': '1',
            'EVIGROUND_REACT_BACKEND': 'qwen', 'EVIGROUND_API_PROVIDER': 'qwen',
            'EVIGROUND_CONTEXT_LIMIT': '32768', 'EVIGROUND_MODEL': 'offline-mock',
            'EVIGROUND_BASE_URL': 'https://example.invalid/v1', 'EVIGROUND_API_KEY': 'offline-only'})
        env.start()
        self.addCleanup(env.stop)

    def row(self, index=10):
        return {'csv_row': index, 'scene': 'scene0010_00', 'query': 'small chair near the door',
                'target_id': 1, 'stimulus_candidate_ids': [1, 2]}

    def test_summary_keeps_negative_relations_roles_and_unknown_members(self):
        proof = {'candidate_id': 2, 'status': 'fail', 'source_quote': 'near the door',
            'reference_ids': [8, 9], 'exclusion_reason': 'fails under both door bindings',
            'relation_evidence': [{'status': 'fail', 'anchor_id': 8, 'reason': 'outside support band'}]}
        source = {'query': self.row()['query'], 'parsed_query': {'condition_records': [
            {'source_quote': 'small'}, {'source_quote': 'near the door'}],
            '_v4_relation_evidence': [{**proof, 'unknown_member_ids': [3]}]},
            'evidence_state': {'missing': ['unknown member 3']},
            'observation': {'candidate_evidence': [proof, {**proof, 'candidate_id': 4}],
                'details': {'candidate_evidence': [proof, {**proof, 'candidate_id': 4}]}},
            'history': [{'observation': {'history': ['audit-only'] * 1000}}] * 100}
        compact, audit = planner_state(source)
        grouped = compact['observation']['candidate_evidence'][0]
        self.assertEqual(grouped['candidate_ids'], [2, 4])
        self.assertEqual(grouped['relation_evidence'], proof['relation_evidence'])
        self.assertEqual(grouped['reference_ids'], [8, 9])
        self.assertEqual(grouped['exclusion_reason'], proof['exclusion_reason'])
        self.assertEqual(compact['parsed_query']['_v4_relation_evidence'][0]['unknown_member_ids'], [3])
        self.assertEqual(compact['parsed_query']['condition_records'], source['parsed_query']['condition_records'])
        self.assertNotIn('audit-only', json.dumps(compact))
        self.assertIn('summary_bytes', audit)

    def test_second_compaction_keeps_deletion_proofs_and_event_ids(self):
        event = {'event_id': 7, 'candidate_id': 2, 'dimension': 'position', 'invalidated': False,
            'source_quotes': ['near the door'], 'evidence': [{'candidate_id': 2, 'status': 'fail',
                'source_quote': 'near the door', 'reason': 'visible separated surfaces',
                'explicit_fail_criterion': 'not in contact', 'role_binding_status': 'bound'}]}
        first, _ = planner_state({'evidence_state': {'missing': ['small']}, 'deletion_events': [event]})
        second, _ = planner_state(first, tighter=True)
        self.assertEqual(second['deletion_events'], first['deletion_events'])

    def test_checkpoint_code_identity_ignores_macos_metadata_and_test_files(self):
        from eviground.grounding.v5_runtime import input_identity
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, test = (root / name for name in ('logic.py', '._logic.py', 'test_logic.py'))
            for path in (source, metadata, test):
                path.write_text('initial')
            with patch('eviground.grounding.v5_runtime.Path.glob', return_value=[source, metadata, test]):
                before = input_identity('query', [], {})
                metadata.write_bytes(b'\x00\xff resource metadata')
                test.write_text('new test')
                self.assertEqual(input_identity('query', [], {}), before)
                source.write_text('new source')
                self.assertNotEqual(input_identity('query', [], {}), before)

    def test_full_saved_controller_states_fit_without_losing_conditions(self):
        from eviground.grounding.active_diagnostic import DIMENSIONS, EVIDENCE_CAPABILITIES, INSTRUCTIONS, evidence_state
        root = ROOT / 'local_outputs/v5_development_68_r2'
        if not (root / 'manifest.json').exists():
            self.skipTest('saved controller fixture unavailable')
        manifest = json.loads((root / 'manifest.json').read_text())
        for row in (r for r in manifest['selection']['rows'] if r['csv_row'] in (3157, 5902)):
            saved = json.loads((root / f'q{row["csv_row"]:05d}/pending_inference.json').read_text())
            trace, runtime, parsed = saved['active_trace'], saved['runtime'], saved['parsed_query']
            scene = Path(manifest['scene_root']) / row['scene']
            objects = json.loads((scene / 'objects_gt.json').read_text())
            structural = json.loads((scene / 'scene_instances.json').read_text())
            if isinstance(structural, dict):
                structural = structural.get('objects', [])
            targets = set(trace['category_candidates'])
            labels = lambda rows: [{'bbox_id': o['bbox_id'], 'category': o['target']} for o in rows]
            state = {'query': row['query'], 'parsed_query': parsed,
                'target_table': labels(o for o in objects if o['bbox_id'] in targets),
                'reference_table': labels(o for o in objects + structural if o['bbox_id'] not in targets),
                'category_candidates': trace['category_candidates'],
                'remaining_candidate_ids': saved['diagnosis']['candidate_ids'],
                'history': trace['steps'], 'observation': runtime['observation'],
                'evidence_state': evidence_state(saved['diagnosis']),
                'available_dimensions': list(DIMENSIONS), 'evidence_capabilities': EVIDENCE_CAPABILITIES,
                'evidence_inventory': trace['evidence_inventory'],
                'deletion_events': trace['deletion_provenance']}
            compact, audit = planner_state(state, instruction_bytes=len(INSTRUCTIONS.encode()) + 2000)
            self.assertLess(audit['input_units'], 32768 - 2048)
            self.assertEqual(compact['parsed_query']['condition_records'], parsed['condition_records'])
            self.assertEqual(compact['evidence_state']['missing'], saved['diagnosis']['missing'])
            self.assertEqual(compact['reference_table'], state['reference_table'])

    def test_review_resume_process_failure_preserves_saved_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'q00010'
            directory.mkdir()
            prediction = {'query': self.row()['query'], 'mode': 'active_diagnostic', 'object_table': 'gt',
                'predicted_id': 1, 'accepted': False, 'review_failed': True}
            (directory / 'prediction.json').write_text(json.dumps(prediction))
            (directory / 'pending_review.json').write_text('{}')
            (directory / 'evaluation.json').write_text(json.dumps({**self.row(), 'review_failed': True}))
            def failed_worker(*args, **kwargs):
                self.assertTrue((directory / 'prediction.json').exists())
                return Mock(returncode=9)
            with patch('eviground.experiments.run_size_validation.subprocess.run', side_effect=failed_worker):
                result = run_case(self.row(), root, root, 3, retry_errors=True)
            self.assertTrue(result['inference_error'])
            self.assertEqual(result['predicted_id'], 1)
            self.assertTrue(result['correct'])
            self.assertFalse(result['accepted'])
            self.assertIn('exit code 9', result['error'])

    def test_partial_run_artifact_still_persists_process_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'q00010'
            directory.mkdir()
            (directory / 'run.json').write_text('{')
            with patch('eviground.experiments.run_size_validation.subprocess.run', return_value=Mock(returncode=1)):
                result = run_case(self.row(), root, root, 3)
            self.assertTrue(result['inference_error'])
            self.assertEqual(json.loads((directory / 'evaluation.json').read_text()), result)

    def test_malformed_parser_response_is_not_repaired_with_an_extra_api_call(self):
        from eviground.grounding.real_scene import parse_query
        from eviground.api.qwen_api import QwenAPIError
        calls = []
        with patch('eviground.grounding.real_scene.chat', return_value={'_content': '{broken', '_finish_reason': 'stop',
                '_baseline': {'latency_s': .01}, 'usage': {}}) as request:
            with self.assertRaises(QwenAPIError):
                parse_query('the chair', mode='active_diagnostic', calls=calls)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(calls[0]['stage'], 'parse_query')

    def test_batch_main_stdout_contains_only_prediction_and_correct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'scenes/scene0010_00'
            scene.mkdir(parents=True)
            for name in ('scene.npz', 'scene_instances.json', 'provenance.json'):
                (scene / name).write_text('{}')
            (scene / 'objects_gt.json').write_text('[{"bbox_id":1},{"bbox_id":2}]')
            rows = [self.row(10), self.row(11)]
            manifest = root / 'selection.json'
            manifest.write_text(json.dumps({'rows': rows, 'csv_sha256': 'offline'}))
            def completed_worker(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                (output / 'prediction.json').write_text(json.dumps({'query': self.row()['query'],
                    'object_table': 'gt', 'mode': 'active_diagnostic', 'predicted_id': 1}))
                return Mock(returncode=0)
            stdout, stderr = StringIO(), StringIO()
            argv = ['run_size_validation.py', '--manifest', str(manifest), '--scene-root', str(root / 'scenes'),
                    '--output', str(root / 'results'), '--gt', '--workers', '10']
            with patch('sys.argv', argv), patch('eviground.experiments.run_size_validation.ROOT', root), \
                    patch('eviground.experiments.run_size_validation.subprocess.run', side_effect=completed_worker), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                main()
            self.assertEqual([json.loads(line) for line in stdout.getvalue().splitlines()],
                             [{'predicted_id': 1, 'correct': True}] * 2)
            self.assertIn('2/2', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
