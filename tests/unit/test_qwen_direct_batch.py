"""Offline fake-worker checks: denominator, GT boundary, resume and stop handling."""
import io
import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import qwen_direct_batch as batch
import qwen_direct_fresh as fresh


def fixture(tmp_path, count=5):
    rows = [{'scene': 'scene_fake', 'csv_row': i, 'query': f'raw utterance{i}'} for i in range(count)]
    assets = {'scene_entries': {'scene_fake': {'paths': {'scene.npz': str(tmp_path/'scene.npz'),
                                                       'category': str(tmp_path/'objects_top5.json')}, 'hashes': {}}}}
    batch.save(tmp_path/'manifest.json', {'frozen_fake': True})
    batch.save(tmp_path/'parent-evaluation/selection.json', {'rows': [{**r, 'target_id': 7} for r in rows]})
    return rows, assets


def worker_of(kinds, calls, after=None):
    def fake(command, directory, stop):
        request = batch.read(Path(command[-1]))
        assert set(request['sample']) == {'scene', 'query', 'csv_row'}
        assert 'target_id' not in json.dumps(request)
        assert 'parent-evaluation' not in json.dumps(request)
        calls.append(request['sample']['csv_row'])
        kind = kinds[request['sample']['csv_row']]
        episode = directory/'episode'
        strict = kind.startswith('strict')
        status = 'completed' if strict else 'tentative' if kind == 'tentative' else 'abstained' if kind == 'null' else kind
        predicted = 7 if kind in ('strict_right', 'tentative') else 8 if strict else None
        batch.save(episode/'result.json', {'status': status, 'predicted_id': predicted,
                   'strict_monitor_accepted': strict, 'terminal_mode': 'strict_monitor' if strict else 'tentative' if kind == 'tentative' else 'failed',
                   'budgets': {'B_used': 1, 'R_used': 3, 'T_used': 2}, 'prediction': {}})
        batch.save(episode/'calls/001/receipt.json', {'http_sent': True, 'http_status': 200,
                   'status': 'completed', 'tag': 'fake', 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}})
        batch.save(directory/'worker-start.json', {'pid': 999999, 'fake': True})
        if after:
            after(stop)
        return 0, .01
    return fake


def test_completed_denominator_keeps_wrong_null_failed_and_tentative_resume(tmp_path):
    rows, assets = fixture(tmp_path); calls = []
    kinds = ['strict_right', 'strict_wrong', 'tentative', 'null', 'failed']
    original_read = batch.read
    def guarded_read(path):
        if Path(path).name == 'selection.json':
            freezes = list(tmp_path.glob('q*/prediction-freeze.json'))
            assert freezes, 'Annotations decoded before any per-sample freeze'
            # At scoring time this sample pointer is already atomically frozen.
            assert all(original_read(p)['annotation_read'] is False for p in freezes)
        return original_read(path)
    with patch.object(batch, 'read', side_effect=guarded_read):
        result = batch.run_batch(tmp_path, {}, rows, assets, worker_command=worker_of(kinds, calls))
    assert calls == list(range(5))
    summary = batch.read(tmp_path/'summary.json')
    assert summary['correct'] == 2 and summary['completed'] == 5
    assert summary['accuracy_completed'] == .4
    assert summary['strict'] == 2 and summary['tentative'] == 1 and summary['abstained'] == 1 and summary['failed'] == 1
    assert summary['known_prompt_tokens'] == 50
    assert all((tmp_path/f'q{i:05d}/evaluation.json').exists() for i in range(5))
    with patch.object(batch, 'parent_evaluate', side_effect=AssertionError('do not reevaluate/reinvoke completed')):
        result = batch.run_batch(tmp_path, {}, rows, assets,
            worker_command=lambda *a: pytest.fail('completed sample was re-sent'))
    assert len(result) == 5


def test_ctrl_c_drains_current_and_stops_scheduling_then_resumes(tmp_path):
    rows, assets = fixture(tmp_path, 3); calls = []; stop = batch.StopState()
    old = signal.signal(signal.SIGINT, stop.signal)
    try:
        result = batch.run_batch(tmp_path, {}, rows, assets, stop=stop,
            worker_command=worker_of(['strict_right']*3, calls, after=lambda s: s.signal(signal.SIGINT, None)))
    finally:
        signal.signal(signal.SIGINT, old)
    assert calls == [0] and len(result) == 1
    assert batch.read(tmp_path/'summary.json')['state'] == 'stopped'
    result = batch.run_batch(tmp_path, {}, rows, assets, worker_command=worker_of(['strict_right']*3, calls))
    assert calls == [0, 1, 2] and len(result) == 3


def test_interrupted_and_unknown_are_not_blindly_resent_or_in_denominator(tmp_path):
    rows, assets = fixture(tmp_path, 1); stop = batch.StopState(); calls = []
    def after(s):
        s.signal(signal.SIGINT, None); s.signal(signal.SIGINT, None)
    result = batch.run_batch(tmp_path, {}, rows, assets, stop=stop,
        worker_command=worker_of(['interrupted'], calls, after=after))
    assert result == [] and calls == [0]
    assert batch.read(tmp_path/'summary.json')['completed'] == 0
    with pytest.raises(ValueError, match='Interrupted'):
        batch.run_batch(tmp_path, {}, rows, assets, worker_command=lambda *a: pytest.fail('unsafe resend'))
    p = tmp_path/'q00000/result.json'; p.unlink(); (tmp_path/'q00000/episode/result.json').unlink()
    batch.save(tmp_path/'q00000/episode/calls/002/receipt.json', {'http_sent': True, 'status': 'sent', 'tag': 'unknown'})
    assert batch.call_accounting(tmp_path/'q00000/episode')['unknown_usage'] == ['unknown']
    with pytest.raises(ValueError, match='Unfinished worker'):
        batch.run_batch(tmp_path, {}, rows, assets, worker_command=lambda *a: pytest.fail('unknown resend'))
    summary = batch.read(tmp_path/'summary.json')
    assert summary['completed'] == 0
    assert summary['unknown_usage_calls_including_unfinished'] == 1
    assert summary['unfinished_cases'][0]['usage']['remote_execution_unknown'] == ['unknown']


def test_worker_rejects_annotation_fields_without_model_calls(tmp_path):
    request = tmp_path/'unsafe.json'
    batch.save(request, {'sample': {'scene': 'scene_fake', 'query': 'x', 'csv_row': 1, 'target_id': 7},
                         'scene_dir': 'x', 'category_path': 'x', 'shared_manifest': 'x', 'input_hashes': {}})
    with patch.dict(fresh.os.environ, {fresh.SILICONFLOW_KEY_ENV: 'mock-private'}), patch.object(fresh.requests.Session, 'post', side_effect=AssertionError('offline test cannot call model')):
        with pytest.raises(ValueError, match='forbidden'):
            fresh.run(tmp_path/'bad-worker', request)


def test_prediction_must_be_frozen_before_annotation_read(tmp_path):
    row = {'scene': 'scene_fake', 'csv_row': 1, 'query': 'x'}
    batch.save(tmp_path/'result.json', {'status': 'failed', 'predicted_id': None, 'prediction': {}})
    annotation = tmp_path/'annotation.json'
    batch.save(annotation, {'rows': [{**row, 'target_id': 9}]})
    original_read = batch.read
    def guard(path):
        if Path(path) == annotation:
            assert (tmp_path/'prediction-freeze.json').exists()
            assert original_read(tmp_path/'prediction-freeze.json')['result_sha256'] == batch.sha(tmp_path/'result.json')
        return original_read(path)
    with patch.object(batch, 'read', side_effect=guard):
        result = batch.parent_evaluate(row, tmp_path, annotation, 'shared-hash')
    assert result['correct'] is False and result['target_id'] == 9
    assert 'target_id' not in batch.inference_selection(annotation, expected=1)[0]


def test_safe_worker_process_and_parent_stop_does_not_interrupt_saved_response(tmp_path):
    """Exercise the real subprocess/log loop using a fake script, no API imports."""
    script_dir = tmp_path/'source/scripts'; script_dir.mkdir(parents=True)
    script = script_dir/'fake_worker.py'
    directory = tmp_path/'q'; directory.mkdir()
    script.write_text("import json,time\nfrom pathlib import Path\np=Path("+repr(str(directory))+ ")\nprint('fake request started',flush=True)\ntime.sleep(.3)\n(p/'fake-response.json').write_text(json.dumps({'saved':True}))\nprint('fake response saved',flush=True)\n")
    stop = batch.StopState(); stop.requested = True; stop.interrupts = 1
    rc, _ = batch.run_worker([sys.executable, str(script)], directory, stop)
    assert rc == 0 and batch.read(directory/'fake-response.json')['saved'] is True
    assert 'fake response saved' in (directory/'console.log').read_text()


def test_freeze_resume_identity_and_gt_not_in_worker_selection(tmp_path, monkeypatch):
    rows = [{'scene': 'scene_fake', 'csv_row': 1, 'query': 'x'}]
    selection = tmp_path/'annotation-source.json'; batch.save(selection, {'rows': [{**rows[0], 'target_id': 7}]})
    cloud = tmp_path/'scene.npz'; cloud.write_bytes(b'fake point cloud')
    assets_path = tmp_path/'assets.json'
    assets = {'ready': True, 'selection_sha256': batch.sha(selection), 'scene_entries': {
        'scene_fake': {'hashes': {str(cloud): batch.sha(cloud)}, 'paths': {'scene.npz': str(cloud)}}}}
    batch.save(assets_path, assets)
    output = tmp_path/'output'; output.mkdir()
    original_selection = batch.inference_selection
    monkeypatch.setattr(batch, 'inference_selection', lambda p: original_selection(p, expected=1))
    monkeypatch.setattr(batch, 'source_closure', lambda: {'pyproject.toml': batch.sha(batch.ROOT/'pyproject.toml')})
    import qwen_direct_fresh
    monkeypatch.setattr(qwen_direct_fresh, 'build_frozen_manifest',
                        lambda **kwargs: {'model': 'fake', 'source_closure': {}, 'inputs': {}})
    one, selected, _ = batch.freeze_run(output, selection, assets_path)
    two, _, _ = batch.freeze_run(output, selection, assets_path)
    assert one == two and selected == rows
    assert 'target_id' not in (output/'inference-selection.json').read_text()
    assert 'target_id' not in (output/'manifest.json').read_text()
    assert (output/'source/pyproject.toml').exists()
    cloud.write_bytes(b'changed cloud')
    with pytest.raises(ValueError, match='Changed shared input'):
        batch.freeze_run(output, selection, assets_path)


def test_zero_request_startup_failure_pauses_after_first_and_keeps_denominator(tmp_path, capsys):
    rows, assets = fixture(tmp_path, 3); calls = []
    def no_api_failure(command, directory, stop):
        sample = batch.read(Path(command[-1]))['sample']; calls.append(sample['csv_row'])
        batch.save(directory/'episode/result.json', {'status': 'failed', 'predicted_id': None,
                   'strict_monitor_accepted': False, 'terminal_mode': 'failed', 'budgets': {'B_used': 0, 'R_used': 0, 'T_used': 0},
                   'prediction': {'failure_type': 'PackageNotFoundError', 'failure': 'No package metadata was found for spacy'}})
        return 0, .1
    results = batch.run_batch(tmp_path, {}, rows, assets, worker_command=no_api_failure)
    assert calls == [0] and len(results) == 1
    summary = batch.read(tmp_path/'summary.json')
    assert summary['completed'] == 1 and summary['failed'] == 1 and summary['correct'] == 0
    assert summary['state'] == 'attention_global_zero_api_startup_failure'
    assert 'PackageNotFoundError' in capsys.readouterr().out


def test_real_worker_bootstrap_failure_is_fail_closed_before_batch(tmp_path, monkeypatch):
    rows, assets = fixture(tmp_path, 1)
    script = tmp_path/'source/scripts/qwen_direct_fresh.py'; script.parent.mkdir(parents=True)
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--run-dir');p.add_argument('--worker-input');p.add_argument('--bootstrap-check',action='store_true');a=p.parse_args();d=Path(a.run_dir);d.mkdir();(d/'result.json').write_text(json.dumps({'status':'failed','prediction':{'failure_type':'ModuleNotFoundError','failure':'missing required module'}}))\n")
    with pytest.raises(ValueError, match='Real worker startup failed'):
        batch.bootstrap_worker(tmp_path, rows, assets)
    assert batch.read(tmp_path/'worker-bootstrap-status.json')['ready'] is False
    assert not list(tmp_path.glob('q[0-9]*'))
