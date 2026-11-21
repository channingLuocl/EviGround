"""Offline replay of saved requests through Step2's FINAL production packer/gate."""
import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import qwen_direct_fresh as fresh


def restore(folder):
    request = json.loads((folder / 'request-redacted.json').read_text())
    receipt = json.loads((folder / 'receipt.json').read_text())
    images = iter(receipt.get('input_images', []))
    for message in request['messages']:
        if not isinstance(message.get('content'), list): continue
        for part in message['content']:
            if part.get('type') != 'image_url': continue
            image = next(images)
            path = folder / image['path']
            if fresh.sha256(path) != image['sha256']:
                raise ValueError('Archived actual image hash mismatch: ' + str(path))
            part['image_url']['url'] = 'data:' + image['mime'] + ';base64,' + base64.b64encode(path.read_bytes()).decode()
    if next(images, None) is not None:
        raise ValueError('Unmatched archived image')
    return request, receipt


def prepare(request, receipt, root):
    direct = fresh.DirectCalls(root)
    direct.preview_only = True
    state = fresh.extract_state(request['messages']) if receipt['role'] == 'semantic_consultant' else None
    direct.context_state = state
    before = deepcopy(state)
    with patch.object(direct.session, 'post', side_effect=AssertionError('Offline: HTTP forbidden')):
        try:
            direct._qwen(receipt['tag'], receipt['role'], request['messages'],
                         request['response_format']['json_schema']['schema'], reserve=request['max_tokens'])
        except fresh.PreparedCall as call:
            assert state == before and not direct.ledger
            return call
    raise AssertionError('Final payload boundary was not reached')


def tentative_previews(source, output):
    """Use the actual tentative routine, with one explicitly synthetic repair."""
    episode = source/'q00171/episode'
    state_path = episode/'controller-contracts/t03/full-state.json'
    if not state_path.exists(): return []
    state = json.loads(state_path.read_text())
    history = json.loads((state_path.parent/'full-native-monitor-history.json').read_text())
    folder = next((episode/'calls').glob('*-t03-action-0-*'))
    request, _ = restore(folder)
    messages = json.loads((state_path.parent/'native-request-redacted.json').read_text())
    images = [p for m in request['messages'] if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
    slots = [p for m in messages if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
    if len(images) != len(slots): raise ValueError('Tentative original image count mismatch')
    for slot, image in zip(slots, images): slot['image_url']['url'] = image['image_url']['url']
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        direct = fresh.DirectCalls(Path(tmp)); direct.preview_only = True; direct.t_used = 3
        direct.monitor = SimpleNamespace(plan=deepcopy(history[-1]['plan']), history=deepcopy(history), active_ids=lambda: list(state['active_target_ids']))
        original = deepcopy(direct.monitor.plan)
        captured = {}
        method = direct._qwen
        def capture(*args, **kwargs):
            captured.update(messages=deepcopy(args[2]), schema=args[3])
            return method(*args, **kwargs)
        direct._qwen = capture
        try: direct.tentative_final(messages, state, 'offline_capacity_preview_only')
        except fresh.PreparedCall as first: calls = [('tentative_initial', first)]
        repair = captured['messages'] + [{'role': 'user', 'content': json.dumps({
            'errors': [{'path': ['evidence'], 'error': 'Missing required evidence field in synthetic test'}],
            'previous_raw_answer': '{}', 'instruction': 'ONE structural repair; no Monitor mutation'})}]
        try: method('t03-tentative-final-1', 'semantic_consultant', repair, captured['schema'])
        except fresh.PreparedCall as second: calls.append(('tentative_structural_repair', second))
        assert original == direct.monitor.plan and direct.ledger == []
        for name, call in calls:
            fresh.save(output/name/'final-payload-redacted.json', fresh.redact(call.payload))
            fresh.save(output/name/'preflight.json', call.preflight)
            fresh.save(output/name/'packing-audit.json', call.packing)
            results.append({'case': 'q00171', 'call': name, 'role': 'semantic_consultant',
                'offline_constructed_terminal_path': True, 'source_state_sha256': fresh.sha256(state_path),
                'maximum_with_output_reserve': call.preflight['maximum'], 'output_reserve': call.payload['max_tokens'],
                'reserved_including_output_cushion': call.preflight['output_reserved'],
                'strict_local_fit': call.preflight['maximum'] < fresh.OUTPUT_LIMIT, 'native_monitor_unchanged': True,
                'model_requests': 0, 'new_provider_usage': None,
                'synthetic_repair_not_a_model_response': name == 'tentative_structural_repair'})
    return results


def run(source, output, all_saved=False, baseline_report=None):
    paths, baseline = [], {}
    if baseline_report is not None:
        previous = json.loads(baseline_report.read_text())
        baseline = {(row['case'], row['call']): row for row in previous['results']}
        for (case, call), row in baseline.items():
            folder = source/case/'episode/calls'/call
            for filename, key in [('request-redacted.json', 'source_request_sha256'), ('receipt.json', 'source_receipt_sha256')]:
                if fresh.sha256(folder/filename) != row[key]: raise ValueError('Changed paired source: ' + str(folder/filename))
            paths.append(folder)
    for folder in sorted(source.glob('q*/episode/calls/*')):
        if baseline: break
        if not (folder/'receipt.json').exists() or not (folder/'request-redacted.json').exists(): continue
        case = folder.parents[2].name
        chosen = all_saved or case == 'q00171' or (case == 'q00173' and folder.name.startswith('008-')) or (
            case == 'q00021' and folder.name[:3] in ('008', '009', '010')) or (
            case == 'q00048' and '-t03-' in folder.name) or (case == 'q00030' and '-t04-' in folder.name) or 'parser_repair' in folder.name
        receipt = json.loads((folder/'receipt.json').read_text())
        if chosen and receipt.get('http_status') == 200: paths.append(folder)
    results = []
    for folder in paths:
        request, receipt = restore(folder)
        target = output / folder.parents[2].name / folder.name
        with tempfile.TemporaryDirectory() as tmp:
            try:
                call = prepare(request, receipt, Path(tmp))
                old = fresh.qwen_preflight(request['messages'], request['response_format'], request['max_tokens'], allow_legacy_image_fields=True)
                image_meta = []
                for message in call.payload['messages']:
                    if not isinstance(message.get('content'), list): continue
                    for part in message['content']:
                        if part.get('type') != 'image_url': continue
                        header, encoded = part['image_url']['url'].split(',', 1)
                        raw = base64.b64decode(encoded)
                        image_meta.append({'sha256': __import__('hashlib').sha256(raw).hexdigest(), 'bytes': len(raw)})
                        if call.packing.get('image_derivations'):
                            image = target/f'sent-image-{len(image_meta)-1:02d}.png'
                            image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(raw)
                fresh.save(target/'final-payload-redacted.json', fresh.redact(call.payload))
                fresh.save(target/'packing-audit.json', call.packing)
                fresh.save(target/'preflight.json', call.preflight)
                usage = receipt.get('usage') or {}
                result = {'case': folder.parents[2].name, 'call': folder.name, 'role': receipt['role'],
                    'source_request_sha256': fresh.sha256(folder/'request-redacted.json'),
                    'source_receipt_sha256': fresh.sha256(folder/'receipt.json'),
                    'old_actual_prompt_tokens': usage.get('prompt_tokens'),
                    'old_actual_completion_tokens': usage.get('completion_tokens'),
                    'old_input_byte_bound': old['maximum'] - old['output_reserved'],
                    'old_bound_covers_old_actual_usage': old['maximum'] - old['output_reserved'] >= usage.get('prompt_tokens', 0),
                    'old_visual_formula_equals_old_usage': old['image_tokens_upper_bound'] == usage.get('prompt_tokens_details', {}).get('image_tokens') if 'image_tokens' in usage.get('prompt_tokens_details', {}) else None,
                    'maximum_with_output_reserve': call.preflight['maximum'],
                    'output_reserve': call.payload['max_tokens'], 'strict_local_fit': call.preflight['maximum'] < fresh.OUTPUT_LIMIT,
                    'reserved_including_output_cushion': call.preflight['output_reserved'],
                    'sent_images': image_meta,
                    'all_measurement_evidence_retained': not call.packing['discarded_evidence'],
                    'all_original_pixels_transmitted': call.packing['sent_original_resolution'],
                    'adaptive_selected_stage': call.packing['adaptive_selected_stage'],
                    'adaptive_attempts': call.packing['adaptive_attempts'],
                    'archived_geometry_count': len(call.packing.get('archived_geometry_fields_by_id', {})),
                    'model_requests': 0, 'new_provider_usage': None}
            except (ValueError, RuntimeError, KeyError, StopIteration) as exc:
                result = {'case': folder.parents[2].name, 'call': folder.name,
                          'packing_failure': type(exc).__name__ + ': ' + str(exc), 'strict_local_fit': False, 'model_requests': 0}
            results.append(result)
            base = baseline.get((result['case'], result['call']))
            if base is not None:
                result['baseline_fit'] = base['strict_local_fit']
                result['baseline_maximum'] = base.get('maximum_with_output_reserve')
                result['baseline_refused_now_fits'] = not base['strict_local_fit'] and result['strict_local_fit']
    terminal_paths = tentative_previews(source, output/'tentative-paths')
    report = {'source': str(source), 'requests_replayed': len(results),
              'fit': sum(r['strict_local_fit'] for r in results),
              'nonfit_or_failure': sum(not r['strict_local_fit'] for r in results),
              'model_requests': 0, 'GT_read': False,
              'paired_baseline': str(baseline_report) if baseline_report else None,
              'baseline_refused_now_fits': sum(row.get('baseline_refused_now_fits', False) for row in results),
              'adaptive_rescues': sum(row['strict_local_fit'] and row.get('adaptive_attempts', [{}])[0].get('maximum', 0) >= fresh.OUTPUT_LIMIT for row in results),
              'statement': 'Local engineering upper bounds under documented assumptions; new payload actual provider token counts unknown. Historical usage audit applies to ORIGINAL payloads only.',
              'results': results, 'tentative_offline_previews': terminal_paths}
    fresh.save(output/'replay-report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False))
    for result in results:
        if not result['strict_local_fit'] or result['case'] == 'q00171': print(json.dumps(result, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=fresh.PROJECT/'outputs/qwen-direct/full-nr3d-20261120-v3')
    parser.add_argument('--output', type=Path, default=fresh.PROJECT/'outputs/qwen-direct/step2-adaptive-validation-20261120/replay')
    parser.add_argument('--baseline-report', type=Path, default=fresh.PROJECT/'outputs/qwen-direct/step2-local-validation-20261120/final-replay/replay-report.json')
    parser.add_argument('--all-saved', action='store_true')
    args = parser.parse_args()
    run(args.source, args.output, args.all_saved, args.baseline_report)
