"""Zero-HTTP real-request regressions; selector answers are explicitly SIMULATED."""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import qwen_direct_fresh as fresh
import qwen_step2_context as codec
import qwen_evidence_selector as selector
from check_qwen_step2_context import restore, tentative_previews


def original_request(folder):
    request, receipt = restore(folder)
    original = folder/'original-request-redacted.json'
    if not original.exists(): return request, receipt
    messages = json.loads(original.read_text())
    sources = {row['sha256']: row for row in receipt.get('original_input_images', receipt['input_images'])}
    mapping_file = folder/'context-packing.json'
    mapping = json.loads(mapping_file.read_text()).get('image_delivery', []) if mapping_file.exists() else []
    slots = selector.image_parts(messages)
    if len(mapping) != len(slots):
        if len(sources) != len(slots): raise ValueError('Cannot reconstruct exact original image order')
        mapping = [{'sha256': row['sha256']} for row in sources.values()]
    for slot, row in zip(slots, mapping):
        meta = sources[row['sha256']]; file = folder/meta['path']
        if fresh.sha256(file) != row['sha256']: raise ValueError('Original image hash changed')
        slot['image_url']['url'] = 'data:' + meta['mime'] + ';base64,' + base64.b64encode(file.read_bytes()).decode()
    request['messages'] = messages
    return request, receipt


def run_case(source, prefix, output, simulate):
    folder = next((source/'calls').glob(prefix+'-*'))
    request, receipt = original_request(folder)
    state = fresh.extract_state(request['messages']); before = deepcopy(state)
    direct = fresh.DirectCalls(output); direct.preview_only = True; direct.context_state = state
    if simulate == 'required-only':
        direct.selector_preview = lambda meta: [e for e in meta['allowed_ids'] if e in meta['mandatory_ids']]
    elif simulate == 'all': direct.selector_preview = lambda meta: meta['allowed_ids']
    schema_file = folder/'original-response-schema.json'
    schema = json.loads(schema_file.read_text()) if schema_file.exists() else request['response_format']['json_schema']['schema']
    result = {'source': str(folder), 'source_receipt_sha256': fresh.sha256(folder/'receipt.json'),
              'simulation': simulate, 'actual_model_requests': 0, 'provider_usage_for_new_payload': None}
    with patch.object(direct.session, 'post', side_effect=AssertionError('NO HTTP authorized')):
        try:
            direct._qwen(receipt['tag'], receipt['role'], request['messages'], schema, reserve=request['max_tokens'])
        except fresh.PreparedCall as call:
            fresh.save(output/'main-payload-redacted.json', fresh.redact(call.payload))
            fresh.save(output/'main-preflight.json', call.preflight)
            fresh.save(output/'main-packing.json', call.packing)
            for i, part in enumerate(selector.image_parts(call.payload['messages'])):
                raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                image = output/'main-images'/f'{i:03d}.png'; image.parent.mkdir(parents=True, exist_ok=True); image.write_bytes(raw)
            selection = call.packing.get('evidence_selection')
            result.update(status='prepared_not_sent', main_bound=call.preflight['maximum'],
                selector_triggered=bool(selection), selection=selection,
                selected_original_image_sha256=[i['sha256'] for i in call.preflight['images']],
                no_resize=not call.packing['image_derivations'],
                model_visible_native_evidence_ids=call.packing['visible_native_evidence_ids'])
        except selector.EvidenceSelectionError as exc:
            result.update(status='explicit_selection_failure_not_sent', reason=str(exc), audit=exc.audit)
    assert state == before
    assert not any(row.get('http_sent') for row in direct.ledger)
    result['native_state_unchanged'] = True
    result['simulated_batch_count'] = len(list(output.glob('evidence-selections/s*/batch-*/SIMULATED-selection-not-model-response.json')))
    fresh.save(output/'case-report.json', result)
    return result


def quote_regression(output):
    fixture = json.loads((fresh.PROJECT/'tests/fixtures/qwen_step2_exact_quotes.json').read_text())
    root = fresh.PROJECT/'outputs/qwen-direct/step2-adaptive-smoke-20261120/q00171/episode'
    state = json.loads((root/'controller-contracts/t01/full-state.json').read_text())
    native = json.loads((root/'controller-contracts/t01/native-action-schema.json').read_text())
    wire = codec.compact_wire_schema({'properties': {'state_version': {'const': 'offline'}, 'action': {}}}, state)
    validator = fresh.Draft202012Validator(wire['properties']['action'])
    bad = [{'source': row['source'], 'wire_rejects': not validator.is_valid(row['action']),
            'native_rejects': not fresh.Draft202012Validator(native).is_valid(row['action'])} for row in fixture['bad_real_actions']]
    positive = fixture['positive_real_action']['action']
    result = {'description_verbatim': state['description'],
        'allowed_quotes': __import__('eviground.grounding.dag_monitor', fromlist=['exact_source_quotes']).exact_source_quotes(state['description'], state['monitor_requirements']),
        'bad_real_actions': bad, 'positive_real_wire_valid': validator.is_valid(positive),
        'positive_real_native_valid': fresh.Draft202012Validator(native).is_valid(positive),
        'no_quote_correction_performed': True, 'GT_read': False, 'new_HTTP': 0}
    assert all(row['wire_rejects'] and row['native_rejects'] for row in bad)
    assert result['positive_real_wire_valid'] and result['positive_real_native_valid']
    fresh.save(output/'quote-regression.json', result)
    return result


def raw_measurement_batch(output):
    folder = next((fresh.PROJECT/'outputs/qwen-direct/full-nr3d-20261120-v3/q00048/episode/calls').glob('005-*'))
    request, _ = restore(folder); state = fresh.extract_state(request['messages']); full, units = selector.registry(state, request['messages'])
    unit = next(row for row in units if row['raw'].get('tool') == 'measure')
    messages, schema, _ = selector.batch_request({'description': full['description'], 'active_target_ids': full['active_target_ids']},
        [unit], selector.digest({'diagnostic': unit['sha256']}), {unit['id']})
    prepared = fresh.prepare_qwen_payload(messages, schema, 'evidence_selector', 768)
    decoded = codec.decode_tables(json.loads(prepared.payload['messages'][1]['content'][0]['text']))
    assert decoded['evidence_units'][0]['raw'] == unit['raw']
    result = {'source': str(folder), 'evidence_id': unit['id'], 'unit_sha256': unit['sha256'],
        'directed_pairs': len(unit['raw']['directed_pair_measurements']),
        'conditional_frames': len(unit['raw']['conditional_observer_frames']),
        'selector_bound': prepared.preflight['maximum'], 'raw_group_and_frame_roundtrip_equal': True, 'HTTP': 0}
    fresh.save(output/'raw-measurement-selector-payload.json', fresh.redact(prepared.payload))
    fresh.save(output/'raw-measurement-selector-proof.json', result)
    return result


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists(): raise FileExistsError('Use a new output; do not overwrite a diagnostic freeze')
    base = fresh.PROJECT/'outputs/qwen-direct'
    cases = [run_case(base/'step2-adaptive-smoke-20261120/q00171/episode', '005', output/'q171-quote-repair', None),
             run_case(base/'full-nr3d-20261120-v3/q00173/episode', '006', output/'q173-required-only-SIMULATED', 'required-only'),
             run_case(base/'full-nr3d-20261120-v3/q00173/episode', '006', output/'q173-select-all-SIMULATED', 'all'),
             run_case(base/'full-nr3d-20261120-v3/q00048/episode', '005', output/'q48-mandatory-too-large', 'required-only')]
    report = {'quote_regression': quote_regression(output), 'cases': cases,
              'measurement_batch': raw_measurement_batch(output),
              'tentative': tentative_previews(base/'full-nr3d-20261120-v3', output/'tentative'),
              'all_selector_responses_are_simulations_not_model_results': True, 'actual_new_HTTP': 0, 'GT_read': False}
    fresh.save(output/'report.json', report)
    print(json.dumps({'cases': [{k: v for k, v in row.items() if k in ('status', 'main_bound', 'selector_triggered', 'reason', 'simulated_batch_count')} for row in cases],
                      'quote_regression': report['quote_regression'], 'measurement_batch': report['measurement_batch'], 'HTTP': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=fresh.PROJECT/'outputs/qwen-direct/step2-selector-validation-20261120/replay')
    main(parser.parse_args().output)
