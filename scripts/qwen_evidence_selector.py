"""Batch original evidence for ID-only Qwen selectors; the main model decides."""
import base64
from copy import deepcopy
import hashlib
import json
import re

from qwen_step2_context import compact_catalog, encode_tables, packed


class EvidenceSelectionError(RuntimeError):
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def image_parts(messages):
    return [part for message in messages if isinstance(message.get('content'), list)
            for part in message['content'] if part.get('type') == 'image_url']


def registry(state, messages):
    from qwen_direct_fresh import expand_history_references
    from eviground.grounding.dag_monitor import usable_evidence_observation
    full = expand_history_references(state)
    images = image_parts(messages)
    views = full.get('views', [])
    if len(images) != len(views):
        raise EvidenceSelectionError('Native view/original-image count mismatch')
    observations = [row for row in full.get('observations', []) if usable_evidence_observation(row)]
    count = len(views) - sum(len(row.get('views', [])) for row in observations if row.get('tool') == 'render')
    if count <= 0:
        raise EvidenceSelectionError('No original scene preview can be identified')
    rows = [('initial', {'tool': 'initial_preview', 'views': views[:count]}, list(range(count)))]
    cursor = count
    for row in observations:
        evidence = row.get('evidence_id')
        if not isinstance(evidence, str) or not re.fullmatch(r'observation_[0-9]+', evidence):
            raise EvidenceSelectionError('Missing/unsafe native evidence ID')
        n = len(row.get('views', [])) if row.get('tool') == 'render' else 0
        rows.append((evidence, deepcopy(row), list(range(cursor, cursor + n))))
        cursor += n
    units = []
    for evidence, raw, indexes in rows:
        attachments = [deepcopy(images[i]) for i in indexes]
        hashes = [hashlib.sha256(base64.b64decode(p['image_url']['url'].split(',', 1)[1], validate=True)).hexdigest() for p in attachments]
        group = [row['id'] for row in raw.get('objects', [])] if raw.get('tool') == 'measure' else raw.get('focus_ids', [])
        units.append({'id': evidence, 'raw': raw, 'images': attachments, 'source_view_indices': indexes,
                      'image_sha256': hashes, 'object_group': group,
                      'sha256': digest({'id': evidence, 'raw': raw, 'image_sha256': hashes})})
    if len({row['id'] for row in units}) != len(units):
        raise EvidenceSelectionError('Duplicate evidence IDs; no cross-batch aliases allowed')
    return full, units


def required_evidence(state, units, prior_visible=()):
    plan = state.get('task_dag') or {}
    nodes = list(plan.get('nodes', [])) + ([plan['verification']] if plan.get('verification') else [])
    needed = {'initial'} | set(prior_visible)
    for node in nodes: needed.update(node.get('evidence_ids', []))
    bound = set(state.get('active_target_ids', []))
    for node in nodes: bound.update(node.get('bound_object_ids', []))
    measures = [row for row in units if row['raw'].get('tool') == 'measure']
    if measures: needed.add(measures[-1]['id'])
    # Retain all original measurements covering the full currently active/bound
    # comparison, so a selector cannot silently discard an acquired counter-frame.
    if bound:
        needed.update(row['id'] for row in measures if bound <= set(row['object_group']))
    available = {row['id'] for row in units}
    if needed - available:
        raise EvidenceSelectionError('Accepted/previously visible evidence missing: ' + str(sorted(needed - available)))
    return needed


def project(state, messages, units, chosen):
    from eviground.grounding.dag_monitor import usable_evidence_observation
    selected = [unit for unit in units if unit['id'] in chosen]
    result = deepcopy(state)
    result['observations'] = [deepcopy(row) for row in state.get('observations', [])
                              if not usable_evidence_observation(row) or row.get('evidence_id') in chosen]
    indexes = [i for row in selected for i in row['source_view_indices']]
    result['views'] = [deepcopy(state['views'][i]) for i in indexes]
    result['supporting_evidence_ids'] = [e for e in state.get('supporting_evidence_ids', []) if e in chosen]
    result['supporting_evidence_objects'] = {key: deepcopy(value) for key, value in state.get('supporting_evidence_objects', {}).items() if key in chosen}
    omitted = [row['id'] for row in units if row['id'] not in chosen]
    result['evidence_visibility'] = {'actually_delivered_native_ids': [row['id'] for row in selected],
        'omitted_archived_ids_not_seen': omitted,
        'original_view_indices_in_delivery_order': indexes,
        'rule': 'Only delivered evidence may support NEW claims/completions. Omitted metadata is NOT seen pixels or numbers. Keep missing evidence unresolved; do not cite an omitted observation. Original archive remains available to future selector turns.'}
    attachments = [deepcopy(part) for row in selected for part in row['images']]
    # Native state snapshots are replaced ONLY in the model-visible copy. Keep
    # the exact system contract and current candidate/structural-repair messages.
    projected = [deepcopy(message) for message in messages if message.get('role') == 'system']
    projected.append({'role': 'user', 'content': [{'type': 'text', 'text': packed(result)}, *attachments]})
    projected.extend(deepcopy(message) for message in messages
                     if message.get('role') != 'system' and isinstance(message.get('content'), str))
    for key in ('description', 'allowed_target_ids', 'active_target_ids', 'task_dag'):
        assert result.get(key) == state.get(key), 'Evidence projection changed native task state'
    assert [row for row in result['observations'] if usable_evidence_observation(row)] == [
        row for row in state.get('observations', []) if usable_evidence_observation(row) and row.get('evidence_id') in chosen]
    return result, projected


SELECTOR_RULES = ('You are an independent evidence-selector call of qwen3.8-max, NOT the grounding decision maker. '
    'Read ONLY this batch of ORIGINAL image pixels and complete raw observation groups/frames. '
    'Return only batch_id and selected_evidence_ids from the supplied enum. Select relevance to the full utterance, '
    'current task, active rivals and bound references, INCLUDING potentially contradicting evidence. '
    'Do not output a target, conclusion, ranking, explanation, binding, exclusion or DAG/Monitor changes. '
    'Unknown is not false; Top5 categories are priors. Do not invent cross-batch facts. '
    'A measurement is one original group with its own exact ordered frames; never splice groups. '
    'An empty selection is allowed if this batch has no relevant evidence. '
    'Decode exact tables: {t:i,r:rows} uses table_columns[i]; {o:i,v:values} is one object; '
    '{s:i} is shared_strings[i]. Zip recursively, no rounding. IDs/metadata alone are not seen pixels.')


def batch_request(context, rows, registry_id, mandatory):
    ids = [row['id'] for row in rows]
    batch_id = digest({'registry': registry_id, 'ids': ids, 'context': context})
    schema = {'type': 'object', 'properties': {'batch_id': {'const': batch_id},
        'selected_evidence_ids': {'type': 'array', 'uniqueItems': True, 'items': {'type': 'string', 'enum': ids}}},
        'required': ['batch_id', 'selected_evidence_ids'], 'additionalProperties': False}
    packet = encode_tables({'selector_context': context, 'registry_sha256': registry_id, 'batch_id': batch_id,
        'required_input_dependencies': [e for e in ids if e in mandatory],
        'evidence_units': [{'id': row['id'], 'sha256': row['sha256'], 'raw': row['raw']} for row in rows]})
    # Label image-to-unit association in plain text next to the ACTUAL bytes.
    parts = [{'type': 'text', 'text': packed(packet)}]
    image_index = 0
    for row in rows:
        if row['images']:
            parts.append({'type': 'text', 'text': packed({'actual_original_images_for_evidence_id': row['id'],
                'attachment_indices': list(range(image_index, image_index + len(row['images']))),
                'source_native_view_indices': row['source_view_indices'],
                                                        'sha256_in_order': row['image_sha256']})})
            parts.extend(deepcopy(row['images']))
            image_index += len(row['images'])
    return [{'role': 'system', 'content': SELECTOR_RULES}, {'role': 'user', 'content': parts}], schema, batch_id


def select_context(state, messages, prepare_selector, request_selector, prepare_main, prior_visible=()):
    full, units = registry(state, messages)
    mandatory = required_evidence(full, units, prior_visible)
    registry_id = digest({'state_sha256': digest(state), 'units': [(row['id'], row['sha256']) for row in units]})
    catalog, _, _ = compact_catalog(full)
    context = {key: catalog[key] for key in ('description', 'allowed_target_ids', 'active_target_ids', 'objects',
        'category_vocabulary', 'archived_catalog_geometry', 'task_dag', 'monitor_requirements',
        'target_category_bindings', 'reference_category_bindings') if key in catalog}
    minimum_state, minimum_messages = project(full, messages, units, mandatory)
    minimum = prepare_main(minimum_messages, minimum_state)
    audit = {'registry_sha256': registry_id, 'source_state_sha256': digest(state),
        'mandatory_ids': sorted(mandatory), 'mandatory_bound': minimum.preflight['maximum'],
        'units': [{key: row[key] for key in ('id', 'sha256', 'source_view_indices', 'image_sha256', 'object_group')} for row in units],
        'batches': [], 'selected_ids': [], 'omitted_ids': [], 'no_resize': True}
    if minimum.preflight['maximum'] >= 32768:
        raise EvidenceSelectionError('Required original-image/cited/current-comparison set cannot fit: ' + str(minimum.preflight['maximum']), audit)
    batches, current = [], []
    for unit in units:
        candidate = current + [unit]
        bm, schema, batch_id = batch_request(context, candidate, registry_id, mandatory)
        probe = prepare_selector(bm, schema)
        if probe.preflight['maximum'] < 32768:
            current = candidate
            continue
        if not current:
            raise EvidenceSelectionError('One complete original evidence unit cannot fit selector: ' + unit['id'], audit)
        batches.append(current); current = [unit]
        bm, schema, batch_id = batch_request(context, current, registry_id, mandatory)
        if prepare_selector(bm, schema).preflight['maximum'] >= 32768:
            raise EvidenceSelectionError('One complete original evidence unit cannot fit selector: ' + unit['id'], audit)
    if current: batches.append(current)
    chosen = set(mandatory)
    for index, rows in enumerate(batches):
        bm, schema, batch_id = batch_request(context, rows, registry_id, mandatory)
        probe = prepare_selector(bm, schema)
        try:
            answer, provenance = request_selector(bm, schema, {'batch_id': batch_id, 'batch_index': index,
                'allowed_ids': [row['id'] for row in rows], 'mandatory_ids': sorted(mandatory),
                'preflight': probe.preflight, 'registry_sha256': registry_id})
        except EvidenceSelectionError as exc:
            audit['failed_batch'] = {'batch_id': batch_id, 'allowed_ids': [row['id'] for row in rows],
                                     'preflight': probe.preflight}
            exc.audit = {**audit, **exc.audit}
            raise
        from jsonschema import validate
        try: validate(answer, schema)
        except Exception as exc: raise EvidenceSelectionError('Illegal selector reply, no default selection: ' + str(exc), audit) from exc
        chosen.update(answer['selected_evidence_ids'])
        audit['batches'].append({'batch_id': batch_id, 'allowed_ids': [row['id'] for row in rows],
            'selected_ids': answer['selected_evidence_ids'], 'preflight': probe.preflight, 'provenance': provenance})
    projected_state, projected_messages = project(full, messages, units, chosen)
    main = prepare_main(projected_messages, projected_state)
    audit.update(selected_ids=[row['id'] for row in units if row['id'] in chosen],
                 omitted_ids=[row['id'] for row in units if row['id'] not in chosen],
                 selected_main_bound=main.preflight['maximum'],
                 main_view_source_indices=projected_state['evidence_visibility']['original_view_indices_in_delivery_order'])
    if main.preflight['maximum'] >= 32768:
        raise EvidenceSelectionError('Model-selected original evidence union cannot fit main: ' + str(main.preflight['maximum']), audit)
    return main, audit, units


def validate_visible_references(answer, visible_ids):
    visible = set(visible_ids)
    def visit(value, claim=False):
        if isinstance(value, list):
            for row in value: visit(row, claim)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key == 'evidence_ids' and isinstance(child, list) and set(child) - visible:
                    raise ValueError('New citation not actually delivered: ' + str(sorted(set(child) - visible)))
                visit(child, key in ('reason', 'observation', 'conclusion', 'comparison'))
        elif claim and isinstance(value, str):
            missing = set(re.findall(r'\bobservation_[0-9]+\b', value)) - visible
            if missing: raise ValueError('Claim cites archived/unseen evidence: ' + str(sorted(missing)))
    visit(answer)
