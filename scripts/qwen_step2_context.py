"""Model-visible, lossless evidence packing; no target choice or Monitor edits."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from collections import Counter
import re
from PIL import Image


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


RULES = """You are the planner/semantic Monitor; return one JSON object with the supplied schema. B/R/T are record-only; capacity is separate and never forces evidence-free completion. The original utterance outranks parser/Top5 priors. Unknown is not false; ambiguous appearance is not confirmed. Compare every active target and strongest rivals; keep negation, noun attachment and the full reference/comparison group. Bind nested support references before comparing targets.
Inspect appearance/category with targets AND references; isolated tiles have independent cameras/scales, not spatial relations. Up to12 IDs; azimuth[0,360), elevation[10,75], null defaults315/30. Measure distance/support/height/order. frame=null for non-directional distances; origin_ids=[] means room center, facing_ids is what is faced, not the origin. Groups differ. Conditional frames require explicit binding/assumptions. Honor raw same-frame signs, units and rows_left_to_right; never reverse them. AABB does not identify bed head/foot.
YOU propose all bindings/exclusions/completions/rollback. Completed nodes cite existing object-covering evidence and observed conclusions, not priors/unknown. initial is delivered pixels/basic geometry. Cover every requirement. n0..n10 are tasks; n11 is finish-only verification depending transitively on all tasks. For tools current node is pending, allows the tool, dependencies completed; verification stays pending. allowed_actions restricts NEXT tool, not evidence that may complete an old task. Complete supported old tasks and switch to another ready pending task in the SAME proposal. Delta changes state only; change goals/tools/dependencies via COMPLETE replace_plan with verification and all requirement_nodes. Reopen contradicted old IDs/descendants; preserve unchanged completed states verbatim.
inspect/measure: predicted_id=null,evidence=[]; inspect:frame=null. finish:focus_ids=[],context/camera/frame=null. Strict non-null finish needs all nodes completed,current=n11,unresolved=[], supported exclusions leaving only that ID, and rival comparison. Otherwise null with missing evidence. Tentative never changes Monitor or claims strict acceptance. reason<=60words,evidence<=4,observation<=35words; evidence source_quote=FULL original; Monitor quotes=exact substrings.
Decode packet: root state/context_and_exact_feedback. {t:i,r:rows} uses table_columns[i]; {o:i,v:values} is one object; {s:i} is exact shared_strings[i]. Zip columns/values recursively; no rounding. category_index names category_vocabulary; hypotheses=[index,rank,gap]; top5_category_indices gives rank1..5 WITHOUT archived gaps. All IDs/category alternatives remain. Active, possible lexical reference and model-focused/bound geometry/gaps stay detailed; other geometry/gaps explicitly archived. measure any known IDs for raw geometry, inspect to establish category; never cite unseen archived facts. Lexical scope is not semantic binding. Delivered pair/frame groups are complete; selected-context omissions are explicitly listed and NOT seen. pair_refs and frame_refs resolve raw catalogs WITHIN original observation/group; never mix frames/groups. image_delivery maps supplied views to actual attachments; identical bytes may share one. Metadata is not pixels. Latest exact rejection/rollback and repair are authoritative. anyOf branches are alternatives.
"""

REPORT_PLANNING_RULES = """Use the exact description and full reference group. Category hypotheses are priors. Compare every active rival. Use same-group raw measurements; for non-directional distance use frame=null. Include candidates and relevant references. Read delivered measurements before requesting more. Never repeat an action in unavailable_evidence_actions; choose a new group, frame or camera if evidence is missing. If a measurement can resolve the gap, prefer it to rereading the same camera.
Plan only distinct evidence tasks; share tasks across requirements. Parsing and planning are not evidence. Finish belongs only to n11. Initially submit the bare plan, with n0 first and dependency-free; list all task IDs in verification.depends_on. Later update the current and changed nodes, or replace_plan for structural changes. The next tool node stays pending and ready. To complete the only task, finish or propose a replacement with a new ready task. Record exclusions on the completed evidence task. Cite native evidence IDs on nodes and visual finding IDs in conclusions. Reopen contradicted tasks and descendants.
For adjacency inspect scene views; isolated views establish appearance only. Respect the schema's focus_ids limit and give a short, nonempty monitor.revision_reason when planning or changing the plan. Inspect needs scene/isolated context and null frame. Measure needs null context and camera. Strict finish requires completed tasks, a unique supported target and unresolved=[]. Use null when essential evidence remains missing and list the gaps. Keep reason, goal and conclusions short.
Decode {t:i,r:rows} and {o:i,v:values} with table_columns[i], {s:i} with shared_strings[i]. A table's c gives constant [column_index,value] pairs. Object catalog tables hold records keyed by ID; use category_vocabulary and hypotheses=[index,rank,gap]. pair_refs and frame_refs resolve exact raw tables within one observation. Respect the latest rejection and rollback. repair lists [path,error] for an unaccepted proposal.
"""


def encode_tables(value, constants=False):
    """Same exact table representation as native compact_state, shared headers."""
    columns = []
    counts = Counter()

    def count(item):
        if isinstance(item, str):
            counts[item] += 1
        elif isinstance(item, dict):
            for value in item.values(): count(value)
        elif isinstance(item, list):
            for value in item: count(value)
    count(value)
    strings = [s for s, n in counts.items() if n > 1 and len(s) >= 16]

    def schema(keys):
        if keys not in columns:
            columns.append(keys)
        return columns.index(keys)

    def visit(item):
        if isinstance(item, list):
            if item and all(isinstance(row, dict) and list(row) == list(item[0]) for row in item):
                keys = list(item[0])
                rows = [[visit(row[k]) for k in keys] for row in item]
                table = {"t": schema(keys), "r": rows}
                if constants and len(rows) > 1:
                    fixed = [i for i in range(len(keys)) if all(packed(row[i]) == packed(rows[0][i]) for row in rows)]
                    factored = {"t": table['t'], 'c': [[i, rows[0][i]] for i in fixed],
                                'r': [[value for i, value in enumerate(row) if i not in fixed] for row in rows]}
                    if len(packed(factored)) < len(packed(table)):
                        return factored
                return table
            return [visit(row) for row in item]
        if isinstance(item, dict):
            keys = list(item)
            return {"o": schema(keys), "v": [visit(item[key]) for key in keys]}
        if isinstance(item, str) and item in strings:
            return {"s": strings.index(item)}
        return item

    data = visit(value)
    return {"table_columns": columns, "shared_strings": strings, "data": data}


def decode_tables(packet):
    def visit(item):
        if isinstance(item, dict) and set(item) == {"s"}:
            return packet["shared_strings"][item["s"]]
        if isinstance(item, dict) and set(item) == {"o", "v"}:
            return dict(zip(packet["table_columns"][item["o"]], [visit(v) for v in item["v"]], strict=True))
        if isinstance(item, dict) and set(item) in ({"t", "r"}, {"t", "r", "c"}):
            keys = packet["table_columns"][item["t"]]
            fixed = dict(item.get('c', []))
            result = []
            for row in item['r']:
                values = iter(row)
                full = [fixed[i] if i in fixed else next(values) for i in range(len(keys))]
                sentinel = object()
                if next(values, sentinel) is not sentinel: raise ValueError('Extra values in constant-column table')
                result.append(dict(zip(keys, [visit(v) for v in full], strict=True)))
            return result
        if isinstance(item, dict):
            return {key: visit(value) for key, value in item.items()}
        if isinstance(item, list):
            return [visit(value) for value in item]
        return item
    return visit(packet["data"])


def pack_measurements(state):
    """Atomize by directed object pair and original frame, retaining every group."""
    state = deepcopy(state)
    pairs, frames = [], []
    pair_index, frame_index = {}, {}

    def intern(items, value):
        index = pair_index if items is pairs else frame_index
        key = packed(value)
        if key not in index:
            index[key] = len(items)
            items.append(value)
        return index[key]

    for observation in state.get("observations", []):
        if observation.get("tool") != "measure":
            continue
        if "directed_pair_measurements" in observation:
            rows = observation.pop("directed_pair_measurements")
            observation["pair_refs"] = [intern(pairs, row) for row in rows]
            observation["comparison_group"] = [row["id"] for row in observation.get("objects", [])]
        for field in ("selected_observer_frame", "conditional_observer_frames"):
            if field not in observation:
                continue
            value = observation.pop(field)
            observation[field + "_refs"] = ([intern(frames, row) for row in value]
                                             if isinstance(value, list) else
                                             [intern(frames, value)] if value is not None else [])
    state["measurement_pairs"], state["measurement_frames"] = pairs, frames
    return state


def unpack_measurements(state):
    state = deepcopy(state)
    pairs, frames = state.pop("measurement_pairs"), state.pop("measurement_frames")
    for row in state.get("observations", []):
        if "pair_refs" in row:
            row["directed_pair_measurements"] = [pairs[i] for i in row.pop("pair_refs")]
            row.pop("comparison_group")
        for field in ("selected_observer_frame", "conditional_observer_frames"):
            if field + "_refs" in row:
                values = [frames[i] for i in row.pop(field + "_refs")]
                row[field] = values if field == "conditional_observer_frames" else values[0] if values else None
    return state


def compact_wire_schema(schema, state=None, *, text_visual_reports=False, compact_replacement=False,
                        native_finish=False):
    """Only generation hints shrink; exact original Schema still validates locally."""
    if state is not None and 'action' in schema.get('properties', {}):
        string = {'type': 'string'}
        reason = {**string, 'minLength': 1}
        ids = {'type': 'array', 'items': {'type': 'integer'}}
        strings = {'type': 'array', 'items': string}
        def obj(fields):
            return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
        from eviground.grounding.dag_monitor import exact_source_quotes
        quotes = exact_source_quotes(state.get('description', ''), state.get('monitor_requirements', []))
        quote_array = {**strings, 'minItems': 1, 'maxItems': len(quotes),
                       'items': {'type': 'string', 'enum': quotes}} if quotes else strings
        evidence = obj({'source_quote': {'const': state['description']} if 'description' in state else string,
                        'object_ids': ids, 'observation': string})
        action = obj({'action': {'enum': ['inspect', 'measure', 'finish']}, 'reason': reason, 'focus_ids': ids,
            'context': {'enum': ['scene', 'isolated', None]}, 'azimuth_deg': {'type': ['number', 'null']},
            'elevation_deg': {'type': ['number', 'null']}, 'frame': {'anyOf': [{'type': 'null'}, obj({'origin_ids': ids, 'facing_ids': ids})]},
            'predicted_id': {'type': ['integer', 'null']}, 'evidence': {'type': 'array', 'items': evidence},
            'unresolved': strings, 'monitor': {'type': 'object'}})
        node_ids = [f'n{i}' for i in range(12)]
        node_id = {'enum': node_ids}
        evidence_ids = sorted(set(state.get('supporting_evidence_ids', ())))
        citations = {**strings, 'items': {'enum': evidence_ids}, 'maxItems': len(evidence_ids)} if evidence_ids else strings
        excluded = ids
        if text_visual_reports and 'allowed_target_ids' in state:
            allowed = state['allowed_target_ids']
            excluded = ({**ids, 'items': {'enum': allowed}, 'maxItems': len(allowed)}
                        if allowed else {'const': []})
        node_state = {'id': node_id, 'status': {'enum': ['pending', 'completed']},
                      'bound_object_ids': ids, 'evidence_ids': citations, 'conclusion': string,
                      'excluded_target_ids': excluded}
        task = obj({**node_state, 'id': {'enum': node_ids[:-1]}, 'goal': reason,
                    'kind': {'enum': ['binding', 'attribute', 'relation', 'comparison']},
                    'source_quotes': quote_array,
                    'depends_on': {**strings, 'items': {'enum': node_ids[:-1]}, 'maxItems': 11},
                    'allowed_actions': {'type': 'array', 'maxItems': 2, 'enum': [
                        [], ['inspect'], ['measure'], ['inspect', 'measure'], ['measure', 'inspect']]}})
        verification = obj({**task['properties'], 'id': {'const': 'n11'}, 'kind': {'const': 'verification'},
                            'allowed_actions': {'enum': [['finish']]}})

        def by_status(fields):
            if not text_visual_reports:
                return obj(fields)
            pending = {**fields, 'status': {'const': 'pending'}, 'evidence_ids': {'const': []},
                       'conclusion': {'const': ''}, 'excluded_target_ids': {'const': []}}
            if fields.get('kind') == task['properties']['kind']:
                pending['allowed_actions'] = {'enum': task['properties']['allowed_actions']['enum'][1:]}
            return {'anyOf': [
                obj(pending),
                obj({**fields, 'status': {'const': 'completed'}, 'evidence_ids': {**citations, 'minItems': 1},
                     'conclusion': reason}),
            ]}

        assigned = {**strings, 'items': node_id, 'minItems': 1, 'maxItems': 12}
        changes = {'current_node_id': node_id, 'reopen_node_ids': {**assigned, 'minItems': 0},
                   'revision_reason': reason}
        initial = not state.get('task_dag')
        nodes = {'type': 'array', 'minItems': 1, 'maxItems': 11, 'items': by_status(task['properties'])}
        if text_visual_reports and initial:
            verification['properties']['depends_on'] = {'enum': [node_ids[:count] for count in range(1, 12)]}
        plan = obj({
                'nodes': nodes,
                'verification': by_status(verification['properties']),
                'requirement_nodes': obj({row['id']: assigned for row in state.get('monitor_requirements', [])}),
                **changes})
        if initial:
            action['properties']['monitor'] = plan
        else:
            delta = obj({'node_updates': {'type': 'array', 'maxItems': 12, 'items': by_status(node_state)}, **changes})
            replacement = obj({
                'nodes': {'type': 'array', 'minItems': 1, 'maxItems': 11, 'items': {'type': 'object'}},
                'verification': {'type': 'object'},
                'requirement_nodes': {'type': 'object'},
                **changes,
            }) if compact_replacement else plan
            action['properties']['monitor'] = {'anyOf': [delta, obj({'replace_plan': replacement})]}
        focus_limits = {}
        inspect_angles = {}
        finish_fields = []
        for variant in schema['properties']['action'].get('anyOf', []):
            native = variant.get('properties', {})
            kinds = native.get('action', {}).get('enum', [])
            if kinds == ['inspect']:
                inspect_angles = {key: native[key] for key in ('azimuth_deg', 'elevation_deg') if key in native}
            if native_finish and kinds == ['finish']:
                finish_fields.append({key: native[key] for key in ('predicted_id', 'evidence', 'unresolved')})
            limit = native.get('focus_ids', {}).get('maxItems')
            if len(kinds) == 1 and limit is not None:
                focus_limits[kinds[0]] = limit
        branches = []
        for kind in ('inspect', 'measure', 'finish'):
            branch = deepcopy(action)
            fields = branch['properties']
            fields['action'] = {'const': kind}
            if kind != 'finish':
                fields['evidence'] = {'const': []} if text_visual_reports else {**fields['evidence'], 'maxItems': 0}
                fields['predicted_id'] = {'type': 'null'}
            if text_visual_reports:
                if kind == 'inspect':
                    fields.update(context={'enum': ['scene', 'isolated']}, frame={'type': 'null'})
                    fields.update(deepcopy(inspect_angles))
                else:
                    fields.update(context={'type': 'null'}, azimuth_deg={'type': 'null'},
                                  elevation_deg={'type': 'null'})
                if kind == 'finish':
                    fields.update(focus_ids={'const': []}, frame={'type': 'null'})
                    if finish_fields:
                        for terminal in finish_fields:
                            finish = deepcopy(branch)
                            finish['properties'].update(deepcopy(terminal))
                            branches.append(finish)
                        continue
                else:
                    fields['focus_ids'] = {**ids, 'minItems': 1, 'maxItems': focus_limits.get(kind, 8)}
                    if not initial:
                        task_ids = [node['id'] for node in state['task_dag']['nodes']]
                        delta_fields = fields['monitor']['anyOf'][0]['properties']
                        delta_fields['current_node_id'] = {'enum': task_ids}
                        delta_fields['node_updates']['minItems'] = 1
                        delta_fields['node_updates']['maxItems'] = len(task_ids)
                        updates = delta_fields['node_updates']['items']['anyOf']
                        for update in updates:
                            update['properties']['id'] = {'enum': task_ids}
                        if len(task_ids) == 1:
                            delta_fields['node_updates']['items'] = updates[0]
            branches.append(branch)
        return obj({'state_version': schema['properties']['state_version'], 'action': {'anyOf': branches}})
    if isinstance(schema, list):
        return [compact_wire_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: compact_wire_schema(value) for key, value in schema.items()
              if key not in ("description", "title", "examples")}
    fields = schema.get('properties', {})
    if 'main_rivals' in fields and 'const' in fields.get('predicted_id', {}):
        result['properties']['main_rivals']['items']['properties']['object_id'] = (
            fields['main_rivals']['items']['properties']['object_id'])
    values = result.get("enum", [])
    if len(values) > 5 and all(type(value) is int for value in values):
        result.pop("enum")
        result["type"] = "integer"
    return result


def compact_catalog(state):
    """Retrieval scope only: no category truth, binding, exclusion or target choice."""
    state = deepcopy(state)
    objects = state.get('objects', [])
    vocabulary = sorted({str(row.get('category', '')) for row in objects} |
                        {h['category'] for row in objects for h in row.get('category_hypotheses', [])})
    detailed = set(state.get('active_target_ids', state.get('allowed_target_ids', [])))
    for key in ('target_category_bindings', 'reference_category_bindings'):
        for row in state.get(key, []):
            if 'id' in row: detailed.add(row['id'])
            detailed.update(candidate['id'] for candidate in row.get('candidates', []))
    for row in (state.get('task_dag') or {}).get('nodes', []):
        detailed.update(row.get('bound_object_ids', []))
    for row in state.get('previous_actions', []):
        detailed.update(row.get('focus_ids', []))
    description = state.get('description', '').lower()
    names = [str(x).lower() for x in state.get('parsed_reference_categories', [])]
    geometry = {'center_x_m', 'center_y_m', 'center_z_m', 'extent_x_m', 'extent_y_m', 'height_m', 'bottom_z_m', 'top_z_m'}
    archived = {}
    for row in objects:
        categories = [row.get('category', '')] + [h['category'] for h in row.get('category_hypotheses', [])]
        if any(c.lower() in names or re.search(r'\b' + re.escape(c.lower()) + r'(?:s|es)?\b', description) for c in categories if c):
            detailed.add(row['id'])
        if row['id'] not in detailed:
            archived[str(row['id'])] = sorted(key for key in geometry if key in row)
            for key in archived[str(row['id'])]: row.pop(key)
        if 'category' in row:
            row['category_index'] = vocabulary.index(row.pop('category'))
        hypotheses = row.get('category_hypotheses', [])
        if hypotheses and all(set(h) == {'category', 'rank', 'gap_from_top1'} for h in hypotheses):
            values = row.pop('category_hypotheses')
            if row['id'] in detailed or sorted(h['rank'] for h in values) != list(range(1, len(values) + 1)):
                row['hypotheses'] = [[vocabulary.index(h['category']), h['rank'], h['gap_from_top1']] for h in values]
            else:
                row['top5_category_indices'] = [vocabulary.index(h['category']) for h in sorted(values, key=lambda h: h['rank'])]
    state['category_vocabulary'] = vocabulary
    state['archived_catalog_geometry'] = {'ids': [int(i) for i in archived], 'possible_fields': sorted(geometry),
                                           'explicit_readback_action': 'measure'}
    return state, sorted(detailed), archived


def compact_feedback(value):
    value = deepcopy(value)
    for key in ('native_validation_errors', 'errors'):
        if key not in value or not isinstance(value[key], list): continue
        excluded = set()
        for row in value[key]:
            if not isinstance(row, dict): continue
            path, schema = row.get('path'), row.get('schema_path', [])
            # Native anyOf alternatives for a DIFFERENT explicit action/terminal
            # shape are irrelevant, not simultaneous repair requirements.
            is_status = path and path[-1] == 'status'
            if path not in (['action', 'action'], ['action', 'predicted_id']) and not is_status: continue
            if row.get('validator') not in ('enum', 'type'): continue
            if is_status:
                try:
                    proposal = json.loads(value.get('previous_raw_answer', '{}'))
                    for keypart in path: proposal = proposal[keypart]
                except (ValueError, KeyError, IndexError, TypeError): continue
                if proposal not in ('pending', 'completed'): continue
            if 'anyOf' in schema:
                index = max(i for i, part in enumerate(schema) if part == 'anyOf') if is_status else schema.index('anyOf')
                excluded.add(tuple(schema[:index + 2]))
        relevant = []
        for row in value[key]:
            schema = row.get('schema_path', []) if isinstance(row, dict) else []
            if any(tuple(schema[:len(prefix)]) == prefix for prefix in excluded): continue
            relevant.append(row)
        unique = []
        for row in relevant or value[key]:
            reduced = {k: v for k, v in row.items() if k != 'schema_path'} if isinstance(row, dict) else row
            if reduced not in unique: unique.append(reduced)
        value[key] = sorted(unique, key=lambda row: 0 if isinstance(row, dict) and
                            any(part in ('source_quote', 'source_quotes') for part in row.get('path', [])) else 1)
    return value


def compact_call(messages, state=None, *, text_visual_reports=False, visual_report_rows=False):
    """No semantic evidence selection: retain ALL object rows, raw facts and images."""
    if state is None:
        return deepcopy(messages), {"mode": "parser_verbatim", "discarded_evidence": []}
    from qwen_direct_fresh import expand_history_references, extract_state
    from eviground.grounding.dag_monitor import usable_evidence_observation
    original = expand_history_references(state)
    visible, detailed, archived = compact_catalog(original)
    history = visible.pop("previous_actions", [])
    # Action history remains on disk; last model action is exact, not a new summary.
    if history:
        visible["last_action"] = {key: value for key, value in history[-1].items() if key in
            ('action', 'focus_ids', 'context', 'azimuth_deg', 'elevation_deg', 'frame', 'predicted_id')}
    visible["archived_action_count"] = len(history)
    attachment_rows, attachments, hashes = [], [], []
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            import base64
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            sha = hashlib.sha256(raw).hexdigest()
            if sha not in hashes:
                hashes.append(sha)
                attachments.append(deepcopy(part))
            attachment_rows.append({"original_view_index": len(attachment_rows),
                "attachment_index": hashes.index(sha), "sha256": sha})
    if text_visual_reports and attachments:
        raise ValueError("Text-only grounding must not receive image attachments")
    if len(attachment_rows) != len(original.get("views", [])):
        raise ValueError("Original view metadata/image attachment count mismatch; no guessed pixel coverage")
    visible["image_delivery"] = attachment_rows
    # Exact latest native rejection plus full rollback/exclusion state; earlier
    # failures remain local but are never semantic evidence.
    observations = original.get("observations", [])
    failures = [i for i, row in enumerate(observations) if not usable_evidence_observation(row)]
    visible["observations"] = [deepcopy(row) for i, row in enumerate(observations)
        if usable_evidence_observation(row) or i == (failures[-1] if failures else -1)]
    visible["archived_rejection_count"] = max(0, len(failures) - 1)
    extras, feedback_audit = [], []
    for message_index, message in enumerate(messages):
        if message.get("role") == "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except ValueError:
            extras.append(content)
            continue
        if isinstance(value, dict) and "candidate_contract" in value:
            value = deepcopy(value)
            # Native state already contains these exact snapshots; keep guidance.
            contract = value["candidate_contract"]
            value['candidate_contract'] = {k: contract[k] for k in (
                'legal_current_prediction_ids', 'accepted_native_revision', 'latest_native_rollback_verbatim') if k in contract}
        reduced = compact_feedback(value) if isinstance(value, dict) else value
        if isinstance(value, dict):
            for field in ('native_validation_errors', 'errors'):
                if field in value:
                    feedback_audit.append({'original_message_index': message_index, 'field': field,
                        'original_count': len(value[field]), 'visible_count': len(reduced[field]),
                        'original_message_sha256': hashlib.sha256(content.encode()).hexdigest(),
                        'rule': 'Different explicit-action/terminal anyOf branches removed; remaining exact path/validator/error values deduplicated, full schema_paths archived in original-request-redacted.json'})
        extras.append(reduced)
    wire_state = pack_measurements(visible)
    if text_visual_reports:
        # Group the ID-keyed catalog by columns so sparse rows share a table.
        groups = {}
        for row in wire_state.pop('objects', []):
            groups.setdefault(tuple(row), []).append(row)
        wire_state['object_catalog_tables'] = list(groups.values())
    if visual_report_rows:
        wire_state['visual_report_rows'] = []
        for report in wire_state.pop('visual_reports'):
            findings = report['findings']
            ranks = {finding['finding_id']: i for i, finding in enumerate(findings, 1)}
            number = int(report['report_id'].split('_')[-1])
            assert all(finding_id == f'visual_{number:04d}_f{rank:02d}' for finding_id, rank in ranks.items())
            rows = [[int(f['image_id'].split('_')[-1]), f['object_ids'], int(f['status'] == 'observed'),
                     f['observation']] for f in findings]
            assessments = [[group['assessment'], [ranks[fid] for fid in group['finding_ids']]]
                           for group in report['groups']]
            wire_state['visual_report_rows'].append([number, rows, assessments])
        for image in wire_state['image_catalog']:
            image.pop('sha256')
    encoded = encode_tables({'state': wire_state, 'context_and_exact_feedback': extras},
                            constants=text_visual_reports)
    decoded = unpack_measurements(decode_tables(encoded)['state'])
    if text_visual_reports:
        objects = {row['id']: row for group in decoded.pop('object_catalog_tables') for row in group}
        decoded['objects'] = [objects[row['id']] for row in visible.get('objects', [])]
        if 'objects' not in visible:
            decoded.pop('objects')
    if visual_report_rows:
        image_evidence = {row['image_id']: row['evidence_id'] for row in decoded['image_catalog']}
        restored = []
        for number, rows, assessments in decoded.pop('visual_report_rows'):
            report_id = f'visual_{number:04d}'
            findings = []
            for rank, (image_number, ids, observed, observation) in enumerate(rows, 1):
                image_id = f'img_{image_number:04d}'
                findings.append({'finding_id': f'{report_id}_f{rank:02d}', 'image_id': image_id,
                                 'evidence_id': image_evidence[image_id], 'object_ids': ids,
                                 'status': 'observed' if observed else 'unknown', 'observation': observation})
            groups = [{'assessment': status, 'finding_ids': [f'{report_id}_f{rank:02d}' for rank in ranks]}
                      for status, ranks in assessments]
            restored.append({'report_id': report_id, 'findings': findings, 'groups': groups})
        assert restored == [{**{key: report[key] for key in ('report_id', 'groups')},
                             'findings': [{k: v for k, v in f.items() if k != 'question'}
                                          for f in report['findings']]} for report in visible['visual_reports']]
        decoded['visual_reports'] = visible['visual_reports']
        decoded['image_catalog'] = visible['image_catalog']
    assert decoded == visible
    tentative = [m['content'] for m in messages if m.get('role') == 'system' and
                 isinstance(m.get('content'), str) and 'performing ONE final tentative evaluation' in m['content']]
    rules = RULES
    if text_visual_reports:
        from qwen_visual_reports import TEXT_RULES
        rules = REPORT_PLANNING_RULES + ' Return one JSON object per Schema. '
        if visual_report_rows:
            rules += ('visual_report_rows: [report_number, findings, groups]. Finding: [image_number, object_ids, '
                      '1 observed or 0 unknown, exact observation]. Group: [assessment, one-based finding numbers]. '
                      'Report 1 is visual_0001, its first finding visual_0001_f01; image 1 is img_0001. '
                      'Find the native evidence ID in image_catalog. ')
            if original.get('task_dag') and not tentative:
                rules += ('If replanning, replace_plan must contain nodes, verification, requirement_nodes, '
                          'current_node_id, reopen_node_ids and revision_reason. Use the same node fields as task_dag. ')
        tentative = [text.replace('raw evidence and actual attached images', 'raw measurements and text visual reports')
                     for text in tentative]
        rules += TEXT_RULES
    result = [{"role": "system", "content": rules + '\n'.join(tentative)}, {"role": "user", "content": [
        {"type": "text", "text": packed(encoded)}, *attachments]}]
    if not tentative and not text_visual_reports:
        shape = {'node_fields': ['id', 'goal', 'kind', 'source_quotes', 'depends_on', 'allowed_actions', 'status',
                                'bound_object_ids', 'evidence_ids', 'conclusion', 'excluded_target_ids'],
                 'full_compact_plan_fields': ['nodes', 'verification', 'requirement_nodes', 'current_node_id', 'reopen_node_ids', 'revision_reason'],
                 'initial_monitor': 'FULL_COMPACT_PLAN',
                 'later_monitor_alternatives': [{'node_updates': ['id', 'status', 'bound_object_ids', 'evidence_ids', 'conclusion', 'excluded_target_ids'],
                                                 'current_node_id': 'string', 'reopen_node_ids': 'string[]', 'revision_reason': 'string'},
                                                {'replace_plan': 'FULL_COMPACT_PLAN'}],
                 'rule': 'nodes is task array; verification is n11 node; requirement_nodes maps EVERY qID to node ID array. Delta node_updates is array of complete node-state objects. First update is pending current tool node, or completed n11 on finish. Pending evidence/exclusions empty, conclusion empty. Node kind=binding|attribute|relation|comparison; n11 kind=verification. Lists of IDs are sets.'}
        if not original.get('task_dag'):
            shape.pop('later_monitor_alternatives')
            shape['initial_rule'] = 'NO accepted DAG exists. monitor MUST be the bare FULL_COMPACT_PLAN with all six top-level fields. NEVER wrap it in replace_plan or node_updates on this first proposal.'
        result[1]['content'].append({'type': 'text', 'text': packed({'exact_monitor_field_contract': shape})})
    if not text_visual_reports:
        from eviground.grounding.dag_monitor import exact_source_quotes
        result[1]['content'].append({'type': 'text', 'text': packed({
            'description_verbatim_including_literal_quote_characters': original.get('description'),
            'allowed_monitor_source_quotes': exact_source_quotes(original.get('description', ''), original.get('monitor_requirements', [])),
            'quote_rule': 'Choose an exact listed string. Do NOT strip outer ASCII single quotes, normalize or repair the description. Tool inspect/measure evidence must be [].'})})
    # This field is a digest of the FULL native state; compaction never changes it.
    return result, {"mode": "lossless_evidence_tables_and_pair_frame_catalogs",
        "all_object_ids": [row["id"] for row in original.get("objects", [])],
        "all_active_target_ids": original.get("active_target_ids", []),
        "all_raw_evidence_ids": [row['evidence_id'] for row in observations if usable_evidence_observation(row)],
        "original_images": len(attachment_rows), "delivered_images": len(attachments),
        "image_delivery": attachment_rows, "discarded_evidence": [],
        "detailed_geometry_object_ids": detailed, "archived_geometry_fields_by_id": archived,
        "archived_classifier_gap_object_ids": [row['id'] for row in visible.get('objects', []) if 'top5_category_indices' in row],
        "feedback_projection": feedback_audit,
        "catalog_geometry_readback": "All object IDs remain known; explicitly measure any ID group. Previously acquired exact raw measurement rows and frames remain in full.",
        "archived_actions": len(history), "archived_rejections": max(0, len(failures) - 1),
        "native_state_unchanged": state == extract_state(messages)}


TIGHT_RULES = ('\nOverflow encoding only: table c=[[column_index,value],...] supplies exact constant columns; '
               'row r supplies the other columns in order. Reinsert state.object_constants into EVERY objects row. '
               'previous_proposal is parsed previous_raw_answer JSON, not new reasoning; error_value_path points '
               'to its exact invalid field, and rule_suffix completes the original error. Full raw text is archived.\n')


def tighten_overflow_metadata(messages):
    """Factor repeated carriers/proposal values, never physical measurements."""
    result = deepcopy(messages)
    audit = {'object_constant_fields': [], 'parsed_previous_proposals': 0, 'factored_errors': 0}
    changed = False
    from qwen_direct_fresh import unique_json_object
    for message in result:
        if not isinstance(message.get('content'), list): continue
        for part in message['content']:
            if part.get('type') != 'text': continue
            try: packet = json.loads(part['text'])
            except ValueError: continue
            if not isinstance(packet, dict) or not {'table_columns', 'data'} <= set(packet): continue
            value = decode_tables(packet)
            before = deepcopy(value)
            state = value.get('state', {})
            objects = state.get('objects', [])
            defaults = {}
            for field in ('category_score_type',):
                if objects and all(field in row and packed(row[field]) == packed(objects[0][field]) for row in objects):
                    defaults[field] = objects[0][field]
                    for row in objects: row.pop(field)
            if defaults:
                state['object_constants'] = defaults
                audit['object_constant_fields'] = list(defaults)
            raw_proposals = []
            for feedback in value.get('context_and_exact_feedback', []):
                if not isinstance(feedback, dict) or not isinstance(feedback.get('previous_raw_answer'), str): continue
                raw = feedback['previous_raw_answer']
                try: proposal = json.loads(raw, object_pairs_hook=unique_json_object)
                except ValueError: continue
                raw_proposals.append((feedback, raw))
                feedback['previous_proposal'] = proposal
                feedback.pop('previous_raw_answer')
                audit['parsed_previous_proposals'] += 1
                for name in ('native_validation_errors', 'errors'):
                    for error in feedback.get(name, []):
                        if not isinstance(error, dict) or not isinstance(error.get('error'), str): continue
                        try:
                            actual = proposal
                            for key in error['path']: actual = actual[key]
                        except (KeyError, IndexError, TypeError): continue
                        prefix = repr(actual)
                        if error['error'].startswith(prefix):
                            original = error.pop('error')
                            error['error_value_path'] = deepcopy(error['path'])
                            error['rule_suffix'] = original[len(prefix):]
                            audit['factored_errors'] += 1
            encoded = encode_tables(value, constants=True)
            # Verify exact semantic round trip, restoring raw proposal text and
            # validator messages, before selecting this representation.
            restored = decode_tables(encoded)
            for row in restored.get('state', {}).get('objects', []): row.update(defaults)
            restored.get('state', {}).pop('object_constants', None)
            originals = iter(raw for _, raw in raw_proposals)
            for feedback in restored.get('context_and_exact_feedback', []):
                if not isinstance(feedback, dict) or 'previous_proposal' not in feedback: continue
                proposal = feedback.pop('previous_proposal')
                feedback['previous_raw_answer'] = next(originals)
                for name in ('native_validation_errors', 'errors'):
                    for error in feedback.get(name, []):
                        if not isinstance(error, dict) or 'error_value_path' not in error: continue
                        actual = proposal
                        for key in error.pop('error_value_path'): actual = actual[key]
                        error['error'] = repr(actual) + error.pop('rule_suffix')
            assert restored == before, 'Overflow metadata packing changed facts'
            part['text'] = packed(encoded)
            changed = True
    if changed:
        if result[0].get('role') == 'system' and isinstance(result[0].get('content'), str):
            result[0]['content'] += TIGHT_RULES
        else:
            result.insert(0, {'role': 'system', 'content': TIGHT_RULES})
    audit['packet_found'] = changed
    return result, audit


def resize_overflow_images(messages, max_side):
    """Derived send-side PNGs, always resized from the full-resolution input."""
    result, transforms = deepcopy(messages), []
    for message in result:
        if not isinstance(message.get('content'), list): continue
        for part in message['content']:
            if part.get('type') != 'image_url': continue
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                width, height = source.size
                scale = min(1.0, max_side / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                if size != source.size:
                    stream = io.BytesIO()
                    source.resize(size, Image.Resampling.LANCZOS).save(stream, format='PNG')
                    sent = stream.getvalue()
                    part['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(sent).decode()
                else:
                    sent = raw
            transforms.append({'attachment_index': len(transforms),
                'original_sha256': hashlib.sha256(raw).hexdigest(), 'sent_sha256': hashlib.sha256(sent).hexdigest(),
                'original_size': [width, height], 'sent_size': list(size),
                'scale_x': size[0] / width, 'scale_y': size[1] / height,
                'changed': size != (width, height), 'lossy_downsampling': size != (width, height),
                'original_bytes': len(raw), 'sent_bytes': len(sent)})
    changed = [row for row in transforms if row['changed']]
    if changed:
        notice = {'sent_image_resolution': [[r['attachment_index'], *r['original_size'], *r['sent_size']] for r in transforms],
            'columns': ['attachment_index', 'archived_width', 'archived_height', 'sent_width', 'sent_height'],
            'warning': 'LOSSY downsampling. You see ONLY sent-resolution pixels. Hashes and visibility/pixel counts in original view/evidence metadata describe archived originals. Do not claim unseen original detail, material certainty or absence from lost pixels; acquire better evidence or retain uncertainty/null. Full hash/scale mappings are logged.'}
        if isinstance(result[-1].get('content'), list):
            result[-1]['content'].append({'type': 'text', 'text': packed(notice)})
        else:
            result.append({'role': 'user', 'content': [{'type': 'text', 'text': packed(notice)}]})
    return result, transforms


def adapt_overflow(messages, response_format, reserve, packing, first_preflight, preflight_fn, limit):
    """Return first fitting package, or the smallest explicitly refused attempt."""
    audit = deepcopy(packing)
    attempts = []
    def record(stage, candidate, preflight, transforms=None, metadata=None):
        attempts.append({'stage': stage, 'maximum': preflight['maximum'], 'text_bytes': preflight['text_bytes'],
            'image_tokens_upper_bound': preflight['image_tokens_upper_bound'], 'output_reserved': preflight['output_reserved'],
            'fits': preflight['maximum'] < limit, 'image_derivations': transforms or [], 'metadata': metadata or {}})
        return candidate, preflight
    best, bound = record('initial', messages, first_preflight)
    selected = 0
    if bound['maximum'] >= limit:
        tight, metadata = tighten_overflow_metadata(messages)
        tight_bound = preflight_fn(tight, response_format, reserve)
        record('metadata', tight, tight_bound, metadata=metadata)
        if tight_bound['maximum'] < bound['maximum']:
            best, bound, selected = tight, tight_bound, 1
        if bound['maximum'] >= limit:
            base = best
            for side in (896, 768, 640, 512):
                candidate, transforms = resize_overflow_images(base, side)
                if not any(row['changed'] for row in transforms): continue
                current = preflight_fn(candidate, response_format, reserve)
                record('images_max_side_' + str(side), candidate, current, transforms=transforms)
                if current['maximum'] < bound['maximum']:
                    best, bound, selected = candidate, current, len(attempts) - 1
                if bound['maximum'] < limit: break
    audit['adaptive_attempts'] = attempts
    audit['adaptive_selected_attempt'] = selected
    audit['adaptive_selected_stage'] = attempts[selected]['stage']
    audit['image_derivations'] = attempts[selected]['image_derivations']
    audit['sent_original_resolution'] = not any(row['changed'] for row in audit['image_derivations'])
    audit['visual_readability_verified'] = False
    audit['native_monitor_mutated_by_adaptation'] = False
    return best, audit, bound


def original_resolution_pack(messages, response_format, reserve, packing, first_preflight, preflight_fn, limit):
    """Production selector route: reversible text only, NEVER resize pixels."""
    audit = deepcopy(packing)
    attempts = [{'stage': 'initial', 'maximum': first_preflight['maximum']}]
    best, bound, selected = messages, first_preflight, 0
    if bound['maximum'] >= limit:
        tight, metadata = tighten_overflow_metadata(messages)
        current = preflight_fn(tight, response_format, reserve)
        attempts.append({'stage': 'metadata', 'maximum': current['maximum'], 'metadata': metadata})
        if current['maximum'] < bound['maximum']:
            best, bound, selected = tight, current, 1
    audit.update(adaptive_attempts=attempts, adaptive_selected_attempt=selected,
                 adaptive_selected_stage=attempts[selected]['stage'], image_derivations=[],
                 sent_original_resolution=True, production_resize_fallback=False)
    return best, audit, bound
