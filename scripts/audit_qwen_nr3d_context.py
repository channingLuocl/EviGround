"""Read-only audit of the frozen Qwen NR3D request and evaluation records."""
import argparse
import base64
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from tempfile import TemporaryDirectory


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, source, message):
    if not condition:
        raise ValueError(f'{source}: {message}')


def audit_requests(episode, frozen, levels, stats, roles):
    seen = set()
    sent = 0
    allowed_roles = {'parser', 'parser_repair', 'semantic_consultant', 'visual_worker'}
    if levels == 3:
        allowed_roles.add('visual_coordinator')
    for path in sorted((episode/'calls').glob('*/receipt.json')):
        receipt = read(path)
        index, role = receipt['index'], receipt['role']
        require(index not in seen, path, 'duplicate call index')
        seen.add(index)
        if receipt['status'] == 'rejected_preflight':
            require(not receipt['http_sent'] and (receipt['preflight']['maximum'] >= frozen.OUTPUT_LIMIT
                    or receipt.get('server_prompt_count', 0) + receipt['output_reserved'] >= frozen.OUTPUT_LIMIT),
                    path, 'invalid capacity refusal')
            stats['preflight_refusals'] += 1
        if not receipt['http_sent']:
            continue
        sent += 1
        require(role in allowed_roles, path, f'unexpected role {role}')
        roles[role] += 1
        payload = read(path.parent/'request-redacted.json')
        require(payload['model'] == receipt['requested_model'] == frozen.MODEL, path, 'unexpected request model')
        parts = [part for msg in payload['messages'] if isinstance(msg['content'], list)
                 for part in msg['content'] if part['type'] == 'image_url']
        images = receipt['input_images']
        require(len(parts) == len(images) == int(role == 'visual_worker'), path, 'image delivery violates role policy')
        for part, img in zip(parts, images):
            raw = (path.parent/img['path']).read_bytes()
            original = {entry['sha256'] for entry in receipt['original_input_images']}
            require(hashlib.sha256(raw).hexdigest() == img['sha256'] and img['sha256'] in original,
                    path, 'delivered image differs from original')
            part['image_url']['url'] = f"data:{img['mime']};base64," + base64.b64encode(raw).decode('ascii')
        actual = frozen.qwen_preflight(payload['messages'], payload.get('response_format'), payload['max_tokens'])
        recorded = receipt['preflight']
        require(recorded['enforced'] and recorded['limit'] == frozen.OUTPUT_LIMIT
                and all(recorded[key] == actual[key] for key in ('maximum', 'images', 'text_bytes', 'output_reserved'))
                and actual['maximum'] < frozen.OUTPUT_LIMIT, path, 'reconstructed request fails the recorded context gate')
        stats['max_sent_context_bound'] = max(stats['max_sent_context_bound'], actual['maximum'])
        glm_text = frozen.MODEL == 'GLM-4.5V' and not images
        if glm_text:
            count = receipt.get('server_prompt_count')
            require(type(count) is int and 0 < count < frozen.OUTPUT_LIMIT - actual['output_reserved'],
                    path, 'missing or over-capacity vLLM text tokenization')
            stats['server_text_preflights'] += 1
        if receipt.get('http_status') != 200:
            stats['pending_or_unknown_http' if receipt.get('http_status') is None else 'http_errors'] += 1
            continue
        provider = read(path.parent/'provider-response.json')
        usage = receipt['usage']
        require(provider['model'] == frozen.MODEL and usage == provider['usage']
                and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')),
                path, 'missing or inconsistent provider model or token usage')
        prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
        reserve = actual['output_reserved']
        require(prompt <= actual['maximum'] - reserve and prompt + reserve < frozen.OUTPUT_LIMIT
                and completion <= reserve, path, 'provider usage exceeds the reconstructed reservation')
        if glm_text:
            require(prompt == count, path, 'vLLM prompt usage differs from pre-send text tokenization')
        stats['max_reported_context_used'] = max(stats['max_reported_context_used'], prompt + completion)
        stats['provider_usage_verified'] += 1
    require(seen == set(range(1, len(seen) + 1)), episode, 'missing call indices')
    return sent


def audit_visual_batches(episode, levels, stats):
    registry_path = episode/'visual-reports/registry/images.json'
    paths = list((episode/'visual-reports').glob('visual_*/report.json'))
    if not paths:
        return
    images = {row['image_id']: row for row in read(registry_path)}
    calls, repairs = {}, {}
    for path in (episode/'calls').glob('*/receipt.json'):
        receipt = read(path)
        if receipt['role'] in ('visual_coordinator', 'visual_worker'):
            tag = receipt['tag']
            table = repairs if tag.endswith('-format-repair') else calls
            original_tag = tag.removesuffix('-format-repair')
            require(original_tag not in table, path, 'duplicate visual call tag')
            table[original_tag] = (path.parent, receipt)
    for tag, (folder, repair) in repairs.items():
        require(tag in calls, folder, 'visual format repair has no original call')
        original_folder, original = calls[tag]
        original_request = read(original_folder/'original-request-redacted.json')
        repaired_request = read(folder/'original-request-redacted.json')
        require(original['index'] + 1 == repair['index'] and original['role'] == repair['role']
                and original['status'] == 'failed' and original.get('http_status') == 200
                and original['error_type'] in ('ValidationError', 'JSONDecodeError', 'ValueError', 'EvidenceSelectionError')
                and repaired_request[:-1] == original_request and repaired_request[-1]['role'] == 'user'
                and read(original_folder/'original-response-schema.json') == read(folder/'original-response-schema.json'),
                folder, 'visual format repair must retain its failed original task and schema')
        if repair['status'] == 'completed':
            calls[tag] = (folder, repair)
    for path in paths:
        report = read(path)
        batch = report['report_id']
        plan = read(path.parent/'plan.json')
        if levels == 3:
            folder, receipt = calls[batch + '-plan']
            require(receipt['role'] == 'visual_coordinator' and receipt['status'] == 'completed'
                    and read(folder/'response.json') == plan['plan'],
                    path, 'worker plan does not match coordinator response')
        findings = []
        for task in plan['plan']['tasks']:
            image_id = task['image_id']
            dispatched = {**task, 'object_ids': list(dict.fromkeys(task['object_ids']))}
            folder, receipt = calls[f'{batch}-{image_id}']
            payload = read(folder/'request-redacted.json')
            request = json.loads(payload['messages'][1]['content'][0]['text'])
            image = images[image_id]
            require(receipt['role'] == 'visual_worker' and receipt['status'] == 'completed'
                    and receipt['input_images'][0]['sha256'] == image['sha256']
                    and request['task'] == dispatched and request['evidence_id'] == image['evidence_id'],
                    path, 'worker did not receive its assigned task and registered original')
            answer = read(folder/'response.json')
            require(answer['image_id'] == image_id, path, 'worker response uses another image ID')
            findings.extend(dict(row, image_id=image_id, image_sha256=image['sha256'], question=task['question'],
                                 evidence_id=image['evidence_id']) for row in answer['findings'])
        actual = [{key: value for key, value in row.items() if key != 'finding_id'} for row in report['findings']]
        require(findings == actual, path, 'worker findings changed in the visual report')
        if levels == 3:
            folder, receipt = calls[batch + '-review']
            payload = read(folder/'request-redacted.json')
            request = json.loads(payload['messages'][1]['content'])
            expected = [{key: value for key, value in row.items() if key not in ('image_sha256', 'evidence_id', 'question')}
                        for row in report['findings']]
            require(receipt['role'] == 'visual_coordinator' and receipt['status'] == 'completed'
                    and request['findings'] == expected,
                    path, 'coordinator review did not receive the exact worker findings')
        stats['visual_batches_verified'] += 1


def audit_readbacks(episode, frozen, stats):
    from qwen_step2_context import decode_tables, unpack_measurements

    actions = {}
    for path in (episode/'calls').glob('*-semantic_consultant/receipt.json'):
        receipt = read(path)
        actions[receipt['index']] = (path.parent, receipt)
    indices = sorted(actions)
    for path in episode.glob('controller-contracts/t*/evidence-readback-*.json'):
        page = read(path)
        source, selected = page['source_id'], page['source_call_index']
        folder, receipt = actions[selected]
        answer = read(folder/'response.json')['action']
        require(receipt['status'] == 'completed' and answer == {'read_evidence_page': source}
                and source in read(folder/'context-packing.json')['readback_ids'],
                path, 'readback does not match a model-selected archived page')
        following = bisect_right(indices, selected)
        if following == len(indices):
            stats['readbacks_pending'] += 1
            continue
        folder, receipt = actions[indices[following]]
        if not receipt['http_sent']:
            stats['readbacks_unsent'] += 1
            continue
        require(receipt['tag'].split('-')[0] == path.parent.name, path, 'readback crossed decision turns')
        payload = read(folder/'request-redacted.json')
        wire = decode_tables(json.loads(payload['messages'][1]['content'][0]['text']))['state']
        if source.startswith('observation_'):
            full = frozen.expand_history_references(read(path.parent/'full-state.json'))
            wanted = [row for row in full['observations'] if row.get('evidence_id') == source]
            actual = [row for row in unpack_measurements(wire)['observations'] if row.get('evidence_id') == source]
            require(len(wanted) == 1 and wanted[0]['tool'] == 'measure' and actual == wanted,
                    path, 'original measurement missing or changed in the next request')
        else:
            report = read(episode/'visual-reports'/source/'report.json')
            require(report in read(folder/'context-packing.json')['delivered_visual_reports'],
                    path, 'original visual report missing or changed in delivery receipt')
            if 'visual_reports' in wire:
                expected = dict(report, findings=[{key: value for key, value in finding.items()
                    if key != 'image_sha256'} for finding in report['findings']])
                require(expected in wire['visual_reports'], path, 'visual report missing or changed on the wire')
            else:
                findings = report['findings']
                lookup = {finding['finding_id']: index for index, finding in enumerate(findings, 1)}
                rows = [[int(f['image_id'].split('_')[-1]), f['object_ids'],
                         int(f['status'] == 'observed'), f['observation']] for f in findings]
                groups = [[g['assessment'], [lookup[key] for key in g['finding_ids']]] for g in report['groups']]
                require([int(source.split('_')[-1]), rows, groups] in wire['visual_report_rows'],
                        path, 'compact visual report missing or changed on the wire')
        stats['readbacks_verified'] += 1


def iter_episodes(root, archives, reuse=None):
    missing = {}
    for sample in sorted(root.glob('q[0-9]*')):
        episode = sample/'episode'
        if reuse and reuse.get('continuation') and str(int(sample.name[1:])) in reuse['evaluations']:
            from qwen_direct_batch import reused_episode_path
            require(not episode.exists(), sample, 'continued sample has a new local episode')
            episode = reused_episode_path(reuse, int(sample.name[1:]))
            require(episode.is_dir(), episode, 'continued episode unavailable at its original path')
            yield sample, episode
            continue
        if archives and not episode.exists() and (sample/'evaluation.json').is_file():
            missing[sample.name] = sample
        else:
            yield sample, episode

    for path in archives:
        with TemporaryDirectory(prefix='nr3d-audit-', dir=root.parent) as scratch:
            archived_samples = set()
            members_seen = set()
            with tarfile.open(path, mode='r|*') as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    require(len(parts) >= 2 and parts[0] in missing and parts[1] == 'episode'
                            and '..' not in parts and parts not in members_seen
                            and (member.isfile() or member.isdir()),
                            path, f'invalid or duplicate archived episode member: {member.name}')
                    members_seen.add(parts)
                    archived_samples.add(parts[0])
                    archive.extract(member, scratch, filter='data')
            require(archived_samples, path, 'archive contains no episodes')
            for name in sorted(archived_samples):
                episode = Path(scratch)/name/'episode'
                require(episode.is_dir(), path, f'missing episode directory for {name}')
                yield missing.pop(name), episode

    for sample in missing.values():
        yield sample, sample/'episode'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--episode-archive', type=Path, action='append', default=[],
                        help='Tar archive containing only qNNNNN/episode trees for finalized, offloaded samples. Repeat per archive.')
    args = parser.parse_args()
    root = args.output.resolve()
    manifest_path = root/'manifest.json'
    manifest, summary = read(manifest_path), read(root/'summary.json')
    manifest_hash = sha(manifest_path)
    for name, expected in manifest['source_closure'].items():
        require(sha(root/'source'/name) == expected, name, 'frozen source hash changed')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root/'source/scripts'))
    import qwen_direct_fresh as frozen

    protocol = manifest['protocol']
    levels = protocol['visual_levels']
    if protocol['model'] == 'GLM-4.5V':
        frozen.configure_glm(protocol['endpoint'])
    elif protocol['model'] == 'glm-4.5v':
        frozen.configure_bigmodel()
    expected_limit = {'qwen3.8-max': 32768, 'GLM-4.5V': 65536, 'glm-4.5v': 65536}.get(protocol['model'])
    require(levels in (2, 3) and protocol['model'] == frozen.MODEL
            and protocol['endpoint'] == frozen.BASE_URL and frozen.OUTPUT_LIMIT == expected_limit
            and protocol['qwen_context_policy']['reference_bound'] == expected_limit,
            manifest_path, 'unexpected model, endpoint, role policy, or window')
    source = root/'parent-evaluation/selection.json'
    require(sha(source) == manifest['selection_sha256'], source, 'frozen selection hash changed')
    chosen = manifest['sample_rows']
    annotation_rows = read(source)['rows']
    source_rows = {row['csv_row']: row for row in annotation_rows}
    require(len(source_rows) == len(annotation_rows) == 7805, source, 'invalid complete Nr3D selection')
    selected = set(source_rows) if chosen is None else set(chosen)
    selection = read(root/'inference-selection.json')
    wanted = [{k: row[k] for k in ('scene', 'csv_row', 'query')} for row in source_rows.values()
              if row['csv_row'] in selected]
    require(selection == {'rows': wanted, 'GT_fields': []} and len(wanted) == manifest['total'],
            root/'inference-selection.json', 'inference selection or annotation isolation check failed')
    reuse = manifest.get('reused_results')
    reused = reuse['evaluations'] if reuse else {}
    if reuse:
        reused_manifest = root/'reused-manifest.json'
        archived_rows = read(reused_manifest)['sample_rows']
        right_scope = (archived_rows is None if reuse.get('correct_only') or reuse.get('continuation')
                       else set(reused) == set(map(str, archived_rows or [])))
        require(chosen is None and sha(reused_manifest) == reuse['manifest_sha256'] and right_scope
                and set(map(int, reused)) <= selected, reused_manifest, 'reused subset provenance changed')
        if reuse.get('correct_only') or reuse.get('continuation'):
            from qwen_direct_batch import reuse_manifest_path
            worker_manifests = reuse['worker_manifests']
            require(set(worker_manifests) <= set(reused), reused_manifest, 'unexpected inherited rows')
            for digest in set(worker_manifests.values()):
                require(len(digest) == 64 and all(char in '0123456789abcdef' for char in digest),
                        reused_manifest, 'invalid worker manifest digest')
                archived = reuse_manifest_path(root, reuse, digest)
                require(sha(archived) == digest, archived, 'inherited worker manifest changed')
    stats, roles, evaluated = Counter(), Counter(), {}
    for sample, episode in iter_episodes(root, args.episode_archive, reuse):
        row_id = int(sample.name[1:])
        require(row_id in selected, sample, 'sample outside frozen selection')
        reused_hash = reused.get(str(row_id))
        sample_manifest_hash = (reuse.get('worker_manifests', {}).get(str(row_id), reuse['manifest_sha256'])
                                if reused_hash else manifest_hash)
        record = episode/'frozen-manifest.json'
        if record.exists():
            context = read(record)
            require(context['shared_manifest_sha256'] == sample_manifest_hash
                    and context['sample'] == {k: source_rows[row_id][k] for k in ('scene', 'csv_row', 'query')},
                    record, 'worker input or manifest identity changed')
        sent = audit_requests(episode, frozen, levels, stats, roles)
        audit_visual_batches(episode, levels, stats)
        audit_readbacks(episode, frozen, stats)
        evaluation = sample/'evaluation.json'
        if reused_hash:
            require(record.exists() and evaluation.exists() and sha(evaluation) == reused_hash,
                    sample, 'reused evaluation or original worker record changed')
        if not evaluation.exists():
            continue
        value = read(evaluation)
        result_path = sample/'result.json'
        result = read(result_path)
        freeze_path = sample/'prediction-freeze.json'
        target = int(source_rows[row_id]['target_id'])
        prediction_hash = sha(result_path)
        correct = result['status'] in ('completed', 'tentative', 'abstained', 'tentative_abstained') and (
            result['predicted_id'] == target)
        if reused_hash and reuse.get('correct_only'):
            require(correct, evaluation, 'correct-only reuse contains a wrong prediction')
        require(read(freeze_path) == {'result_sha256': prediction_hash,
                'sample': {key: source_rows[row_id][key] for key in ('scene', 'csv_row', 'query')},
                'shared_manifest_sha256': sample_manifest_hash, 'annotation_read': False},
                freeze_path, 'prediction freeze no longer matches sample and result')
        require(value['shared_manifest_sha256'] == sample_manifest_hash and value['csv_row'] == row_id
                and value['target_id'] == target and value['correct'] is correct
                and value['predicted_id'] == result['predicted_id'] and value['status'] == result['status']
                and value['strict'] == result.get('strict_monitor_accepted', False)
                and value['terminal_mode'] == result.get('terminal_mode', 'failed')
                and value['usage']['client_attempts'] == sent
                and prediction_hash == value['frozen_result_sha256']
                and sha(freeze_path) == value['prediction_freeze_sha256'],
                evaluation, 'evaluation differs from the frozen result or annotation')
        evaluated[row_id] = value
    if args.require_complete:
        from qwen_direct_batch import totals
        require(summary['state'] == 'completed' and evaluated.keys() == selected
                and sum(roles.values()) > 0 and stats['pending_or_unknown_http'] == 0,
                root, 'run or request accounting is incomplete')
        ledger = read(root/'results.json')
        require(ledger == {'summary': summary, 'results': [evaluated[row['csv_row']] for row in wanted]},
                root/'results.json', 'terminal results differ from per-sample evaluations')
        if reuse and (reuse.get('correct_only') or reuse.get('continuation')):
            recomputed = totals(list(evaluated.values()), len(selected), 'completed', reuse)
        else:
            recomputed = totals(list(evaluated.values()), len(selected), 'completed')
        require(all(summary[key] == value for key, value in recomputed.items() if key != 'updated_utc'),
                root/'summary.json', 'terminal summary differs from per-sample evaluations')
        if reuse:
            require(summary['reused'] == len(reused) and all(summary[f'reused_{key}'] ==
                    sum(evaluated[int(row)]['usage'][key] for row in reused)
                    for key in ('client_attempts', 'known_prompt_tokens', 'known_completion_tokens')),
                    root/'summary.json', 'reused call accounting differs from original evaluations')
    print(json.dumps({'state': summary['state'], 'levels': levels, 'samples_evaluated': len(evaluated),
                      'samples_selected': len(selected), 'requests_sent': sum(roles.values()), 'roles': dict(roles),
                      **{k: stats[k] for k in ('preflight_refusals', 'pending_or_unknown_http', 'http_errors',
                                              'visual_batches_verified', 'readbacks_verified', 'readbacks_unsent', 'readbacks_pending',
                                              'max_sent_context_bound', 'max_reported_context_used')},
                      **dict(stats)}, sort_keys=True))


if __name__ == '__main__':
    main()
