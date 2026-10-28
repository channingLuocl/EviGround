"""Recover an empty category pool using only classifier-covered instances."""


def recover_candidates(candidates, objects):
    original = [obj['bbox_id'] for obj in candidates]
    recovered = list(candidates) if candidates else list(objects)
    audit = {
        'policy': 'empty_pool_visual_recall_v1',
        'fallback_applied': not bool(candidates) and bool(objects),
        'original_candidate_ids': original,
        'effective_candidate_ids': [obj['bbox_id'] for obj in recovered],
        'source': 'category_match' if candidates else 'classifier_covered_scene_instances',
        'ground_truth_used_for_recall': False,
        'reason': None if candidates else 'no retained category hypothesis matches the parsed target',
    }
    return recovered, audit


def null_reason(prediction):
    if prediction.get('error'):
        return 'runtime_or_api_error'
    if prediction.get('predicted_id') is not None:
        return None
    if prediction.get('active_trace', {}).get('stop_reason') == 'category_gate_abstention':
        return 'empty_candidate_pool'
    return 'model_abstention'
