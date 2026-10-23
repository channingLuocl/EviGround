"""Training-free inference over a public vocabulary; no learned head or part states."""
import numpy as np

from seeground.categories.category_common_v82 import category_vocabulary
from seeground.classifiers.pointcloud_classifier_v82 import PointCloudClassifier
from seeground.classifiers.pointcloud_classifier_v83 import FrozenFeatures


VERSION = 'object-category-v8.3-zero-shot-r1'
SAMPLING_OFFSET = 104729
LOCAL_WEIGHT = .1  # Fixed research default, not selected with evaluation labels.


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError('Expected finite nonzero semantic features')
    return values / norms


def fuse_zero_shot(first, second, text):
    """Positive local support; inconsistent/missing support never proves absence."""
    text = normalize(text)
    global_a, global_b = normalize(first['global_features']), normalize(second['global_features'])
    local_a, local_b = normalize(first['local_features']), normalize(second['local_features'])
    geometry = np.asarray(first['geometry'])
    if (text.ndim != 2 or len(text) < 2 or global_a.shape != (text.shape[1],)
            or global_b.shape != global_a.shape or local_a.ndim != 2 or local_a.shape != local_b.shape
            or local_a.shape[1] != text.shape[1] or not len(local_a)
            or geometry.shape != (len(local_a), 7) or not np.isfinite(geometry).all()
            or not np.array_equal(geometry, second['geometry'])):
        raise ValueError('Mismatched paired global/local features or crop layout')
    baseline = global_a @ text.T
    global_scores = (baseline + global_b @ text.T) / 2
    scores_a, scores_b = local_a @ text.T, local_b @ text.T
    labels_a, labels_b = scores_a.argmax(-1), scores_b.argmax(-1)
    margins_a = np.sort(scores_a, axis=-1)[:, -1] - np.sort(scores_a, axis=-1)[:, -2]
    margins_b = np.sort(scores_b, axis=-1)[:, -1] - np.sort(scores_b, axis=-1)[:, -2]
    stable = labels_a == labels_b
    reliability = np.clip((local_a*local_b).sum(-1), 0., 1.)
    # Repeated/full-object crops are not additional local evidence.
    eligible = geometry[:, -1] < .95
    for i in range(len(geometry)):
        if any(np.allclose(geometry[i], geometry[j], atol=1e-6, rtol=0) for j in range(i)):
            eligible[i] = False
    support, counts = np.zeros(len(text)), np.zeros(len(text), dtype=np.int64)
    for i in np.flatnonzero(stable & eligible):
        c = labels_a[i]
        margin = min(margins_a[i], margins_b[i])
        if margin > 0:
            support[c] += float(margin * reliability[i]) / len(local_a)
            counts[c] += 1
    candidates = np.argsort(-global_scores, kind='stable')[:5]
    allowed = np.zeros(len(text), dtype=bool)
    allowed[candidates] = True
    support[(counts < 2) | ~allowed] = 0
    return {'scores': global_scores + LOCAL_WEIGHT*support,
            'baseline_scores': baseline, 'global_scores': global_scores,
            'local_support': support, 'support_counts': counts,
            'patch_agreement': stable, 'patch_eligible': eligible,
            'patch_reliability': reliability, 'patch_categories': labels_a}


class ZeroShotClassifierV83:
    def __init__(self, checkpoint, text_features, global_only=False):
        self.global_only = global_only
        if global_only:
            self.base = PointCloudClassifier('pointnext', checkpoint, text_features)
        else:
            self.extractor = FrozenFeatures(checkpoint, text_features)
            self.base = self.extractor.base
        self.metadata = {'version': VERSION, 'method': 'zero-shot-global' if global_only else 'zero-shot',
                         'learning_regime': 'training_free_no_task_labels_no_parameter_updates',
                         'base': self.base.metadata, 'local_weight': LOCAL_WEIGHT if not global_only else 0,
                         'sampling_offset': SAMPLING_OFFSET if not global_only else None,
                         'pretraining_provenance': 'not_fully_audited_no_strict_unseen_class_claim'}

    def predict(self, xyz, rgb, seed):
        if self.global_only:
            return {**self.base.predict(xyz, None, seed), 'method': 'zero-shot-global'}
        first = self.extractor.extract(xyz, seed)
        second = self.extractor.extract(xyz, seed, sampling_offset=SAMPLING_OFFSET)
        text = self.base.text_embeddings.detach().cpu().numpy()
        result = fuse_zero_shot(first, second, text)
        scores = result['scores']
        vocabulary = category_vocabulary()
        order = np.argsort(-scores, kind='stable')[:5]
        baseline_order = np.argsort(-result['baseline_scores'], kind='stable')[:5]
        return {'status': 'completed', 'method': 'zero-shot',
                'predicted_object': vocabulary[order[0]],
                'top5': [{'category': vocabulary[i], 'score': float(scores[i])} for i in order],
                'scores': scores.tolist(), 'score_type': 'cosine_plus_positive_support_not_probability',
                'baseline_top1': vocabulary[baseline_order[0]],
                'baseline_top5': [vocabulary[i] for i in baseline_order],
                'global_consensus_top1': vocabulary[int(result['global_scores'].argmax())],
                'local_evidence': [{'category': vocabulary[i], 'positive_support': float(result['local_support'][i]),
                                    'agreeing_crops': int(result['support_counts'][i]),
                                    'state': 'observed_support' if result['local_support'][i] > 0 else 'insufficient_support'}
                                   for i in order],
                'patch_diagnostics': [{'category': vocabulary[c], 'resampling_agreement': bool(agreement),
                                       'eligible': bool(eligible), 'feature_stability': float(stability)}
                                      for c, agreement, eligible, stability in zip(result['patch_categories'],
                                          result['patch_agreement'], result['patch_eligible'], result['patch_reliability'])],
                'evidence_note': 'Crop-level class support only, not named-part existence or physical visibility. '
                                 'Missing support adds no negative evidence; consistency can still be wrong.'}
