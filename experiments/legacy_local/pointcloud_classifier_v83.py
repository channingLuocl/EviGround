"""Frozen global/crop features and trained v8.3 heads; inference accepts geometry only."""
import json
from pathlib import Path

import numpy as np

from category_common_v82 import category_vocabulary, file_hash
from evidence_data_v83 import VERSION, STATES, local_patches
from pointcloud_classifier_v82 import (CLIP_REVISION, POINTNEXT_SHA256, PointCloudClassifier,
                                      prepare_points)


def load_part_text(path, schema):
    with np.load(path, allow_pickle=False) as data:
        metadata, values = json.loads(str(data['metadata'])), data['embeddings']
    if (metadata.get('parts') != schema['parts'] or metadata.get('prompt') != schema['prompt']
            or metadata.get('text_model') != 'openai/clip-vit-base-patch16'
            or metadata.get('text_revision') != CLIP_REVISION
            or values.shape != (len(schema['parts']), 512) or not np.isfinite(values).all()
            or (np.linalg.norm(values, axis=1) <= 0).any()):
        raise ValueError('Part text features do not match schema / frozen CLIP space')
    return values.astype(np.float32)


class FrozenFeatures:
    def __init__(self, checkpoint, category_text):
        from run_categories_v82 import code_hashes
        self.base = PointCloudClassifier('pointnext', checkpoint, category_text)
        self.metadata = {'version': VERSION, 'extractor': 'global-plus-eight-knn-crops-r1',
                         'base': self.base.metadata,
                         'encoder_code_sha256': code_hashes(),
                         'local_semantics': 'frozen_global_encoder_on_crops_not_a_pretrained_part_segmenter'}

    def extract(self, xyz, seed, sampling_offset=0):
        torch = self.base.torch
        sampling_seed = (seed + sampling_offset) % 2**32
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        indices, geometry = local_patches(xyz, seed)
        # First item exactly reuses v8.2 sampling/preprocessing and released projection.
        features = [prepare_points(xyz, None, 'pointnext', sampling_seed)]
        features.extend(prepare_points(xyz[idx], None, 'pointnext', (sampling_seed+i+1) % 2**32)
                        for i, idx in enumerate(indices))
        embeddings = []
        with torch.inference_mode():
            # Keep the original single-object kernel path; avoid a new batch-dependent baseline.
            for values in features:
                tensor = self.base.encoder(torch.from_numpy(values[None]).cuda())
                if tensor.shape != (1, 512) or not torch.isfinite(tensor).all() or tensor.norm() <= 0:
                    raise ValueError('Invalid global/local point embedding')
                embeddings.append(torch.nn.functional.normalize(tensor.float(), dim=-1)[0].cpu().numpy())
        return {'global_features': embeddings[0], 'local_features': np.stack(embeddings[1:]),
                'geometry': geometry}


def load_head(path):
    import torch
    from evidence_head_v83 import EvidenceHead
    if not Path(path).is_file():
        raise ValueError('No trained v8.3 checkpoint found; prepare annotations, cache features and train first '
                         '(local/README_V8_3.md)')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('version') != VERSION or not checkpoint.get('trained')
            or checkpoint.get('vocabulary') != list(category_vocabulary())):
        raise ValueError('Need a trained, compatible v8.3 checkpoint; run cache + train first')
    state = checkpoint['state_dict']
    if (state['category_text'].shape != (len(category_vocabulary()), 512)
            or state['part_text'].shape != (len(checkpoint['parts']), 512)
            or checkpoint['parts_schema']['parts'] != checkpoint['parts']
            or any(not torch.isfinite(value).all() for value in state.values())):
        raise ValueError('Invalid trained head tensors or part vocabulary')
    head = EvidenceHead(state['category_text'], state['part_text'], checkpoint['mode'])
    head.load_state_dict(state, strict=True)
    head.eval()
    return head, checkpoint


class PointCloudClassifierV83:
    def __init__(self, checkpoint, text_features, head_checkpoint):
        self.head, self.record = load_head(head_checkpoint)
        if (file_hash(checkpoint) != POINTNEXT_SHA256
                or self.record['extractor_assets']['pointnext_sha256'] != file_hash(checkpoint)
                or self.record['extractor_assets']['category_text_sha256'] != file_hash(text_features)):
            raise ValueError('Frozen backbone/text assets differ from v8.3 training')
        self.extractor = FrozenFeatures(checkpoint, text_features)
        if self.record['extractor'] != self.extractor.metadata:
            raise ValueError('Extractor/runtime differs from feature-cache training provenance')
        self.metadata = {'head_sha256': file_hash(head_checkpoint), 'mode': self.record['mode'],
                         'extractor': self.extractor.metadata, 'states': list(STATES),
                         'state_semantics': 'uncalibrated_statistical_estimates_not_physical_visibility'}

    def predict(self, xyz, rgb, seed):
        import torch
        features = self.extractor.extract(xyz, seed)
        with torch.inference_mode():
            result = self.head(*(torch.from_numpy(features[k][None]) for k in
                                 ('global_features', 'local_features', 'geometry')))
        scores = result['logits'][0].numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite category logits')
        vocabulary = category_vocabulary()
        order = np.argsort(-scores, kind='stable')[:5]
        answer = {'status': 'completed', 'predicted_object': vocabulary[order[0]],
                  'top5': [{'category': vocabulary[i], 'logit': float(scores[i])} for i in order],
                  'scores': scores.tolist(), 'score_type': 'uncalibrated_logits_not_probability'}
        if 'state_probabilities' in result:
            answer['part_evidence'] = {p: dict(zip(STATES, map(float, values))) for p, values in
                                       zip(self.record['parts'], result['state_probabilities'][0].numpy())}
            answer['evidence_note'] = ('Estimated states; no sensor visibility is measured. ' +
                                       ('Unknown is neutral.' if self.record['mode'] == 'evidence' else
                                        'Control mode uses unrestricted state fusion.'))
        return answer
