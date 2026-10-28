"""Pinned official ULIP-2 colored PointBERT and its matching bigG text space."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from category_common_v82 import category_vocabulary, file_hash
from pointcloud_classifier_v82 import PROMPT, check_inference_platform, prepare_points


SOURCE_REVISION = '95d480fe2b16c06d0558c60b5cfea981b4cdc8eb'
HF_REVISION = '3f33d96e6bfec2a793d06acf0df4c591ddd47489'
CHECKPOINT_NAME = 'ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt'
CHECKPOINT_SHA256 = 'a4b5ed9799d5841a1646e0fb7d24cb8dcdd3b3e6fab2e2575de2531f71274adb'
TEXT_MODEL = 'ViT-bigG-14'
TEXT_PRETRAINED = 'laion2b_s39b_b160k'
TEXT_REPO = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
TEXT_REVISION = '743c27bd53dfe508a0ade0f50698f99b39d03bec'
TEXT_SHA256 = 'db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34'


def to_ulip2_frame(points):
    """B,6,N ScanNet Z-up -> B,N,6 Y-up; preserve RGB and handedness."""
    points = points.transpose(1, 2)
    return torch.cat((points[:, :, 0:1], points[:, :, 2:3],
                      -points[:, :, 1:2], points[:, :, 3:]), dim=-1)


def prepare_ulip2_points(xyz, rgb, seed):
    points = prepare_points(xyz, rgb, 'openshape', seed)
    return to_ulip2_frame(torch.from_numpy(points[None]))[0].numpy()


def load_ulip2_text(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        values = data['embeddings']
    if (metadata.get('vocabulary') != list(category_vocabulary())
            or metadata.get('prompt') != PROMPT
            or metadata.get('text_model') != TEXT_MODEL
            or metadata.get('text_pretrained') != TEXT_PRETRAINED
            or metadata.get('text_revision') != TEXT_REVISION
            or metadata.get('text_checkpoint_sha256') != TEXT_SHA256
            or not metadata.get('strict_load')
            or values.shape != (197, 1280) or not np.isfinite(values).all()
            or (np.linalg.norm(values, axis=1) <= 0).any()):
        raise ValueError('ULIP-2 requires matching frozen bigG text features, vocabulary and prompt')
    return values, metadata


def encoder_state(saved):
    state = {k.removeprefix('module.'): v for k, v in saved['state_dict'].items()}
    unexpected = [k for k in state if k != 'logit_scale'
                  and not k.startswith('point_encoder.') and k != 'pc_projection']
    if unexpected or state['pc_projection'].shape != (768, 1280):
        raise ValueError(f'Unexpected ULIP-2 checkpoint structure: {unexpected}')
    return {k: v for k, v in state.items() if k != 'logit_scale'}


class ULIP2Encoder(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.point_encoder = point_encoder
        self.pc_projection = nn.Parameter(torch.empty(768, 1280))

    def forward(self, points):
        return self.point_encoder(to_ulip2_frame(points)) @ self.pc_projection


def load_encoder(checkpoint, source_dir):
    check_inference_platform()
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('ULIP-2 checkpoint differs from the pinned official release')
    source = Path(source_dir)
    provenance = json.loads((source/'sources.json').read_text())
    if provenance['revision'] != SOURCE_REVISION:
        raise ValueError('ULIP-2 source revision mismatch')
    for name, digest in provenance['runtime_sha256'].items():
        if file_hash(source/name) != digest:
            raise ValueError(f'ULIP-2 runtime source checksum mismatch: {name}')
    package_name = '_seeground_ulip2_official'
    spec = importlib.util.spec_from_file_location(package_name, source/'__init__.py',
                                                 submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    from importlib import import_module
    module = import_module(package_name+'.point_encoder')
    import yaml
    config = SimpleNamespace(**yaml.safe_load((source/'config.yaml').read_text())['model'])
    model = ULIP2Encoder(module.PointTransformer_Colored(config, args=SimpleNamespace()))
    # The hash-verified official release contains NumPy scalar training metadata.
    state = encoder_state(torch.load(checkpoint, map_location='cpu', weights_only=False))
    model.load_state_dict(state, strict=True)
    return model.eval()


class ULIP2Classifier:
    def __init__(self, checkpoint, text_features, source_dir):
        self.encoder = load_encoder(checkpoint, source_dir).cuda()
        values, metadata = load_ulip2_text(text_features)
        self.text_embeddings = nn.functional.normalize(torch.from_numpy(values).float(), dim=-1).cuda()
        self.vocabulary = category_vocabulary()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.metadata = {'backend': 'ulip2', 'model': 'ULIP-2 PointBERT 10k XYZRGB',
                         'source_revision': SOURCE_REVISION, 'checkpoint_revision': HF_REVISION,
                         'checkpoint_sha256': CHECKPOINT_SHA256, 'strict_load': True,
                         'source_manifest_sha256': file_hash(Path(source_dir)/'sources.json'),
                         'text_features_sha256': file_hash(text_features), 'text_features': metadata,
                         'prompt': PROMPT, 'points': 10000, 'groups': 512, 'group_size': 32,
                         'input_gravity_axis': 'Z', 'gravity_axis': 'Y',
                         'frame_transform': '(x,y,z)->(x,z,-y)', 'precision': 'float32',
                         'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                         'gpu': torch.cuda.get_device_name(0)}

    def predict(self, xyz, rgb, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        points = torch.from_numpy(prepare_points(xyz, rgb, 'openshape', seed)[None]).cuda()
        with torch.inference_mode():
            embedding = self.encoder(points)
            if (embedding.shape != (1, 1280) or not torch.isfinite(embedding).all()
                    or embedding.norm() <= 0):
                raise ValueError('Invalid ULIP-2 point embedding')
            scores = (nn.functional.normalize(embedding.float(), dim=-1)
                      @ self.text_embeddings.T)[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite ULIP-2 category scores')
        order = np.argsort(-scores, kind='stable')[:5]
        return {'predicted_object': self.vocabulary[order[0]], 'scores': scores.tolist(),
                'top5': [{'category': self.vocabulary[i], 'similarity': float(scores[i])}
                         for i in order],
                'score_type': 'cosine_similarity_not_probability', 'status': 'completed'}
