"""Official frozen Uni3D-B encoder with its own EVA02-E text space."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seeground.categories.category_common_v82 import category_vocabulary, file_hash
from seeground.classifiers.pointcloud_classifier_v82 import PROMPT, check_inference_platform, prepare_points


UNI3D_REVISION = '64e03c3c42c196e8cb5ed03857810af9fc9ac39c'
ENCODER_SHA256 = '3967a6b84b44314a81c75093bd36a3950eebb3eb96a782b2c6078234eebc6c39'
UNI3D_HF_REVISION = '3d8233b76aa350d72f6213ecd2123c2026b42355'
UNI3D_SHA256 = 'bb533f918957e939f0e500376bef5166437883443da10fd9adccde0d1b512fba'
TEXT_REPO = 'timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k'
TEXT_REVISION = '0162f50c2742be3c90b3a81e4e8066fc5a90e125'
TEXT_MODEL = 'EVA02-E-14-plus'
TEXT_PRETRAINED = 'laion2b_s9b_b144k'
TEXT_SHA256 = 'dcdd43722d13420fd9682b162167bc10e1d2afe89f3da90224adf67658669d97'


def prepare_uni3d_points(xyz, rgb, seed):
    # Keep the same sampling/colours, then rotate ScanNet Z-up to Uni3D Y-up.
    points = np.ascontiguousarray(prepare_points(xyz, rgb, 'openshape', seed).T)
    points[:, :3] = points[:, [0, 2, 1]]
    points[:, 2] *= -1  # (x,y,z) -> (x,z,-y), a proper rotation rather than reflection.
    return points


def load_uni3d_text(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        embeddings = data['embeddings']
    if (metadata.get('vocabulary') != list(category_vocabulary())
            or metadata.get('prompt') != PROMPT
            or metadata.get('text_model') != TEXT_MODEL
            or metadata.get('text_revision') != TEXT_REVISION
            or metadata.get('text_checkpoint_sha256') != TEXT_SHA256
            or embeddings.shape != (len(category_vocabulary()), 1024)
            or not np.isfinite(embeddings).all()
            or (np.linalg.norm(embeddings, axis=1) <= 0).any()):
        raise ValueError('Uni3D requires matching EVA02-E text features, vocabulary and prompt')
    return embeddings, metadata


def point_encoder_state(checkpoint):
    state = checkpoint['module']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    unexpected = set(state) - {'logit_scale'} - {k for k in state if k.startswith('point_encoder.')}
    if unexpected or not any(k.startswith('point_encoder.') for k in state):
        raise ValueError('Unexpected Uni3D checkpoint structure')
    return {k.removeprefix('point_encoder.'): v for k, v in state.items()
            if k.startswith('point_encoder.')}


class Uni3DClassifier:
    def __init__(self, checkpoint, text_features, source_dir):
        check_inference_platform()
        import torch
        import timm
        if not torch.cuda.is_available():
            raise ValueError('Uni3D evaluation requires an NVIDIA CUDA GPU')
        if file_hash(checkpoint) != UNI3D_SHA256:
            raise ValueError('Uni3D-B checkpoint hash differs from the pinned official release')
        source_dir = Path(source_dir)
        assets = json.loads((source_dir/'sources.json').read_text())
        source_path = source_dir/'point_encoder.py'
        if (assets['revision'] != UNI3D_REVISION
                or file_hash(source_path) != ENCODER_SHA256
                or assets['files']['point_encoder.py'] != ENCODER_SHA256):
            raise ValueError('Uni3D source revision/hash mismatch')
        spec = importlib.util.spec_from_file_location('_seeground_uni3d_point_encoder', source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(pc_feat_dim=768, embed_dim=1024, group_size=64,
                               num_group=512, pc_encoder_dim=512, patch_dropout=0.)
        transformer = timm.create_model('eva02_base_patch14_448', pretrained=False,
                                        drop_path_rate=0.)
        self.encoder = module.PointcloudEncoder(transformer, args)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(point_encoder_state(saved), strict=True)
        del saved
        self.encoder.eval().cuda()
        values, text_metadata = load_uni3d_text(text_features)
        self.text_embeddings = torch.nn.functional.normalize(
            torch.from_numpy(values).float(), dim=-1).cuda()
        self.torch, self.vocabulary = torch, category_vocabulary()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.metadata = {
            'backend': 'uni3d', 'model': 'Uni3D-B', 'checkpoint_sha256': UNI3D_SHA256,
            'checkpoint_revision': UNI3D_HF_REVISION, 'strict_load': True,
            'source_revision': UNI3D_REVISION, 'source_sha256': file_hash(source_path),
            'text_features_sha256': file_hash(text_features), 'text_features': text_metadata,
            'prompt': PROMPT, 'points': 10000, 'channels': 'xyz_rgb',
            'input_gravity_axis': 'Z', 'gravity_axis': 'Y',
            'frame_transform': '(x,y,z)->(x,z,-y)',
            'groups': 512, 'group_size': 64, 'precision': 'float32',
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('timm', 'open_clip_torch', 'pointnet2_ops')}}

    def predict(self, xyz, rgb, seed):
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        points = torch.from_numpy(prepare_uni3d_points(xyz, rgb, seed)[None]).cuda()
        with torch.inference_mode():
            embedding = self.encoder(points[:, :, :3].contiguous(), points[:, :, 3:].contiguous())
            if (embedding.shape != (1, 1024) or not torch.isfinite(embedding).all()
                    or embedding.norm() <= 0):
                raise ValueError('Invalid Uni3D point embedding')
            scores = (torch.nn.functional.normalize(embedding.float(), dim=-1)
                      @ self.text_embeddings.T)[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite Uni3D category scores')
        order = np.argsort(-scores, kind='stable')[:5]
        return {'predicted_object': self.vocabulary[order[0]], 'scores': scores.tolist(),
                'top5': [{'category': self.vocabulary[i], 'similarity': float(scores[i])}
                         for i in order],
                'score_type': 'cosine_similarity_not_probability', 'status': 'completed'}
