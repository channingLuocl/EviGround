"""Frozen pretrained encoders. No GT, author labels, queries, or VLM API inputs."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

from eviground.categories.category_common_v82 import ROOT, category_vocabulary, file_hash


POINTNEXT_SHA256 = 'd6551942dda8ddf4af1d840ca8a83ee60f3a7535a618bb5b541e145b95663f6b'
POINTNEXT_URL = 'https://raw.githubusercontent.com/iris0329/SeeGround/master/weights/pnext_cls.pth'
CLIP_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
OPENSHAPE_REVISION = 'd771a992218de10b967eea690e8f474aa96dd758'
PROMPT = 'a {} in a scene'


def load_text_features(path):
    """Public category embeddings prepared with the matching frozen CLIP text tower."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        embeddings = data['embeddings']
    if (metadata['vocabulary'] != list(category_vocabulary()) or metadata['prompt'] != PROMPT
            or metadata['text_model'] != 'openai/clip-vit-base-patch16'
            or metadata['text_revision'] != CLIP_REVISION
            or embeddings.shape != (len(category_vocabulary()), 512)
            or not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) <= 0).any()):
        raise ValueError('Offline text features do not match the PointNeXt text space/vocabulary')
    return embeddings, metadata


def check_inference_platform():
    if sys.platform == 'darwin':
        raise ValueError('当前运行在 Mac。v8.2 的点云推理需要 NVIDIA CUDA，'
                         '请在 NVIDIA GPU 服务器上运行；在 Mac 安装 torch 也不能运行此路径。'
                         '本机可使用 --prepare-only 检查数据，安装说明见 local/README_V8_2.md。')


def prepare_points(xyz, rgb, backend, seed):
    """Preserve aspect ratio and Z-up; match each released encoder's channels."""
    points = np.array(xyz, dtype=np.float64, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Instance has no finite XYZ points')
    points -= points.mean(0)
    radius = np.linalg.norm(points, axis=1).max()
    if radius <= 1e-12:
        raise ValueError('Instance geometry has zero radius')
    points /= radius
    count = 2048 if backend == 'pointnext' else 10000
    indices = np.random.RandomState(seed).choice(len(points), count, replace=len(points) < count)
    points = points[indices]
    if backend == 'pointnext':
        # Original SeeGround model slices off RGB and uses XYZ + sampled height.
        height = points[:, 2:3] - points[:, 2:3].min()
        features = np.concatenate([points, height], axis=1)
    elif backend == 'openshape':
        colors = np.full_like(points, .4) if rgb is None else np.asarray(rgb)[indices] / 255.
        features = np.concatenate([points, colors], axis=1).T
    else:
        raise ValueError(f'Unknown backend: {backend}')
    return np.ascontiguousarray(features, dtype=np.float32)


def load_pointnext(checkpoint):
    import torch
    # These are the released encoder AND learned semantic projection, not answer labels.
    if file_hash(checkpoint) != POINTNEXT_SHA256:
        raise ValueError('PointNeXt checkpoint differs from verified release; run fetch_v82_weights.py')
    sys.path.insert(0, str(ROOT / 'third_party'))
    os.environ['EVIGROUND_POINTNEXT_INFERENCE_ONLY'] = '1'
    # The vendored encoder still reads its original compatibility switch.
    os.environ['SEEGROUND_POINTNEXT_INFERENCE_ONLY'] = '1'
    from models.pcd_classifier import PcdClassifier
    model = PcdClassifier()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, {'checkpoint_sha256': file_hash(checkpoint), 'strict_load': True,
                   'parameters_loaded': len(state), 'projection_shape': list(state['pc_projection'].shape)}


class PointCloudClassifier:
    def __init__(self, backend, checkpoint, text_features=None):
        check_inference_platform()
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != 'torch':
                raise
            raise ValueError(f'当前 Python 环境未安装 PyTorch：{sys.executable}。'
                             '请在 NVIDIA GPU 服务器的这个环境内安装 CUDA 版 PyTorch，'
                             '再按 local/README_V8_2.md 安装依赖和编译算子。') from exc
        if not torch.cuda.is_available():
            raise ValueError(f'当前 Python 无法使用 NVIDIA CUDA：{sys.executable}。'
                             '请检查服务器 GPU、驱动及 PyTorch CUDA 版本；'
                             '只检查数据可使用 --prepare-only。')
        self.backend, self.vocabulary = backend, category_vocabulary()
        self.torch = torch
        torch.cuda.set_device(0)  # Select another physical GPU with CUDA_VISIBLE_DEVICES.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        texts = [PROMPT.format(name) for name in self.vocabulary]
        with torch.inference_mode():
            if backend == 'pointnext':
                self.encoder, metadata = load_pointnext(checkpoint)
                name = 'openai/clip-vit-base-patch16'
                if text_features:
                    values, provenance = load_text_features(text_features)
                    embeddings = torch.from_numpy(values)
                    metadata.update(text_features_sha256=file_hash(text_features),
                                    text_features_provenance=provenance)
                    text_model = None
                else:
                    from transformers import AutoTokenizer, CLIPTextModelWithProjection
                    tokenizer = AutoTokenizer.from_pretrained(name, revision=CLIP_REVISION)
                    text_model = CLIPTextModelWithProjection.from_pretrained(name, revision=CLIP_REVISION)
                    tokens = tokenizer(texts, padding=True, return_tensors='pt')
                    embeddings = text_model.eval()(**tokens).text_embeds
                metadata.update(text_model=name, text_revision=CLIP_REVISION,
                                channels='xyz_height', points=2048)
            else:
                import openshape
                import open_clip
                from huggingface_hub import hf_hub_download
                name = 'OpenShape/openshape-pointbert-vitg14-rgb'
                path = hf_hub_download(name, 'model.pt', revision=OPENSHAPE_REVISION)
                self.encoder = openshape.G14(torch.load(path, map_location='cpu', weights_only=True))
                clip_name, pretrained = 'ViT-bigG-14', 'laion2b_s39b_b160k'
                clip_path = open_clip.download_pretrained(open_clip.get_pretrained_cfg(clip_name, pretrained))
                text_model = open_clip.create_model(clip_name, pretrained=clip_path).eval()
                # Encode text on CPU, then release the large CLIP model; only point encoder uses GPU.
                tokenizer = open_clip.get_tokenizer(clip_name)
                embeddings = torch.cat([text_model.encode_text(tokenizer(texts[i:i+16]))
                                        for i in range(0, len(texts), 16)])
                metadata = {'checkpoint_sha256': file_hash(path), 'strict_load': True,
                            'checkpoint_revision': OPENSHAPE_REVISION, 'text_model': clip_name,
                            'text_pretrained': pretrained, 'text_checkpoint_sha256': file_hash(clip_path),
                            'points': 10000, 'channels': 'xyz_rgb', 'gravity_axis': 'Z',
                            'support_code_sha256': {p.name: file_hash(p) for p in
                                Path(openshape.__file__).parent.glob('*.py')}}
            if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=-1) <= 0).any():
                raise ValueError('Invalid text embeddings')
            self.text_embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).cuda()
            del text_model
            self.encoder = self.encoder.eval().cuda()
        self.metadata = {'backend': backend, 'prompt': PROMPT, 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                         'packages': {name: importlib.metadata.version(name) for name in
                                      (('transformers',) if backend == 'pointnext' else
                                       ('open_clip_torch', 'openshape', 'dgl'))}, **metadata}

    def predict(self, xyz, rgb, seed):
        torch = self.torch
        # Object-specific RNG makes sampling independent of cache/resume ordering.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        features = prepare_points(xyz, rgb, self.backend, seed)
        with torch.inference_mode():
            embedding = self.encoder(torch.from_numpy(features[None]).cuda())
            if (embedding.shape != (1, self.text_embeddings.shape[1])
                    or not torch.isfinite(embedding).all() or embedding.norm() <= 0):
                raise ValueError('Invalid point embedding or mismatched text space')
            scores = (torch.nn.functional.normalize(embedding.float(), dim=-1)
                      @ self.text_embeddings.T)[0].cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite category scores')
        order = np.argsort(-scores, kind='stable')[:5]
        return {'predicted_object': self.vocabulary[order[0]],
                'top5': [{'category': self.vocabulary[i], 'similarity': float(scores[i])} for i in order],
                'scores': scores.tolist(), 'score_type': 'cosine_similarity_not_probability',
                'status': 'completed'}
