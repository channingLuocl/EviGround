"""Download pinned public Uni3D assets and encode its matching EVA02-E text tower."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import numpy as np

from eviground.categories.category_common_v82 import category_vocabulary, file_hash
from eviground.classifiers.pointcloud_classifier_v82 import PROMPT
from eviground.classifiers.uni3d_classifier import (ENCODER_SHA256, TEXT_MODEL, TEXT_PRETRAINED, TEXT_REPO,
                              TEXT_REVISION, TEXT_SHA256, UNI3D_HF_REVISION,
                              UNI3D_REVISION, UNI3D_SHA256, load_uni3d_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    root = args.asset_dir
    root.mkdir(parents=True, exist_ok=True)
    source = root/'source'
    source.mkdir(exist_ok=True)
    encoder = source/'point_encoder.py'
    if not encoder.exists():
        url = f'https://raw.githubusercontent.com/baaivision/Uni3D/{UNI3D_REVISION}/models/point_encoder.py'
        with urllib.request.urlopen(url, timeout=60) as response:
            encoder.write_bytes(response.read())
    if file_hash(encoder) != ENCODER_SHA256:
        raise ValueError('Pinned official point_encoder.py checksum mismatch')
    (source/'sources.json').write_text(json.dumps({
        'repository': 'https://github.com/baaivision/Uni3D', 'revision': UNI3D_REVISION,
        'files': {'point_encoder.py': ENCODER_SHA256}}, indent=2)+'\n')

    from huggingface_hub import hf_hub_download
    files = [('BAAI/Uni3D', UNI3D_HF_REVISION, 'modelzoo/uni3d-b/model.pt', UNI3D_SHA256),
             (TEXT_REPO, TEXT_REVISION, 'open_clip_model.safetensors', TEXT_SHA256)]
    paths = []
    for repo, revision, name, digest in files:
        print(f'Downloading {repo}@{revision}/{name}', flush=True)
        path = root/'weights'/repo/name
        if not path.exists():
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=root/'weights'/repo))
        if file_hash(path) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
        print(f'Verified {path}: {digest}', flush=True)
        paths.append(path)
    if args.download_only:
        return
    target = root/'text_features_eva02e_197.npz'
    if target.exists():
        load_uni3d_text(target)
        print(f'Existing text features verified: {target}', flush=True)
        return

    import open_clip
    import torch
    from safetensors.torch import load_file
    print('Constructing official EVA02-E-14-plus and strictly loading text/image state', flush=True)
    model = open_clip.create_model(TEXT_MODEL, pretrained=None)
    state = load_file(str(paths[1]), device='cpu')
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().cuda()
    tokenizer = open_clip.get_tokenizer(TEXT_MODEL)
    vocabulary = list(category_vocabulary())
    with torch.inference_mode():
        values = torch.cat([model.encode_text(tokenizer([
            PROMPT.format(c) for c in vocabulary[i:i+16]]).cuda()).float().cpu()
                            for i in range(0, len(vocabulary), 16)]).numpy()
    metadata = dict(vocabulary=vocabulary, prompt=PROMPT, text_model=TEXT_MODEL,
                    text_pretrained=TEXT_PRETRAINED, text_repository=TEXT_REPO,
                    text_revision=TEXT_REVISION, text_checkpoint_sha256=TEXT_SHA256,
                    strict_load=True, torch=torch.__version__,
                    open_clip_torch=importlib.metadata.version('open_clip_torch'))
    with target.open('xb') as handle:
        np.savez_compressed(handle, embeddings=values, metadata=json.dumps(metadata))
    load_uni3d_text(target)
    print(f'Saved and validated {values.shape}: {target}', flush=True)


if __name__ == '__main__':
    main()
