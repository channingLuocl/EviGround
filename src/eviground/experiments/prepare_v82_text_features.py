"""Prepare public category text embeddings on a host that can reach Hugging Face."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from eviground.categories.category_common_v82 import category_vocabulary, file_hash
from eviground.classifiers.pointcloud_classifier_v82 import CLIP_REVISION, PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    name = 'openai/clip-vit-base-patch16'
    tokenizer = AutoTokenizer.from_pretrained(name, revision=CLIP_REVISION)
    model = CLIPTextModelWithProjection.from_pretrained(name, revision=CLIP_REVISION).eval()
    vocabulary = list(category_vocabulary())
    tokens = tokenizer([PROMPT.format(c) for c in vocabulary], padding=True, return_tensors='pt')
    with torch.inference_mode():
        embeddings = model(**tokens).text_embeds.float().numpy()
    metadata = dict(vocabulary=vocabulary, prompt=PROMPT, text_model=name,
                    text_revision=CLIP_REVISION, torch=torch.__version__,
                    transformers=importlib.metadata.version('transformers'),
                    checkpoint_sha256=file_hash(hf_hub_download(name, 'pytorch_model.bin', revision=CLIP_REVISION)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, embeddings=embeddings, metadata=json.dumps(metadata))
    print(f'Saved {embeddings.shape}: {args.output}; SHA256 {file_hash(args.output)}')


if __name__ == '__main__':
    main()
