"""Encode the shared 197-category prompt with the matching official frozen bigG."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from eviground.categories.category_common_v82 import category_vocabulary, file_hash
from eviground.classifiers.pointcloud_classifier_v82 import PROMPT
from eviground.classifiers.ulip2_classifier import (TEXT_MODEL, TEXT_PRETRAINED, TEXT_REPO, TEXT_REVISION,
                              TEXT_SHA256, load_ulip2_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        load_ulip2_text(args.output)
        return
    if file_hash(args.checkpoint) != TEXT_SHA256:
        raise ValueError('Official bigG text/image checkpoint checksum mismatch')
    import open_clip
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    model = open_clip.create_model(TEXT_MODEL, pretrained=None)
    state = load_file(str(args.checkpoint), device='cpu')
    model.load_state_dict(state, strict=True)
    del state
    model.eval().cuda()
    tokenizer = open_clip.get_tokenizer(TEXT_MODEL)
    vocabulary = list(category_vocabulary())
    with torch.inference_mode():
        values = torch.cat([model.encode_text(tokenizer([
            PROMPT.format(c) for c in vocabulary[i:i+16]]).cuda()).float().cpu()
                            for i in range(0, len(vocabulary), 16)]).numpy()
    metadata = dict(vocabulary=vocabulary, prompt=PROMPT, text_model=TEXT_MODEL,
                    text_pretrained=TEXT_PRETRAINED, text_repository=TEXT_REPO,
                    text_revision=TEXT_REVISION, text_checkpoint_sha256=TEXT_SHA256,
                    strict_load=True, torch=str(torch.__version__),
                    open_clip_torch=importlib.metadata.version('open_clip_torch'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, embeddings=values, metadata=json.dumps(metadata))
    load_ulip2_text(args.output)
    print(f'Saved and validated {values.shape}: {args.output}', flush=True)


if __name__ == '__main__':
    main()
