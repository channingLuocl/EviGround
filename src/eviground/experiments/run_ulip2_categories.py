"""ULIP-2 on the identical v8.2 instance selection and scoring protocol."""
from eviground.paths import source_file
import argparse
from pathlib import Path

from eviground.categories.category_common_v82 import ROOT, file_hash
from eviground.experiments.run_categories_v82 import run
from eviground.classifiers.ulip2_classifier import CHECKPOINT_SHA256, ULIP2Classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'text-features', 'source-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/validation/nr3d_val_full.json')
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.backend, args.cloud, args.bbox_id = 'ulip2', None, None
    extra = {'model': 'ULIP-2 PointBERT 10k XYZRGB', 'checkpoint_sha256': CHECKPOINT_SHA256,
             'text_features_sha256': file_hash(args.text_features),
             'sources_sha256': file_hash(args.source_dir/'sources.json'),
             'extension_code_sha256': {name: file_hash(source_file(name)) for name in
                                      ('run_ulip2_categories.py', 'ulip2_classifier.py')}}
    return run(args, classifier_factory=lambda *unused: ULIP2Classifier(
        args.checkpoint, args.text_features, args.source_dir), extra_config=extra)


if __name__ == '__main__':
    raise SystemExit(main())
