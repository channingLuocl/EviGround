"""Uni3D-B on the same instances, vocabulary and scoring as v8.2."""
from eviground.paths import source_file
import argparse
from pathlib import Path
import sys

from eviground.categories.category_common_v82 import ROOT, file_hash
from eviground.experiments.run_categories_v82 import run
from eviground.classifiers.uni3d_classifier import Uni3DClassifier, UNI3D_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--text-features', type=Path, required=True,
                        help='197 x 1024 EVA02-E features prepared for Uni3D')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/validation/nr3d_val_full.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT/'local_outputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.backend, args.cloud, args.bbox_id = 'uni3d', None, None
    try:
        if file_hash(args.checkpoint) != UNI3D_SHA256:
            raise ValueError('Uni3D-B checkpoint hash mismatch')
        extra = {'model': 'Uni3D-B', 'checkpoint_sha256': UNI3D_SHA256,
                 'text_features_sha256': file_hash(args.text_features),
                 'sources_sha256': file_hash(args.source_dir/'sources.json'),
                 'point_encoder_sha256': file_hash(args.source_dir/'point_encoder.py'),
                 'extension_code_sha256': {name: file_hash(source_file(name))
                                          for name in ('run_uni3d_categories.py', 'uni3d_classifier.py')}}
        def factory(backend, checkpoint, text_features):
            return Uni3DClassifier(checkpoint, text_features, args.source_dir)
        return run(args, classifier_factory=factory, extra_config=extra)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\nSee local/README_UNI3D.md\n')


if __name__ == '__main__':
    sys.exit(main())
