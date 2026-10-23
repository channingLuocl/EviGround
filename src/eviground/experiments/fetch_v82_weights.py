"""Fetch the small, verified PointNeXt release without replacing other weights."""
import argparse
from pathlib import Path
import urllib.request

from eviground.categories.category_common_v82 import ROOT, file_hash
from eviground.classifiers.pointcloud_classifier_v82 import POINTNEXT_SHA256, POINTNEXT_URL


def fetch(path):
    if path.exists():
        if file_hash(path) != POINTNEXT_SHA256:
            raise ValueError(f'Existing file has a different hash; preserving it: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    try:
        with urllib.request.urlopen(POINTNEXT_URL, timeout=60) as response, temporary.open('xb') as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if file_hash(temporary) != POINTNEXT_SHA256:
            raise ValueError('Downloaded checkpoint failed SHA-256 verification')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'weights/pnext_cls.pth')
    args = parser.parse_args()
    fetch(args.output)
    print(f'Verified: {args.output}\nSHA256: {POINTNEXT_SHA256}')
