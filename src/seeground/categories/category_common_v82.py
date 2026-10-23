"""Data and scoring shared by the independent v8.2 point-cloud experiment."""
from contextlib import contextmanager
import fcntl
import hashlib
from functools import lru_cache
import re
import runpy
from pathlib import Path

import numpy as np

from seeground.evaluation.run_storage import ensure_run_writable


from seeground.paths import ROOT
VERSION = 'object-category-v8.2-r1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def category_vocabulary():
    labels = runpy.run_path(str(ROOT / 'data/scannet200_constants.py'))['CLASS_LABELS_200']
    return tuple(name for name in labels if name not in ('wall', 'floor', 'ceiling'))


def normalized_category(label):
    if label is None:
        return None
    text = ' '.join(re.sub(r'[-_]', ' ', label.casefold()).split())
    aliases = {'sofa': 'couch', 'television': 'tv', 'trashcan': 'trash can'}
    plurals = {}
    for name in category_vocabulary():
        last = name.split()[-1]
        plural = {'shelf': 'shelves', 'person': 'people'}.get(
            last, last + ('es' if last.endswith(('s', 'sh', 'ch', 'x')) else 's'))
        plurals[' '.join(name.split()[:-1] + [plural])] = name
    return aliases.get(text, plurals.get(text, text))


def load_cloud(path):
    # Deliberately read only geometry, optional colour, and instance membership.
    with np.load(path, allow_pickle=False) as cloud:
        xyz, ids = cloud['xyz'], cloud['ids']
        rgb = cloud['rgb'] if 'rgb' in cloud else None
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
            or not np.isfinite(xyz).all() or ids.shape != (len(xyz),)
            or ids.dtype.kind not in 'iu'):
        raise ValueError('Expected finite xyz[N,3] and integer instance ids[N]')
    if rgb is not None and (rgb.shape != xyz.shape or not np.isfinite(rgb).all()
                            or np.any(rgb < 0) or np.any(rgb > 255)):
        raise ValueError('Expected rgb[N,3] in 0..255; convert 0..1 colours before saving')
    return xyz, rgb, ids


def instance_seed(seed, scene, oid):
    value = f'{seed}:{scene}:{oid}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little')


@contextmanager
def run_lock(output):
    ensure_run_writable(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This output directory is already running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
