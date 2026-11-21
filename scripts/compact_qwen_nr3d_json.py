"""Compact finalized NR3D JSON and optionally share exact APFS image copies."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def replace_verified(source, target, original, original_sha):
    if sha(target) != original_sha:
        raise ValueError(f'Compaction changed file bytes: {source}')
    current = source.stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            original.st_ino, original.st_size, original.st_mtime_ns, original.st_ctime_ns):
        raise ValueError(f'Completed artifact changed during compaction: {source}')
    os.replace(target, source)


def share_image_copies(completed, target):
    # Call clonefile directly. A regular copy would consume space without sharing extents.
    clone = ctypes.CDLL(None, use_errno=True).clonefile
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    clone.restype = ctypes.c_int
    originals = {}
    files = shared_bytes = 0
    for evaluation in completed:
        episode = evaluation.parent/'episode'
        for source in sorted(episode.rglob('*.png')):
            if source.is_symlink() or not source.is_file():
                continue
            original = source.stat()
            digest = sha(source)
            reference = originals.setdefault(digest, source)
            if reference == source:
                continue
            if clone(os.fsencode(reference), os.fsencode(target), 0):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(source))
            shutil.copystat(source, target)
            replace_verified(source, target, original, digest)
            shared_bytes += original.st_size
            files += 1
    return files, shared_bytes


def compact(output, share_images=False):
    completed = sorted(output.glob('q[0-9]*/evaluation.json'))
    saved = files = 0
    with TemporaryDirectory(prefix='.nr3d-compact-', dir=output.parent) as scratch:
        target = Path(scratch)/'compressed.json'
        for evaluation in completed:
            for source in (evaluation.parent/'episode').rglob('*.json'):
                if source.is_symlink() or not source.is_file():
                    continue
                original = source.stat()
                if original.st_size < 64 * 1024 or original.st_blocks * 512 < original.st_size:
                    continue
                original_sha = sha(source)
                subprocess.run(['/usr/bin/ditto', '--hfsCompression', str(source), str(target)], check=True)
                compressed = target.stat()
                if compressed.st_blocks >= original.st_blocks:
                    target.unlink()
                    continue
                replace_verified(source, target, original, original_sha)
                saved += (original.st_blocks - compressed.st_blocks) * 512
                files += 1
        shared_files, shared_bytes = share_image_copies(completed, Path(scratch)/'image.png') if share_images else (0, 0)
    return {'completed_samples_scanned': len(completed), 'files_compressed': files, 'bytes_saved': saved,
            'images_cloned': shared_files, 'duplicate_image_bytes_cloned': shared_bytes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--share-images', action='store_true', help='Use APFS copy-on-write for byte-identical original images.')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('Transparent file compression requires macOS.')
    output = args.output.resolve()
    if not (output/'manifest.json').is_file() or not (output/'summary.json').is_file():
        parser.error('Provide a Qwen NR3D batch output directory.')
    print(json.dumps(compact(output, args.share_images), sort_keys=True))
