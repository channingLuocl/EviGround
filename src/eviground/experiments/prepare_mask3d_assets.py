"""Fetch the official validation checkpoint and pinned inference/build sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


MASK3D = '11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5'
MINKOWSKI = '02fc608bea4c0549b0a7b00ca1bf15dee4a0b228'
ASSETS = {
    'mask3d.tar.gz': f'https://codeload.github.com/JonasSchult/Mask3D/tar.gz/{MASK3D}',
    'minkowski.tar.gz': f'https://codeload.github.com/NVIDIA/MinkowskiEngine/tar.gz/{MINKOWSKI}',
    # The benchmark checkpoint also trains on validation; it must not be used here.
    'scannet200_val.ckpt': 'https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt',
}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, url = item
        path = root/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as handle:
                expected = response.headers.get('Content-Length')
                if expected and shutil.disk_usage(root).free < int(expected)+2*1024**3:
                    raise ValueError('Insufficient space for the download and build')
                shutil.copyfileobj(response, handle, 1024*1024)
            if expected and temporary.stat().st_size != int(expected):
                raise ValueError('Incomplete asset download')
            temporary.replace(path)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        record = {'url': url, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        print(json.dumps({'asset': name, **record}), flush=True)
        return name, record

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = dict(pool.map(fetch, ASSETS.items()))
    for archive, name in [('mask3d.tar.gz', 'Mask3D'), ('minkowski.tar.gz', 'MinkowskiEngine')]:
        target = root/name
        if not target.exists():
            temporary = root/(name+'.extracting')
            temporary.mkdir(exist_ok=True)
            with tarfile.open(root/archive) as handle:
                handle.extractall(temporary, filter='data')
            contents = list(temporary.iterdir())
            if len(contents) != 1 or not contents[0].is_dir():
                raise ValueError('Unexpected source archive layout')
            contents[0].rename(target)
            temporary.rmdir()
    record = {'assets': records, 'mask3d_commit': MASK3D, 'minkowski_commit': MINKOWSKI,
              'checkpoint_split': 'ScanNet200 validation checkpoint; not train+val benchmark checkpoint'}
    (root/'assets.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    prepare(parser.parse_args().asset_dir)
