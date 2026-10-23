"""Fetch unlabeled ScanNet superpoints and alignment metadata for fixed eval scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np


BASE = 'https://kaldir.vc.in.tum.de/scannet/v2/scans'
SUFFIXES = ('.txt', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply')


def prepare(manifest, output):
    records = json.loads(manifest.read_text())
    clouds = {r['scene']: Path(r['cloud']) for r in records['samples']}
    output.mkdir(parents=True, exist_ok=True)

    def scene(name):
        directory = output/name
        directory.mkdir(exist_ok=True)
        hashes = {}
        for suffix in SUFFIXES:
            path = directory/(name+suffix)
            if not path.exists():
                if shutil.disk_usage(output).free < 2*1024**3:
                    raise ValueError('Less than 2 GiB free; stopped downloading')
                with urllib.request.urlopen(f'{BASE}/{name}/{name+suffix}', timeout=120) as response:
                    value = response.read()
                temporary = path.with_suffix(path.suffix+'.part')
                temporary.write_bytes(value)
                temporary.replace(path)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        segments = json.loads((directory/(name+SUFFIXES[1])).read_text())['segIndices']
        with np.load(clouds[name], allow_pickle=False) as cloud:
            if len(segments) != len(cloud['xyz']):
                raise ValueError(f'Vertex count differs: {name}')
        line = next(x for x in (directory/(name+'.txt')).read_text().splitlines()
                    if x.startswith('axisAlignment'))
        matrix = np.array(list(map(float, line.split('=', 1)[1].split()))).reshape(4, 4)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < .99:
            raise ValueError(f'Invalid alignment: {name}')
        from plyfile import PlyData
        vertices = PlyData.read(directory/(name+SUFFIXES[2]))['vertex']
        xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')])
        rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')])
        aligned = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3]
        with np.load(clouds[name], allow_pickle=False) as cloud:
            if (not np.allclose(aligned, cloud['xyz'], atol=2e-5, rtol=0)
                    or not np.array_equal(rgb, cloud['rgb'])):
                raise ValueError(f'Raw mesh vertex order differs from fixed evaluation cloud: {name}')
        record = {'scene': name, 'vertices': len(segments), 'sha256': hashes}
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(scene, sorted(clouds)))
    (output/'inputs.json').write_text(json.dumps({'scenes': result,
        'model_input': 'unlabeled superpoints, RGB, unaligned XYZ; no GT instances/categories'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
