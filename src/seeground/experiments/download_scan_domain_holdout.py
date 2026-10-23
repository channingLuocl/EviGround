"""Resume only the frozen holdout assets, then run the original converter."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from seeground.classifiers.category_v10.data import SUFFIXES
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.experiments.scan_domain_protocol import fetch


def acquire(item):
    scene, suffix, raw = item
    path = raw / scene / (scene + suffix)
    if path.exists():
        if not path.stat().st_size:
            raise ValueError('Empty existing input')
        return 0
    path.parent.mkdir(exist_ok=True)
    part = path.with_suffix(path.suffix + '.partial')
    url = f'https://kaldir.vc.in.tum.de/scannet/v2/scans/{scene}/{scene+suffix}'
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server did not honor the resume offset; partial retained')
                elif response.status != 200:
                    raise ValueError('Unexpected initial HTTP response')
                expected = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
                copied = 0
                with part.open('ab' if offset else 'wb') as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
                        copied += len(block)
                if expected is not None and copied != expected:
                    raise ValueError('Truncated HTTP body')
            if not part.stat().st_size:
                raise ValueError('Empty input')
            part.replace(path)
            return copied
        except (TimeoutError, OSError, urllib.error.URLError):
            if attempt == 2:
                raise
    raise RuntimeError('Asset transfer failed')


def main(args):
    check_frozen(args.output, 'PROTOCOL_FROZEN.json')
    protocol = json.loads((args.output / 'protocol.json').read_text())
    jobs = [(scene, suffix, args.raw) for scene in protocol['holdout_scenes'] for suffix in SUFFIXES]
    args.raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(acquire, item): item for item in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            transferred = future.result()
            print(json.dumps({'stage': 'holdout_download', 'files': index, 'total': len(jobs),
                              'new_bytes': transferred, 'seconds': time.monotonic()-started}), flush=True)
    fetch(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('official-train', 'official-val', 'historical', 'output', 'raw'):
        parser.add_argument('--'+name, required=True, type=Path)
    main(parser.parse_args())
