"""Resumable HTTP range downloads of the two pinned public Uni3D weight files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import requests

from category_common_v82 import file_hash
from uni3d_classifier import TEXT_REPO, TEXT_REVISION, TEXT_SHA256, UNI3D_HF_REVISION, UNI3D_SHA256


def download(root, repo, revision, name, size, digest):
    target = root/'weights'/repo/name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != digest:
            raise ValueError(f'Existing checkpoint checksum mismatch: {target}')
        print(f'Already verified: {target}', flush=True)
        return
    partial = target.with_name(target.name+'.partial')
    journal = target.with_name(target.name+'.ranges.json')
    chunk_size = 8*1024*1024
    state = json.loads(journal.read_text()) if journal.exists() else {
        'sha256': digest, 'size': size, 'chunk_size': chunk_size, 'completed': []}
    if (state['sha256'], state['size'], state['chunk_size']) != (digest, size, chunk_size):
        raise ValueError('Range journal belongs to different weights')
    if not partial.exists():
        if state['completed']:
            raise ValueError('Partial download missing but journal has completed ranges')
        with partial.open('wb') as handle:
            handle.truncate(size)
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{repo}/resolve/{revision}/{name}"
    # Resolve once: repeatedly hitting the mirror for every range triggers its rate limit.
    with requests.get(url+'?download=true', headers={'Range': 'bytes=0-7'}, stream=True,
                      timeout=(15, 30)) as response:
        response.raise_for_status()
        download_url = response.url

    def fetch(index):
        start, end = index*chunk_size, min((index+1)*chunk_size, size)-1
        for attempt in range(3):
            try:
                with requests.get(download_url, stream=True, headers={'Range': f'bytes={start}-{end}'},
                                  timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if (response.status_code != 206 or response.headers.get('Content-Range')
                            != f'bytes {start}-{end}/{size}'):
                        raise ValueError(f'Unexpected HTTP range response: {response.status_code}, '
                                         f"{response.headers.get('Content-Range')}")
                    received = 0
                    with partial.open('r+b') as handle:
                        handle.seek(start)
                        for block in response.iter_content(1024*1024):
                            if received+len(block) > end-start+1:
                                raise ValueError('Server returned bytes beyond requested range')
                            handle.write(block)
                            received += len(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if received != end-start+1:
                        raise ValueError(f'Short HTTP range: {received}')
                return index
            except (requests.RequestException, ValueError) as exc:
                code = exc.response.status_code if isinstance(exc, requests.HTTPError) else None
                print(f'Range {index}, attempt {attempt+1}: {type(exc).__name__} {code}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(2*(attempt+1))

    total = (size+chunk_size-1)//chunk_size
    began = time.monotonic()
    print(f'Downloading {repo}/{name}, {size/1e9:.2f} GB, 32 CDN connections', flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(fetch, i) for i in range(total) if i not in state['completed']]
        for future in as_completed(futures):
            try:
                state['completed'].append(future.result())
            except (requests.RequestException, ValueError) as exc:
                failures.append(type(exc).__name__)
                continue
            temporary = journal.with_name(journal.name+'.tmp')
            temporary.write_text(json.dumps(state))
            temporary.replace(journal)
            print(f'{name}: {len(state["completed"])}/{total} ranges; '
                  f'{time.monotonic()-began:.0f}s', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} ranges failed; completed ranges are saved for resume')
    if file_hash(partial) != digest:
        raise ValueError(f'Complete file checksum mismatch: {partial}')
    partial.replace(target)
    print(f'SHA256 verified: {target}: {digest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    args = parser.parse_args()
    download(args.asset_dir, 'BAAI/Uni3D', UNI3D_HF_REVISION, 'modelzoo/uni3d-b/model.pt',
             178008217, UNI3D_SHA256)
    download(args.asset_dir, TEXT_REPO, TEXT_REVISION, 'open_clip_model.safetensors',
             10089917520, TEXT_SHA256)


if __name__ == '__main__':
    main()
