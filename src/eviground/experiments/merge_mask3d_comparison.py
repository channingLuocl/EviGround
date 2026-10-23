"""Verify all three completed scene shards, then score the fixed full evaluation."""
from eviground.paths import source_file
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from eviground.categories.category_common_v82 import file_hash
from eviground.evaluation.run_storage import write_json


def merge(root):
    output = root/'full'
    expected = None
    all_scenes = set()
    for i in range(3):
        shard = root/f'shard{i}'
        assert json.loads((shard/'status.json').read_text())['stage'] == 'completed'
        for name, digest in json.loads((shard/'FROZEN.json').read_text()).items():
            assert file_hash(shard/name) == digest
        protocol = json.loads((shard/'protocol.json').read_text())
        assert protocol['scene_shard'] == [i, 3]
        scene_list = protocol['scenes']
        common = {k: v for k, v in protocol.items() if k not in
                  ('scenes', 'scene_shard', 'unlabeled_input_sha256')}
        if expected is None:
            expected = common
        assert common == expected, 'Shards used different model/protocol/code'
        for scene in scene_list:
            assert scene not in all_scenes
            all_scenes.add(scene)
            record = json.loads((shard/f'scenes/{scene}.json').read_text())
            assert file_hash(shard/f'scenes/{scene}.masks.npz') == record['masks_sha256']
            for suffix in ('.json', '.masks.npz'):
                source = shard/f'scenes/{scene}{suffix}'
                target = output/f'scenes/{scene}{suffix}'
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert file_hash(target) == file_hash(source)
                else:
                    shutil.copyfile(source, target)
    manifest = root/'../seeground-v10/manifests/eval.json'
    assert all_scenes == set(json.loads(manifest.read_text())['scenes'])
    assert len(all_scenes) == 130
    runner = source_file('run_mask3d_comparison.py')
    assert file_hash(runner) == expected['script_sha256']
    write_json(output/'merge_audit.json', {'shards': 3, 'scenes': 130,
        'model_and_protocol_equal': True, 'all_mask_hashes_verified': True})
    subprocess.run([sys.executable, str(runner), '--manifest', str(manifest),
        '--source', str(root/'assets/Mask3D'), '--checkpoint', str(root/'assets/scannet200_val.ckpt'),
        '--inputs', str(root/'inputs'), '--output', str(output)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    merge(parser.parse_args().root)
