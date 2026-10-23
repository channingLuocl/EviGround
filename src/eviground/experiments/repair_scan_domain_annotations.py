"""Convert exact duplicate ScanNet annotations once, preserving original assets."""
import argparse
import json
from pathlib import Path

from eviground.categories.category_common_v82 import file_hash
from eviground.classifiers.category_v10.data import convert_scene, SUFFIXES
from eviground.evaluation.run_storage import write_json


def unique_groups(groups):
    kept, duplicates, seen, owners = [], [], {}, {}
    ids = [int(g['objectId']) for g in groups]
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError('Invalid or repeated object IDs')
    for group in groups:
        segments = frozenset(group['segments'])
        if not segments:
            raise ValueError('Empty annotation segments')
        key = (group['label'], segments)
        if key in seen:
            duplicates.append({'retained_id': seen[key], 'duplicate_id': group['objectId'],
                               'label': group['label'], 'segments': sorted(segments)})
            continue
        if any(s in owners for s in segments):
            raise ValueError('Nonidentical overlapping annotations require separate investigation')
        seen[key] = group['objectId']
        owners.update({s: group['objectId'] for s in segments})
        kept.append(group)
    return kept, duplicates


def main(args):
    assert not (args.output / 'HOLDOUT_FROZEN.json').exists()
    assert (args.output / 'MODEL_SELECTION_FROZEN.json').exists()
    protocol = json.loads((args.output / 'protocol.json').read_text())
    records = []
    for scene in protocol['holdout_scenes']:
        original = args.raw / scene
        annotation = original / (scene + '.aggregation.json')
        value = json.loads(annotation.read_text())
        kept, duplicates = unique_groups(value['segGroups'])
        destination = args.output / 'heldout_scenes' / scene
        if destination.exists():
            if duplicates:
                raise ValueError('Repair output exists; inspect before retry')
            continue
        source = args.raw
        if duplicates:
            staging = args.output / 'annotation_staging' / scene
            staging.mkdir(parents=True, exist_ok=True)
            if any(staging.iterdir()):
                raise ValueError('Nonempty staging directory; inspect before retry')
            for suffix in SUFFIXES:
                if suffix != '.aggregation.json':
                    (staging / (scene+suffix)).symlink_to(original / (scene+suffix))
            value['segGroups'] = kept
            write_json(staging / annotation.name, value)
            source = staging.parent
        convert_scene(source, scene, destination)
        if duplicates:
            record = {'scene': scene, 'policy': 'identical label and segment set: retain first source group',
                'original_groups': len(kept)+len(duplicates), 'retained_groups': len(kept),
                'duplicates': duplicates, 'original_annotation_sha256': file_hash(annotation),
                'repaired_annotation_sha256': file_hash(source / scene / annotation.name),
                'repair_source_sha256': file_hash(Path(__file__)), 'performance_not_computed': True}
            provenance_path = destination / 'provenance.json'
            provenance = json.loads(provenance_path.read_text())
            provenance['exact_duplicate_annotation_repair'] = record
            write_json(provenance_path, provenance)
            records.append(record)
        print(json.dumps({'stage': 'annotation_conversion', 'scene': scene,
                          'exact_duplicates_removed': len(duplicates)}), flush=True)
    write_json(args.output / 'annotation_repairs.json', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
