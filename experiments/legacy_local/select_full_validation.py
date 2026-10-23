"""Build the full Nr3D validation selection and verify it against released SeeGround results."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from run import ROOT
from run_size_validation import write_json


def select(csv_path, split_path, olt_path, baseline_path, size_path):
    validation = set(split_path.read_text().split())
    size_rows = {r['csv_row']: r for r in json.loads(size_path.read_text())['rows']}
    rows = []
    with ZipFile(olt_path) as archive, csv_path.open(newline='') as handle:
        tables = {}
        for index, source in enumerate(csv.DictReader(handle), 2):
            scene = source['scan_id']
            if scene not in validation or source['mentions_target_class'].lower() != 'true':
                continue
            if scene not in tables:
                for kind in ('gt', 'pred'):
                    archive.getinfo(f'seeground_object_lookup_table/nr3d/{kind}/{scene}.json')
                tables[scene] = {o['bbox_id'] for o in json.loads(archive.read(
                    f'seeground_object_lookup_table/nr3d/gt/{scene}.json'))}
            stimulus = source['stimulus_id'].split('-')
            candidates = list(map(int, stimulus[3:]))
            target = int(source['target_id'])
            if (not candidates or candidates[0] != target or len(candidates) != int(stimulus[2])
                    or not set(candidates).issubset(tables[scene])):
                raise ValueError(f'Invalid stimulus or OLT IDs at CSV row {index}')
            row = {'csv_row': index, 'scene': scene, 'query': source['utterance'],
                   'target_id': target, 'instance_type': source['instance_type'],
                   'stimulus_id': source['stimulus_id'], 'stimulus_candidate_ids': candidates,
                   'stage': 'nr3d_full', 'size_form': 'not_classified'}
            if index in size_rows:
                original = size_rows[index]
                for key in ('scene', 'query', 'target_id', 'stimulus_id', 'stimulus_candidate_ids'):
                    if row[key] != original[key]:
                        raise ValueError(f'300-case annotation mismatch at CSV row {index}')
                row.update(original)
            tokens = ast.literal_eval(source['tokens'])
            row['easy'] = len(candidates) <= 2
            row['view_dep'] = bool(set(tokens) & {'front', 'behind', 'back', 'right', 'left',
                'facing', 'leftmost', 'rightmost', 'looking', 'across'})
            rows.append(row)
    expected = Counter()
    with ZipFile(baseline_path) as archive:
        for name in archive.namelist():
            if ('/nr3d/val/' in name and '/pred/' in name and name.endswith('.json')
                    and not name.startswith('__MACOSX/')):
                for row in json.loads(archive.read(name)):
                    expected[(Path(name).stem, row['gt_id'], row['query'])] += 1
    actual = Counter((r['scene'], r['target_id'], r['query']) for r in rows)
    if not expected or actual != expected:
        raise ValueError('Selection differs from the author-released Nr3D validation results')
    if not set(size_rows).issubset({r['csv_row'] for r in rows}):
        raise ValueError('Full selection does not include all 300 existing cases')
    return {'purpose': 'Full SeeGround Nr3D validation query set; local v7.1 inference adaptation.',
        'selection': 'ScanNet validation scenes, mentions_target_class=True; no correct_guess, '
                     'size-language, deduplication or model-outcome filtering.',
        'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'author_results_url': 'https://github.com/user-attachments/files/18056533/seeground_results.zip',
        'author_results_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'author_queries_match': True,
        'summary': {'cases': len(rows), 'scenes': len({r['scene'] for r in rows}),
                    'overlap_size_300': len(size_rows)}, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'local/validation/nr3d_val_full.json')
    args = parser.parse_args()
    downloads = ROOT / 'local_outputs/downloads'
    manifest = select(downloads / 'nr3d_official.csv', ROOT / 'data/scannet/scannetv2_val.txt',
                      downloads / 'olt.zip', downloads / 'seeground_results.zip',
                      ROOT / 'local/validation/nr3d_size_300.json')
    write_json(args.output, manifest)
    print(json.dumps(manifest['summary'], ensure_ascii=False))
    print(args.output)


if __name__ == '__main__':
    main()
