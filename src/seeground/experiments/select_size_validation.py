"""Reproducibly select fresh NR3D size-language validation cases."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import re
from zipfile import ZipFile

from seeground.rendering.run import ROOT

SIZE = re.compile(r'\b(small(?:est|er)?|large(?:st|r)?|big(?:gest|ger)?|little|'
                  r'tall(?:est|er)?|short(?:est|er)?|skinny|narrow(?:est|er)?|'
                  r'thin(?:nest|ner)?|wide(?:st|r)?|thick(?:est|er)?)\b', re.I)
SUPERLATIVES = {'smallest', 'largest', 'biggest', 'tallest', 'shortest',
                'narrowest', 'thinnest', 'widest', 'thickest'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=300)
    args = parser.parse_args()
    source = ROOT / 'local_outputs/downloads/nr3d_official.csv'
    old = json.loads((ROOT / 'data/validation/nr3d_size_cases.json').read_text())
    excluded = {row['scene'] for row in old['rows']} | {'scene0011_00'}
    validation = set((ROOT / 'data/scannet/scannetv2_val.txt').read_text().split())
    pools = defaultdict(list)
    seen = set()
    with ZipFile(ROOT / 'local_outputs/downloads/olt.zip') as archive:
        tables = {scene: json.loads(archive.read(
            f'seeground_object_lookup_table/nr3d/gt/{scene}.json'))
            for scene in validation - excluded
            if f'seeground_object_lookup_table/nr3d/gt/{scene}.json' in archive.namelist()
            and f'seeground_object_lookup_table/nr3d/pred/{scene}.json' in archive.namelist()}
        with source.open(newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle), 2):
                scene, query = row['scan_id'], row['utterance']
                words = [word.casefold() for word in SIZE.findall(query)]
                if scene not in tables or row['correct_guess'].casefold() != 'true' or not words:
                    continue
                stimulus = row['stimulus_id'].split('-')
                candidate_ids = list(map(int, stimulus[3:]))
                known = {obj['bbox_id'] for obj in tables[scene]}
                if len(candidate_ids) < 2 or not set(candidate_ids).issubset(known):
                    continue
                target = int(row['target_id'])
                if target != candidate_ids[0]:
                    raise ValueError(f'Stimulus target mismatch at CSV row {index}')
                key = (scene, target, ' '.join(query.casefold().split()).rstrip('.!?'))
                if key in seen:
                    continue
                seen.add(key)
                category = stimulus[1].replace('_', ' ')
                aliases = {category, next(o['target'] for o in tables[scene]
                                          if o['bbox_id'] == target)}
                aliases |= {alias.rstrip('s') for alias in aliases}
                aliases |= {alias + 's' for alias in aliases}
                simple = any(re.fullmatch(
                    r'\s*(?:(?:choose|select|find|pick|click)\s+|it is\s+)?(?:the\s+)?'
                    + SIZE.pattern + r'\s+' + re.escape(alias) + r'\s*[.!?]?\s*', query, re.I)
                    for alias in aliases)
                stage = ('simple_superlative' if simple and set(words) & SUPERLATIVES else
                         'simple_relative' if simple else 'compound_size_language')
                # Keyword selection can include size descriptors of anchors; retain this flag.
                target_likely = simple or any(re.search(
                    SIZE.pattern + r'(?:\s+[a-z]+){0,3}\s+' + re.escape(alias) + r'\b',
                    query, re.I) or re.search(
                    r'\b' + re.escape(alias) + r'\s+(?:is|that is|which is)\s+'
                    r'(?:very\s+)?' + SIZE.pattern, query, re.I)
                    for alias in aliases)
                if not target_likely:
                    continue
                pools[scene].append({'csv_row': index, 'scene': scene, 'query': query,
                    'target_id': target, 'instance_type': category,
                    'stimulus_id': row['stimulus_id'], 'stimulus_candidate_ids': candidate_ids,
                    'stage': stage, 'size_words': words,
                    'size_form': ('comparative' if any(w.endswith('er') for w in words) else
                                  'superlative' if set(words) & SUPERLATIVES else 'adjective'),
                    'size_scope': 'target_likely' if target_likely else 'needs_scope_review'})
    rng = random.Random(20261006)
    scenes = sorted(pools)
    rng.shuffle(scenes)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    # Round-robin caps scene dominance without selecting on expected model correctness.
    while len(selected) < args.count and any(pools.values()):
        for scene in scenes:
            if pools[scene] and len(selected) < args.count:
                selected.append(pools[scene].pop())
    if len(selected) != args.count:
        raise ValueError(f'Only {len(selected)} eligible cases')
    selected.sort(key=lambda row: row['stage'] == 'compound_size_language')
    counts = {'cases': len(selected), 'scenes': len({r['scene'] for r in selected}),
              'categories': dict(Counter(r['instance_type'] for r in selected)),
              'stages': dict(Counter(r['stage'] for r in selected)),
              'size_forms': dict(Counter(r['size_form'] for r in selected)),
              'size_scope': dict(Counter(r['size_scope'] for r in selected)),
              'size_words': dict(Counter(w for r in selected for w in r['size_words']))}
    manifest = {'purpose': 'Fresh exploratory NR3D size-language validation; not a full benchmark.',
        'selection': 'ScanNet validation split, correct_guess=True, size keyword, >=2 stimulus '
                     'candidates, GT/pred OLT available and IDs validated, duplicate utterances '
                     'removed, seeded scene round-robin. No metric agreement/model outcomes used. '
                     'A size descriptor must occur near a target-category alias or in a target '
                     'copular clause. This lexical filter still needs scope review; no parser '
                     'or model prediction is used for selection.',
        'seed': 20261006, 'excluded_scenes': sorted(excluded),
        'csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'summary': counts, 'rows': selected}
    output = ROOT / f'data/validation/nr3d_size_{args.count}.json'
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(output)


if __name__ == '__main__':
    main()
