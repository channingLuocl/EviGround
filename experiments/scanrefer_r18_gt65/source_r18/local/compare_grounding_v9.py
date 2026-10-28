"""Paired comparison of a Top-1 control and gated Top-5 run on identical queries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def comparable(manifest):
    value = deepcopy(manifest)
    categories = value['categories']
    categories['gate'].pop('top_k')
    for key in ('files', 'retained_count_histogram', 'mean_retained'):
        categories.pop(key)
    value['api'].pop('api_slot_dir', None)
    return value


def compare(top1, top5):
    dirs = [Path(top1), Path(top5)]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in dirs]
    if [m['categories']['gate']['top_k'] for m in manifests] != [1, 5]:
        raise ValueError('Supply Top-1 control followed by gated Top-5')
    if comparable(manifests[0]) != comparable(manifests[1]):
        raise ValueError('Paired runs differ in selection, source, code, models or settings')
    data = [json.loads((p/'results.json').read_text()) for p in dirs]
    expected = {r['csv_row'] for r in manifests[0]['rows']}
    rows = [{r['csv_row']: r for r in d['results']} for d in data]
    if any(set(r) != expected or len(r) != len(d['results']) for r, d in zip(rows, data)):
        raise ValueError('Both runs must finish the entire identical selection')
    gained = [i for i in sorted(expected) if not rows[0][i]['correct'] and rows[1][i]['correct']]
    lost = [i for i in sorted(expected) if rows[0][i]['correct'] and not rows[1][i]['correct']]
    return {'queries': len(expected), 'top1': data[0]['summary'], 'gated_top5': data[1]['summary'],
            'gained': len(gained), 'lost': len(lost), 'net_correct': len(gained)-len(lost),
            'gained_csv_rows': gained, 'lost_csv_rows': lost}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('top1', type=Path)
    parser.add_argument('top5', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.top1, args.top5), ensure_ascii=False, indent=2))
