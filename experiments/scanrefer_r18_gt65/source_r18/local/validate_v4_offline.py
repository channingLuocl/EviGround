"""Offline v3/v4 geometry comparison; no fresh planner/VLM accuracy claims."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from replay_module_acceptance import replay_inference
from real_scene import structural_objects
from run import load_objects
from v4_visual import boxed_view

SIZE_CASES = [10431, 11862, 29522, 37559, 30825]
WRONG_FALLBACKS = [8614, 11647, 11901, 12579, 15945, 26783, 39009]
GRAMMAR_CASES = [7764, 30152, 7755, 4091, 29515]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-root', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.saved_root / 'results.json').read_text())['results']
    named = set(SIZE_CASES + WRONG_FALLBACKS + GRAMMAR_CASES)
    # These are outside the five-case diagnostic set, not a newly collected dataset.
    holdout = [r['csv_row'] for r in baseline if r['csv_row'] not in named and not r.get('error')][:12]
    args.output.mkdir(parents=True, exist_ok=True)
    output = []
    for row in baseline:
        if row['csv_row'] not in named.union(holdout):
            continue
        saved = json.loads((args.saved_root / f'q{row["csv_row"]:05d}' / 'run.json').read_text())
        case = {k: row[k] for k in ('csv_row', 'scene', 'query')}
        scene = args.scene_root / row['scene']
        for version in ('v3', 'v4'):
            with patch.dict('os.environ', {'SEEGROUND_V4': '1' if version == 'v4' else '0'}):
                result = replay_inference(row['query'], deepcopy(saved), scene, include_size=True)
            observation = result['size_observation']
            case[version] = {'status': result['size_diagnosis']['status'],
                'initial_ids': result['initial_candidate_ids'],
                'retained_ids': result['size_diagnosis']['candidate_ids'],
                'constraints': observation['size_constraints'],
                'measurements': observation['comparison_metrics'],
                'tentative_ids': observation['size_decision'].get('tentative_selected_ids', [])}
        if row['csv_row'] in SIZE_CASES:
            objects = load_objects(scene / 'objects_gt.json')
            with np.load(scene / 'scene.npz', allow_pickle=False) as cloud:
                view = boxed_view(objects, case['v4']['initial_ids'][:6], cloud['xyz'], cloud['rgb'], cloud['ids'],
                                  args.output, f'q{row["csv_row"]}_common_boxes', case['v4']['measurements'])
            case['render'] = view
        # Evaluation annotations are attached only after inference and model-input rendering.
        case['evaluation_only'] = {'target_id': row['target_id'],
                                  'target_retained_v3': row['target_id'] in case['v3']['retained_ids'],
                                  'target_retained_v4': row['target_id'] in case['v4']['retained_ids']}
        output.append(case)
        print(json.dumps({'row': row['csv_row'], 'v3_ids': case['v3']['retained_ids'],
                          'v4_ids': case['v4']['retained_ids'], 'tentative': case['v4']['tentative_ids']}), flush=True)
    result = {'network_requests': 0, 'policy': 'saved parser + recomputed geometry only; not fresh ReAct/VLM evaluation',
              'diagnostic_rows': sorted(named), 'additional_rows': holdout, 'cases': output,
              'target_retained_v3': sum(r['evaluation_only']['target_retained_v3'] for r in output),
              'target_retained_v4': sum(r['evaluation_only']['target_retained_v4'] for r in output)}
    (args.output / 'offline_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('network_requests', 'target_retained_v3', 'target_retained_v4')}))


if __name__ == '__main__':
    main()
