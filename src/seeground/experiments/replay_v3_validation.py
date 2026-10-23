"""Offline policy replay and separate saved-parse syntax comparison; no model calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from seeground.grounding.active_diagnostic import singleton_prediction
from seeground.evaluation.batch_nr3d import summarize
from seeground.experiments.replay_module_acceptance import replay_inference


def singleton_replay(rows, saved_root, scene_root):
    output = []
    for row in rows:
        current = deepcopy(row)
        if not row.get('error'):
            run = json.loads((saved_root / f"q{row['csv_row']:05d}" / 'run.json').read_text())
            trace = run.get('active_diagnostic', {})
            if trace.get('stop_reason') and not run.get('error'):
                objects = json.loads((scene_root / row['scene'] / run['objects_file']).read_text())
                oid, fallback = singleton_prediction(
                    row.get('predicted_id'), trace.get('final_candidate_ids', []), objects)
                current.update(predicted_id=oid,
                    prediction_source='singleton_fallback' if fallback else
                                      'model' if oid is not None else 'abstained')
        # Evaluation annotations enter only after the production selection helper.
        current['correct'] = current.get('predicted_id') == row['target_id']
        output.append(current)
    return {'baseline': summarize(rows), 'singleton_only': summarize(output),
            'fallback_wrong_rows': [r['csv_row'] for r in output
                if r.get('prediction_source') == 'singleton_fallback' and not r['correct']],
            'results': output}


def grammar_replay(rows, saved_root, scene_root):
    cases = []
    for row in rows:
        saved = json.loads((saved_root / f"q{row['csv_row']:05d}" / 'run.json').read_text())
        case = {key: row[key] for key in ('csv_row', 'scene', 'query')}
        for backend in ('none', 'spacy'):
            with patch.dict('os.environ', {'SEEGROUND_SYNTAX_BACKEND': backend}):
                result = replay_inference(row['query'], saved, scene_root / row['scene'], include_size=True)
            diagnosis = result['size_diagnosis']
            case[backend] = {
                'parsed_query': result['parsed_query'],
                'parser_replay': result['parser_replay'],
                'candidate_ids': diagnosis['candidate_ids'], 'status': diagnosis['status'],
                'missing': diagnosis['missing'],
                'size_observation': result['size_observation'],
            }
        case['evaluation_only'] = {'target_id': row['target_id'],
            'target_retained': row['target_id'] in case['spacy']['candidate_ids']}
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--saved-root', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.snapshot.read_text())['results']
    selected = {7764, 30152, 7755, 4091, 29515}
    with patch('requests.sessions.Session.request', side_effect=AssertionError('offline network blocked')), \
            patch('subprocess.Popen', side_effect=AssertionError('offline subprocess blocked')):
        result = {'network_requests': 0,
            'limitations': 'Saved-data counterfactual and deterministic geometry replay; not fresh inference accuracy.',
            'singleton_replay': singleton_replay(rows, args.saved_root, args.scene_root),
            'grammar_replay': grammar_replay([r for r in rows if r['csv_row'] in selected],
                                            args.saved_root, args.scene_root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'singleton_only': result['singleton_replay']['singleton_only'],
        'wrong_fallback_rows': result['singleton_replay']['fallback_wrong_rows'],
        'grammar': [{k: c[k] for k in ('csv_row', 'evaluation_only')} | {
            backend: {k: c[backend][k] for k in ('status', 'candidate_ids', 'missing')}
            for backend in ('none', 'spacy')} for c in result['grammar_replay']],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
