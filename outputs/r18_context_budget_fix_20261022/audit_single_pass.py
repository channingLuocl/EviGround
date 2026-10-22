"""Audit the complete selected denominator and real predictions, without running inference."""
import argparse
import hashlib
import json
from pathlib import Path


def audit_run(directory, expected_rows, expected_code, runtime, expected_total=7805):
    directory, runtime = Path(directory), Path(runtime)
    manifest = json.loads((directory / 'manifest.json').read_text())
    issues = []
    if len(expected_rows) != expected_total or manifest['rows'] != expected_rows:
        issues.append('Selection differs from the complete expected cohort')
    if manifest['code_sha256'] != expected_code:
        issues.append('Inference code differs from the evaluated development version')
    for name, digest in expected_code.items():
        path = runtime / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            issues.append('Runtime source changed: ' + name)
    checks = {
        'model': manifest['models']['SEEGROUND_MODEL'] == 'gpt-6.1-sol',
        'revision': manifest['r8_grounding_revision'] == 'r18',
        'top5': manifest['categories']['gate']['top_k'] == 5,
        'policy': manifest['evaluation_policy'] == 'fixed_predictions_no_historical_correct_reuse',
        'provider': manifest['api']['provider'] == 'openai',
        'thinking_floor': manifest['api']['thinking_token_floor'] == 8192,
        'local_validation': manifest['api']['local_schema_validation'] is True,
        'json_object': manifest['api']['request_options']['response_format']['type'] == 'json_object',
    }
    issues.extend('Incorrect configuration: ' + key for key, valid in checks.items() if not valid)
    expected_ids = {row['csv_row'] for row in expected_rows}
    if len(expected_ids) != len(expected_rows):
        issues.append('Duplicate selected rows')
    recorded, correct, errors, predicted = [], 0, 0, 0
    for row in expected_rows:
        case = directory / f"q{row['csv_row']:05d}"
        evaluation = case / 'evaluation.json'
        if not evaluation.is_file():
            continue
        result = json.loads(evaluation.read_text())
        recorded.append(row['csv_row'])
        if any(result.get(key) != value for key, value in row.items()):
            issues.append(f"Evaluation metadata mismatch: {row['csv_row']}")
        prediction = case / 'prediction.json'
        if not prediction.is_file():
            errors += 1
            if not result.get('error') or result.get('predicted_id') is not None or result.get('correct') is not False:
                issues.append(f"Missing prediction is not recorded as a failure: {row['csv_row']}")
            continue
        data = json.loads(prediction.read_text())
        oid = data.get('predicted_id')
        expected_correct = type(oid) is int and oid == row['target_id']
        correct += int(expected_correct)
        predicted += int(type(oid) is int)
        errors += int(bool(result.get('error')))
        if result.get('predicted_id') != oid or result.get('correct') is not expected_correct:
            issues.append(f"Reported outcome differs from real prediction: {row['csv_row']}")
        if data.get('gt_categories_supplied_to_model') is not False or data.get('query_target_id_supplied_to_model') is not False:
            issues.append(f"Missing/invalid GT-input audit flags: {row['csv_row']}")
        if data.get('diagnosis_policy') != 'predicted_category_multimodal_react':
            issues.append(f"Unexpected inference path: {row['csv_row']}")
        if oid is not None and oid not in data.get('candidate_target_ids', []):
            issues.append(f"Prediction outside actual candidate pool: {row['csv_row']}")
        trace = data.get('active_trace', {})
        if trace.get('stop_reason') != 'r18_react_finish' or trace.get('r18_final_decision', {}).get('predicted_id') != oid:
            issues.append(f"Prediction lacks its completed ReAct decision: {row['csv_row']}")
    missing = sorted(expected_ids - set(recorded))
    complete = not missing and len(recorded) == expected_total and not issues
    accuracy = correct / expected_total
    return {
        'expected_total': expected_total, 'recorded': len(recorded), 'correct': correct,
        'predictions': predicted, 'fatal_errors': errors,
        'full_selected_denominator_accuracy': accuracy,
        'missing_rows': missing, 'structural_issues': issues,
        'verified_complete': complete,
        'threshold': 0.66, 'threshold_passed': complete and accuracy >= 0.66,
        'cumulative': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime', type=Path, help='Inference local/ directory; defaults to the checked-out source')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    root = base.parents[1]
    selection = json.loads((root / 'reproductions/SeeGround/local/validation/nr3d_val_full.json').read_text())
    frozen = json.loads((base / 'freeze_manifest.json').read_text())
    runtime = args.runtime or root / 'reproductions/SeeGround/local'
    report = audit_run(args.run, selection['rows'], frozen['inference']['code_sha256'], runtime)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('missing_rows', 'structural_issues')}, ensure_ascii=False))
