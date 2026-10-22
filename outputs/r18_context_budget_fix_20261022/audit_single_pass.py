"""Audit the complete selected denominator and real predictions, without running inference."""
import argparse
import hashlib
import json
from pathlib import Path


def audit_run(directory, expected_rows, expected_code, runtime, expected_total=7805,
              expected_config=None):
    directory, runtime = Path(directory), Path(runtime)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if expected_config is None:
        expected_config = {'model': 'gpt-6.1-sol', 'provider': 'openai',
                           'thinking_token_floor': 8192}
    pool_field = expected_config.get('candidate_pool_field', 'candidate_target_ids')
    if pool_field not in ('candidate_target_ids', 'category_target_ids'):
        raise ValueError('Unsupported candidate pool provenance')
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
        'model': manifest['models']['SEEGROUND_MODEL'] == expected_config['model'],
        'revision': manifest['r8_grounding_revision'] == 'r18',
        'top5': manifest['categories']['gate']['top_k'] == 5,
        'policy': manifest['evaluation_policy'] == 'fixed_predictions_no_historical_correct_reuse',
        'provider': manifest['api']['provider'] == expected_config['provider'],
        'thinking_floor': manifest['api'].get('thinking_token_floor') == expected_config['thinking_token_floor'],
        'local_validation': manifest['api'].get('local_schema_validation') ==
                            expected_config.get('local_schema_validation', True),
        'json_format': manifest['api']['request_options']['response_format']['type'] ==
                       expected_config.get('request_options', {}).get('response_format', {}).get('type', 'json_object'),
    }
    for key in ('request_options', 'token_parameter', 'allow_http', 'structured_outputs'):
        if key in expected_config:
            checks[key] = manifest['api'].get(key) == expected_config[key]
    issues.extend('Incorrect configuration: ' + key for key, valid in checks.items() if not valid)
    expected_ids = {row['csv_row'] for row in expected_rows}
    if len(expected_ids) != len(expected_rows):
        issues.append('Duplicate selected rows')
    recorded, correct, errors, predicted, category_abstentions = [], 0, 0, 0, 0
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
        run_path = case / 'run.json'
        if run_path.is_file():
            run = json.loads(run_path.read_text())
            for call in run.get('calls', []):
                if call.get('model') is not None and call['model'] != expected_config['model']:
                    issues.append(f"Mixed inference model: {row['csv_row']}")
                if call.get('provider') is not None and call['provider'] != expected_config['provider']:
                    issues.append(f"Mixed inference provider: {row['csv_row']}")
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
        if oid is not None and oid not in data.get(pool_field, []):
            issues.append(f"Prediction outside actual candidate pool: {row['csv_row']}")
        trace = data.get('active_trace', {})
        category_abstention = (
            oid is None and data.get('bbox_3d') is None
            and data.get(pool_field) == [] and data.get('category_target_ids') == []
            and data.get('visual_fallback_target_ids', [] if pool_field == 'category_target_ids' else None) == []
            and data.get('prediction_source') == 'abstained'
            and trace.get('stop_reason') == 'category_gate_abstention'
            and data.get('diagnosis', {}).get('candidate_ids') == []
            and data.get('diagnosis', {}).get('missing') == [
                'no retained category hypothesis matches the parsed target'])
        category_abstentions += int(category_abstention)
        if not category_abstention and data.get('diagnosis_policy') != 'predicted_category_multimodal_react':
            issues.append(f"Unexpected inference path: {row['csv_row']}")
        if not category_abstention and (trace.get('stop_reason') != 'r18_react_finish'
                or trace.get('r18_final_decision', {}).get('predicted_id') != oid):
            issues.append(f"Prediction lacks its completed ReAct decision: {row['csv_row']}")
    missing = sorted(expected_ids - set(recorded))
    complete = not missing and len(recorded) == expected_total and not issues
    accuracy = correct / expected_total
    return {
        'expected_total': expected_total, 'recorded': len(recorded), 'correct': correct,
        'predictions': predicted, 'fatal_errors': errors, 'category_gate_abstentions': category_abstentions,
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
    parser.add_argument('--freeze-manifest', type=Path, help='Explicit frozen source and expected provider configuration')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    root = base.parents[1]
    selection = json.loads((root / 'reproductions/SeeGround/local/validation/nr3d_val_full.json').read_text())
    frozen = json.loads((args.freeze_manifest or base / 'freeze_manifest.json').read_text())
    runtime = args.runtime or root / 'reproductions/SeeGround/local'
    report = audit_run(args.run, selection['rows'], frozen['inference']['code_sha256'], runtime,
                       expected_config=frozen.get('expected_config'))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('missing_rows', 'structural_issues')}, ensure_ascii=False))
