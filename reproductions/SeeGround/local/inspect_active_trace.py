"""Print an active grounding trace without loading a model or making API calls."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', type=Path, help='prediction.json from active_diagnostic')
    args = parser.parse_args()
    prediction = json.loads(args.prediction.read_text())
    trace = prediction['active_trace']
    summary = {key: prediction.get(key) for key in (
        'query', 'predicted_id', 'decision_status', 'accepted', 'api_calls', 'pipeline_latency_s')}
    summary.update({key: trace.get(key) for key in (
        'stop_reason', 'category_candidates', 'final_candidate_ids', 'tool_budget',
        'action_attempt_budget', 'action_attempts', 'successful_tools', 'rejected_actions')})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for index, step in enumerate(trace.get('steps', []), 1):
        observation = step.get('observation', {})
        record = {'step': index, 'action': step.get('action'),
                  'remaining_candidate_ids': step.get('remaining_candidate_ids')}
        for key in ('error', 'no_evidence_acquired', 'candidate_update',
                    'candidate_evidence', 'checks', 'image', 'unmarked_focus_ids'):
            if key in observation:
                record[key] = observation[key]
        if 'evidence_state' in step:
            record['evidence_state'] = step['evidence_state']
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print('Final evidence:')
    print(json.dumps(trace.get('final_evidence_state', {}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
