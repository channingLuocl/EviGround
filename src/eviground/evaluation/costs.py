"""Account actual local calls, including failed attempts, without estimating usage."""

import numpy as np


def sample_cost(run):
    calls = run.get('calls', [])
    attempts, seen = [], set()
    for index, call in enumerate(calls):
        if call.get('cache_hit'):
            continue
        rows = call.get('api_attempts') or call.get('api_diagnostic', {}).get('api_attempts')
        if not rows:
            rows = [dict(usage=call.get('usage'))]
        for position, row in enumerate(rows):
            identity = row.get('local_attempt_id') or (index, position)
            if identity in seen:
                continue
            seen.add(identity)
            attempts.append(row)
    if 'http_attempt_records' in run:
        # This journal survives checkpoint replays and counts real extra HTTP
        # attempts across process restarts. A started-only call has unknown usage.
        attempts = run['http_attempt_records']
    result = dict(actual_http_attempts=len(attempts),
        parse_calls=sum(c.get('stage') != 'react_decision' for c in calls),
        decision_calls=sum(c.get('stage') == 'react_decision' for c in calls),
        online_latency_s=run.get('online_latency_s'),
        stage_timings_s=run.get('stage_timings_s', {}))
    result['parse_http_attempts'] = sum(r.get('stage') == 'parse_query' for r in attempts)
    result['decision_http_attempts'] = sum(r.get('stage') == 'react_decision' for r in attempts)
    for name, stage in (('actual_parse_calls', 'parse_query'), ('actual_decision_calls', 'react_decision')):
        result[name] = sum(c.get('stage') == stage and not c.get('cache_hit') for c in calls)
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [(r.get('usage') or r.get('diagnostic', {}).get('usage') or {}).get(field)
                  for r in attempts]
        known = [v for v in values if type(v) is int and v >= 0]
        result[field] = sum(known) if len(known) == len(values) else None
        result[field + '_observed'] = sum(known)
        result[field + '_missing_attempts'] = len(values) - len(known)
        result[field + '_coverage'] = len(known) / len(values) if values else None
    trace = run.get('active_trace', {})
    steps = [s for s in trace.get('steps', []) if isinstance(s.get('action'), dict)]
    actions = trace.get('react_actions', [s['action'] for s in steps])
    decision_records = [c for c in calls if c.get('stage') == 'react_decision']
    result['logical_final_submissions'] = sum(a.get('action') == 'finish' for a in actions)
    result['final_submissions'] = sum(a.get('action') == 'finish' and
        (i >= len(decision_records) or not decision_records[i].get('cache_hit'))
        for i, a in enumerate(actions))
    for action in ('inspect', 'measure'):
        relevant = [s for s in steps if s['action'].get('action') == action]
        checks = [s for s in relevant if s.get('observation', {}).get('tool') in ('monitor_check', 'finish_check')]
        tools = [s for s in relevant if s not in checks]
        result[action + '_attempts'] = len(tools)
        result[action + '_successes'] = sum(not s.get('observation', {}).get('error')
            and not s.get('observation', {}).get('no_evidence_acquired') for s in tools)
        result[action + '_failures'] = len(tools) - result[action + '_successes']
        result[action + '_no_new_evidence'] = sum(bool(s.get('observation', {}).get('no_evidence_acquired')) for s in tools)
    # A failed reply and the subsequent correction are separate events. The
    # final rejected reply can exhaust the allowance without another correction.
    failed = [bool(s.get('observation', {}).get('error') or
                   s.get('observation', {}).get('no_evidence_acquired') or
                   s.get('observation', {}).get('tool') == 'finish_check') for s in steps]
    result['failed_checks'] = sum(failed)
    logical_repairs = sum(flag and result['decision_calls'] > i + 1
                          for i, flag in enumerate(failed))
    result['logical_repair_calls'] = trace.get('react_budget', {}).get('repair_calls', logical_repairs)
    result['repair_calls'] = sum(flag and i + 1 < len(decision_records) and
        not decision_records[i + 1].get('cache_hit') for i, flag in enumerate(failed))
    if not decision_records:
        result['repair_calls'] = result['logical_repair_calls']
    result['rendered_images'] = trace.get('rendered_image_count')
    result['rendered_files'] = trace.get('rendered_file_count')
    result['rendered_panels'] = trace.get('rendered_panel_count')
    result['termination_reason'] = trace.get('termination_cause') or run.get('error_stage')
    result['error'] = bool(run.get('error'))
    result['budget_exhausted'] = trace.get('termination_cause') in ('budget_exhausted', 'repair_exhausted')
    result['tool_stage_timings_s'] = trace.get('stage_timings_s', {})
    result['abstained'] = trace.get('termination_cause') == 'abstained' or (
        bool(trace.get('react_final_decision')) and trace['react_final_decision'].get('predicted_id') is None)
    previous = run.get('previous_execution_costs', [])
    for field in ('inspect_attempts', 'inspect_successes', 'inspect_failures', 'inspect_no_new_evidence',
                  'measure_attempts', 'measure_successes', 'measure_failures', 'measure_no_new_evidence',
                  'rendered_images', 'rendered_files', 'rendered_panels',
                  'actual_parse_calls', 'actual_decision_calls',
                  'repair_calls', 'failed_checks', 'final_submissions'):
        values = [r.get(field) for r in [*previous, result]]
        result[field] = sum(values) if all(v is not None for v in values) else None
    result['execution_attempts'] = 1 + len(previous)
    for timings in ('tool_stage_timings_s', 'stage_timings_s'):
        cumulative = result['cumulative_' + timings] = {}
        for cost in [*previous, result]:
            for field, value in cost.get(timings, {}).items():
                cumulative[field] = cumulative.get(field, 0) + value
    result['online_latency_observed_s'] = run.get('online_latency_observed_s', result['online_latency_s'])
    result['online_latency_complete'] = run.get('execution_complete', True) and all(
        r.get('online_latency_complete') for r in previous)
    if not result['online_latency_complete']:
        result['online_latency_s'] = None
    return result


def summarize_costs(rows, selected_total):
    costs = [r['cost'] for r in rows if r.get('cost')]
    result = dict(selected_total=selected_total, processed=len(rows), cost_records=len(costs),
        pending=selected_total - len(rows), statistics_denominator='attempted samples with observed field')
    for field in ('actual_http_attempts', 'parse_http_attempts', 'decision_http_attempts',
                  'parse_calls', 'decision_calls', 'actual_parse_calls', 'actual_decision_calls',
                  'final_submissions', 'logical_final_submissions',
                  'repair_calls', 'logical_repair_calls', 'failed_checks',
                  'inspect_attempts', 'inspect_successes', 'inspect_failures', 'inspect_no_new_evidence',
                  'measure_attempts', 'measure_successes', 'measure_failures', 'measure_no_new_evidence',
                  'rendered_images', 'rendered_files', 'rendered_panels',
                  'prompt_tokens', 'completion_tokens', 'total_tokens', 'online_latency_s'):
        values = [c[field] for c in costs if c.get(field) is not None]
        result[field] = dict(observed=len(values), missing=len(rows) - len(values),
            mean=float(np.mean(values)) if values else None)
        if field == 'online_latency_s':
            result[field].update(p50=float(np.percentile(values, 50)) if values else None,
                                 p95=float(np.percentile(values, 95)) if values else None)
    result['failure_rate'] = sum(bool(c.get('error')) for c in costs) / len(costs) if costs else None
    result['budget_exhaustion_rate'] = sum(bool(c.get('budget_exhausted')) for c in costs) / len(costs) if costs else None
    result['token_coverage'] = {}
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        measured = [c for c in costs if field + '_missing_attempts' in c]
        attempts = sum(c['actual_http_attempts'] for c in measured)
        missing = sum(c[field + '_missing_attempts'] for c in measured)
        result['token_coverage'][field] = dict(attempts=attempts, missing_attempts=missing,
            coverage=(attempts - missing) / attempts if attempts else None,
            observed_tokens=sum(c[field + '_observed'] for c in measured))
    return result
