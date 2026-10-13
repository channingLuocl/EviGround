"""Shared, quoted size conditions and conservative measurement comparisons."""
import math
import re

from evidence import canonical_category, retrieval_category


SIZE_RULE_VERSION = 'relative-size-v1'
TIE_RELATIVE_TOLERANCE = .05
COMPARATIVE_RELATIVE_MARGIN = .10
ADJECTIVE_RELATIVE_MARGIN = .20
_LENGTH_ABSOLUTE_MARGINS = {'tie': .02, 'comparative': .03, 'adjective': .05}
_SUPERLATIVES = {'smallest', 'largest', 'biggest', 'tallest', 'highest',
                 'shortest', 'lowest'}
_COMPARATIVES = {'smaller', 'larger', 'bigger', 'taller', 'shorter', 'longer'}
_MIN_WORDS = {'small', 'little', 'smaller', 'smallest', 'short', 'shorter',
              'shortest', 'low', 'lowest'}
SIZE_WORDS = (_SUPERLATIVES | _COMPARATIVES |
              {'small', 'large', 'big', 'little', 'tall', 'short', 'high',
               'low'})


def category_pattern(category):
    """The supplied category and its ordinary plural, without stripping modifiers."""
    category = canonical_category(category)
    if category.endswith('shelf'):
        plural = category[:-5] + 'shelves'
    elif re.search(r'[^aeiou]y$', category):
        plural = category[:-1] + 'ies'
    elif category.endswith(('s', 'x', 'z', 'ch', 'sh')):
        plural = category + 'es'
    else:
        plural = category + 's'
    return '(?:' + '|'.join(re.escape(value) for value in (plural, category)) + ')'


def _scope(query, word, category):
    explicit_reference = re.search(r'\bthan\b[^.!?;]*', query, re.I)
    if word in _COMPARATIVES and explicit_reference:
        return {'scope_kind': 'unbound_explicit_scope',
                'scope_phrase': explicit_reference.group().strip()}
    noun = category_pattern(category)
    match = re.search(r'\b' + re.escape(word) + r'(?:\s+' + noun + r')?\s+'
                      r'(?P<scope>of\s+(?:the\s+)?(?P<two>two\s+)?'
                      r'(?P<modifiers>(?:[a-z-]+\s+)*?)' + noun + r')\b', query, re.I)
    if not match:
        pair = re.search(r'\b' + re.escape(word) + r'(?:\s+' + noun + r')?\s+'
                         r'(?P<scope>of\s+the\s+two)\b', query, re.I)
        if pair:
            if re.match(r'\s+[a-z]', query[pair.end():], re.I):
                return {'scope_kind': 'unbound_explicit_scope',
                        'scope_phrase': query[pair.start('scope'):].strip().rstrip('.!?')}
            return {'scope_kind': 'explicit_pair', 'scope_phrase': pair['scope'],
                    'required_member_count': 2}
        unresolved = re.search(r'\b' + re.escape(word) + r'(?:\s+' + noun + r')?\s+'
                               r'(?P<scope>of\b[^.!?;]*)', query, re.I)
        return ({'scope_kind': 'unbound_explicit_scope', 'scope_phrase': unresolved['scope'].strip()}
                if unresolved else {'scope_kind': 'full_category', 'scope_phrase': ''})
    remainder = query[match.end():]
    if re.match(r'\s+(?:that|which|near|beside|between|behind|above|below|with|in|at|on)\b', remainder, re.I):
        return {'scope_kind': 'unbound_explicit_scope',
                'scope_phrase': (match['scope'] + remainder).strip().rstrip('.!?')}
    modifiers = match['modifiers'].strip()
    return {'scope_kind': ('explicit_pair' if match['two'] else
                           'attribute_subset' if modifiers else 'full_category'),
            'scope_phrase': match['scope'], 'scope_modifiers': modifiers,
            **({'required_member_count': 2} if match['two'] else {})}


def size_specs(phrase, category, *, query='', role='target', source=None):
    """Describe size semantics while retaining the actual quoted adjective."""
    words = dict.fromkeys(re.findall(r'\b\w+\b', phrase.casefold()))
    records = []
    category_name = retrieval_category(category)
    for word in words:
        if word not in SIZE_WORDS:
            continue
        form = ('superlative' if word in _SUPERLATIVES else
                'comparative' if word in _COMPARATIVES else 'adjective')
        if word in {'tall', 'tallest', 'taller', 'high', 'highest',
                    'short', 'shortest', 'shorter', 'low', 'lowest'}:
            metric, reason = 'height_m', 'vertical extent along gravity, not center elevation'
        elif word == 'longer':
            metric, reason = ('horizontal_oriented_long_extent_m',
                              'gravity-aligned horizontal oriented extent, not world X')
        elif category_name in {'door', 'window'}:
            metric, reason = ('vertical_face_area_m2',
                              'door/window face width times height; thickness does not govern size')
        else:
            metric, reason = ('volume_m3',
                              'overall size proxy, oriented volume preferred; footprint ordering must agree')
        descriptive = form == 'adjective' and word not in {'small', 'large', 'big', 'little'}
        records.append({
            'attribute': word, 'word': word, 'original_quote': phrase,
            'source': source or 'validated quoted attribute', 'role': role,
            'category': category, 'form': form,
            'interpretation': ('descriptive_attribute' if descriptive else
                               'contextual_set_comparison' if form == 'adjective' else
                               'set_comparison' if form == 'superlative' else 'pair_or_set_comparison'),
            'metric': metric, 'metric_reason': reason,
            'direction': 'min' if word in _MIN_WORDS else 'max',
            **_scope(query, word, category), 'rule_version': SIZE_RULE_VERSION,
        })
    return records


def relative_height_between(query, category):
    """Recover the explicit local opposition without asserting absolute tallness."""
    noun = category_pattern(category)
    match = re.fullmatch(r'\s*(?:(?:find|choose|select|pick|click)\s+)?(?:the\s+)?'
                         r'(?P<target>short)\s+' + noun + r'\s+'
                         r'(?P<relation>(?:in\s+)?between\s+(?:the\s+)?two\s+'
                         r'(?P<reference>tall)\s+' + noun + r')\s*[.!?]?\s*', query, re.I)
    if not match:
        return None
    return {'target_word': match['target'], 'reference_word': match['reference'],
            'target_phrase': match['target'], 'reference_phrase': match['reference'],
            'relation_phrase': match['relation'], 'required_reference_count': 2,
            'interpretation': 'reference_comparison', 'metric': 'height_m',
            'source': 'quoted short/tall opposition in between clause'}


def _usable(row, metric, legacy_superlative=False):
    if not row or row.get('status') != 'known':
        return False
    value = row.get(metric)
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        return False
    from v4_policy import enabled
    if enabled('size') and metric in ('volume_m3', 'oriented_volume_m3', 'height_m'):
        completeness = row.get('observed_completeness', {})
        if completeness.get('status') == 'suspected_incomplete' or (completeness.get('surface_only_risk')
                and completeness.get('status') != 'supported'):
            return False
    if legacy_superlative and metric in {'volume_m3', 'footprint_m2'} and not enabled('size'):
        return row.get('measurement_reliability', 'usable') == 'usable'
    reliability = row.get('metric_reliability')
    if isinstance(reliability, dict):
        entry = reliability.get(metric, {})
        return entry.get('status') == 'usable' if isinstance(entry, dict) else entry == 'usable'
    if metric in {'horizontal_oriented_long_extent_m', 'vertical_face_area_m2',
                  'oriented_volume_m3', 'oriented_footprint_m2', 'body_plane_area_m2',
                  'body_plane_long_extent_m', 'body_plane_short_extent_m', 'body_plane_thickness_m'}:
        return False
    return row.get('measurement_reliability', 'usable' if legacy_superlative else 'unknown') == 'usable'


def _margin(metric, kind, a, b):
    relative = {'tie': TIE_RELATIVE_TOLERANCE,
                'comparative': COMPARATIVE_RELATIVE_MARGIN,
                'adjective': ADJECTIVE_RELATIVE_MARGIN}[kind]
    absolute = (.0005 if metric.endswith('_m3') else .005 if metric.endswith('_m2')
                else _LENGTH_ABSOLUTE_MARGINS[kind])
    return max(absolute, relative * max(abs(a), abs(b)))


def _proxy_ambiguity(rows, direction, volume, footprint):
    if not all(row.get(volume) is not None and row.get(footprint) is not None for row in rows):
        return None
    choose = min if direction == 'min' else max
    selected = {metric: sorted(row['id'] for row in rows
                               if abs(row[metric] - choose(item[metric] for item in rows)) <= 1e-9)
                for metric in (volume, footprint)}
    if selected[volume] != selected[footprint]:
        return {'status': 'unknown', 'proxy_selected_ids': selected,
                'reason': 'unqualified size metric is ambiguous: volume and footprint extremes disagree'}
    return None


def _height_structure_ambiguity(rows):
    """Same observed top and shifted bottoms cannot certify full object height."""
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            if all(row.get('measurement_completeness_certified') is True for row in (first, second)):
                continue
            bounds = []
            for row in (first, second):
                lower, upper = row.get('min_xyz'), row.get('max_xyz')
                bottom = row.get('bottom_z_m', lower[2] if isinstance(lower, (list, tuple)) and len(lower) == 3 else None)
                top = row.get('top_z_m', upper[2] if isinstance(upper, (list, tuple)) and len(upper) == 3 else None)
                bounds.append((bottom, top))
            if any(value is None for bound in bounds for value in bound):
                continue
            a, b = first['height_m'], second['height_m']
            if (abs(bounds[0][1] - bounds[1][1]) <= _margin('height_m', 'tie', a, b) and
                    abs(bounds[0][0] - bounds[1][0]) > _margin('height_m', 'comparative', a, b)):
                return {'status': 'unknown', 'candidate_ids': sorted([first['id'], second['id']]),
                        'reason': 'common observed top and shifted bottoms leave full-height measurement ambiguous'}
    return None


def _tabletop_floor_ambiguity(rows):
    """A height comparison needs supported, mutually comparable floor planes."""
    planes = []
    for row in rows:
        plane = row.get('tabletop_evidence', {}).get('floor_plane', {})
        normal, point = plane.get('normal_xyz'), plane.get('point_xyz')
        if (plane.get('status') != 'usable' or not isinstance(normal, (list, tuple))
                or not isinstance(point, (list, tuple)) or len(normal) != 3 or len(point) != 3):
            return {'status': 'unknown', 'candidate_ids': [r['id'] for r in rows],
                    'reason': 'tabletop comparison lacks audited local floor plane support'}
        planes.append((normal, point))
    first_normal, first_point = planes[0]
    for normal, point in planes[1:]:
        agreement = sum(a * b for a, b in zip(first_normal, normal))
        offset = abs(sum(n * (b - a) for n, a, b in zip(first_normal, first_point, point)))
        if agreement < math.cos(math.radians(5)) or offset > .04:
            return {'status': 'unknown', 'candidate_ids': [r['id'] for r in rows],
                    'reason': 'different supported local floor planes are not comparable at declared tolerance',
                    'normal_cosine': agreement, 'plane_offset_m': offset,
                    'policy': {'maximum_normal_difference_deg': 5, 'maximum_plane_offset_m': .04}}
    return None


def evaluate_size_condition(spec, measurements, comparison_pool_ids, *, scope_bound,
                            candidate_ids=None, scope_source='full_category', scope_unknown_ids=()):
    """Return per-condition proof states for one frozen comparison set."""
    from v4_policy import v7_enabled
    if v7_enabled() and spec.get('measurement_policy') == 'provided_instance_box':
        from v7_evidence import evaluate_box_size
        return evaluate_box_size(spec, measurements, comparison_pool_ids, scope_bound=scope_bound,
            candidate_ids=candidate_ids, scope_source=scope_source, scope_unknown_ids=scope_unknown_ids)
    pool_ids = sorted(set(comparison_pool_ids))
    candidate_ids = pool_ids if candidate_ids is None else list(dict.fromkeys(candidate_ids))
    by_id = {row['id']: row for row in measurements}
    rows = [by_id[oid] for oid in pool_ids if oid in by_id]
    metric = spec['metric']
    from v4_policy import enabled
    if (enabled('size') and metric == 'oriented_footprint_m2' and rows
            and spec.get('metric_choice_source') == 'category surface structure; declared v4 policy'
            and all(_usable(r, 'oriented_volume_m3') and _usable(r, 'oriented_footprint_m2') for r in rows)
            and _proxy_ambiguity(rows, spec['direction'], 'oriented_volume_m3', 'oriented_footprint_m2') is None):
        metric = 'oriented_volume_m3'
        spec = {**spec, 'tentative_metric': False,
                'metric_reason': 'reliable observed volume with agreeing horizontal area ordering',
                'metric_choice_source': 'reliable volume and agreeing area; static fallback unnecessary'}
    legacy_superlative = spec['form'] == 'superlative'
    if metric == 'volume_m3' and not legacy_superlative:
        metric = 'oriented_volume_m3'
    metric_reason = spec['metric_reason']
    if metric == 'volume_m3':
        metric_reason = 'existing explicit superlative uses AABB volume; AABB footprint extreme must agree'
    elif metric == 'oriented_volume_m3':
        metric_reason = ('relative overall size uses observed oriented volume; conflicting area ordering remains tentative'
                         if enabled('size') else
                         'relative overall size uses gravity-aligned oriented volume; oriented footprint extreme must agree')
    result = {**spec, 'metric': metric, 'metric_reason': metric_reason, 'comparison_pool_ids': pool_ids,
              'comparison_pool_frozen': bool(scope_bound), 'scope_source': scope_source,
              'scope_status': 'bound' if scope_bound else 'unknown',
              'scope_unknown_ids': sorted(set(scope_unknown_ids)),
              'selected_ids': [], 'tie_ids': [], 'candidate_evidence': [],
              'unknown_measurement_ids': [], 'measurement_sources': {
                  str(row['id']): row.get('source', 'unspecified') for row in rows},
              'status': 'unknown', 'reason': None}

    def finish(reason, statuses=None):
        result['reason'] = reason
        statuses = statuses or {}
        result['candidate_evidence'] = [
            {'candidate_id': oid, 'status': statuses.get(oid, 'unknown'),
             'reason': (reason if statuses.get(oid, 'unknown') == 'unknown' else
                        'comparison condition satisfied' if statuses[oid] == 'pass' else
                        'reliable measurement contradicts this size condition'),
             'value': by_id.get(oid, {}).get(metric), 'metric': metric,
             'measurement_reliability': by_id.get(oid, {}).get('metric_reliability', {}).get(
                 metric, by_id.get(oid, {}).get('measurement_reliability', 'unknown'))}
            for oid in candidate_ids]
        result['selected_ids'] = sorted(oid for oid, state in statuses.items() if state == 'pass')
        result['status'] = 'known' if statuses and all(state != 'unknown' for state in statuses.values()) else 'unknown'
        if spec.get('tentative_metric') and result['selected_ids']:
            result['tentative_selected_ids'] = result['selected_ids'][:]
            result['selected_ids'] = []
            result['status'] = 'unknown'
            result['reason'] = 'alternative size proxy supports a tentative ordering; original size remains unproved'
            for item in result['candidate_evidence']:
                item['tentative_status'] = item['status']
                item['status'] = 'unknown'
                item['reason'] = result['reason']
        return result

    if not scope_bound or scope_unknown_ids:
        return finish('size comparison scope is not fully bound')
    if spec.get('scope_kind') == 'unbound_explicit_scope':
        result['scope_status'] = 'unknown'
        result['comparison_pool_frozen'] = False
        return finish('quoted comparison scope is not supported or fully bound')
    required = spec.get('required_member_count')
    if required is not None and len(pool_ids) != required:
        return finish('explicit comparison member count is not bound: expected ' + str(required))
    if spec['interpretation'] == 'descriptive_attribute':
        return finish('descriptive ' + spec['word'] + ' has no calibrated criterion; an extreme is not an attribute proof')
    if len(pool_ids) < 2:
        return finish('size comparison requires at least two bound members')
    unknown = [oid for oid in pool_ids if not _usable(by_id.get(oid), metric, legacy_superlative)]
    result['unknown_measurement_ids'] = unknown
    if unknown:
        return finish('comparison member measurements are unavailable or unreliable')
    if metric == 'height_m':
        ambiguity = _height_structure_ambiguity(rows)
        if ambiguity:
            result['measurement_ambiguity'] = ambiguity
            return finish(ambiguity['reason'])
    from v4_policy import v5_enabled
    if v5_enabled('measurements') and metric == 'tabletop_height_m':
        ambiguity = _tabletop_floor_ambiguity(rows)
        if ambiguity:
            result['measurement_ambiguity'] = ambiguity
            return finish(ambiguity['reason'])
    kind = 'adjective' if spec['form'] == 'adjective' else 'comparative'
    if spec['form'] == 'superlative':
        kind = 'tie'
    if spec['interpretation'] == 'reference_comparison':
        kind = 'comparative'
    result['margin_kind'] = kind
    result['relative_margin'] = {'tie': TIE_RELATIVE_TOLERANCE,
                                 'comparative': COMPARATIVE_RELATIVE_MARGIN,
                                 'adjective': ADJECTIVE_RELATIVE_MARGIN}[kind]
    if spec['interpretation'] == 'reference_comparison':
        refs = list(dict.fromkeys(spec.get('reference_ids', [])))
        result['reference_ids'] = refs
        if (len(refs) != spec.get('required_reference_count', 2) or
                any(oid not in pool_ids for oid in refs)):
            return finish('local reference members are not completely and distinctly bound')
        statuses = {}
        tied = set()
        for oid in candidate_ids:
            if oid not in pool_ids or oid in refs:
                continue
            deltas = [(by_id[reference][metric] - by_id[oid][metric]) *
                      (1 if spec['direction'] == 'min' else -1) for reference in refs]
            margins = [_margin(metric, 'comparative', by_id[oid][metric], by_id[reference][metric])
                       for reference in refs]
            for reference, delta in zip(refs, deltas):
                if abs(delta) <= _margin(metric, 'tie', by_id[oid][metric], by_id[reference][metric]):
                    tied.update((oid, reference))
            statuses[oid] = ('pass' if all(delta > margin for delta, margin in zip(deltas, margins)) else
                             'fail' if any(delta < -margin for delta, margin in zip(deltas, margins)) else 'unknown')
        result['tie_ids'] = sorted(tied)
        return finish('local relative heights are tied or insufficiently separated'
                      if not statuses or 'unknown' in statuses.values() else None, statuses)
    if metric in {'volume_m3', 'oriented_volume_m3'}:
        footprint = 'oriented_footprint_m2' if metric.startswith('oriented_') else 'footprint_m2'
        ambiguity = _proxy_ambiguity(rows, spec['direction'], metric, footprint)
        if ambiguity:
            result['metric_ambiguity'] = ambiguity
            from v4_policy import enabled
            if not enabled('size'):
                return finish(ambiguity['reason'])
            spec = {**spec, 'tentative_metric': True}
            result['tentative_metric'] = True
        from v4_policy import enabled
        if not enabled('size') and metric.startswith('oriented_') and not all(_usable(row, footprint) for row in rows):
            return finish('overall size footprint proxy is unavailable or unreliable')
    if (v5_enabled('measurements') and metric == 'body_plane_area_m2'
            and retrieval_category(spec.get('category', '')) == 'book'):
        result['alternative_metrics'] = {'oriented_volume_m3': {
            'unknown_ids': [row['id'] for row in rows if not _usable(row, 'oriented_volume_m3')],
            'reason': 'book overall size can depend on both body-plane area and observed volume'}}
        if all(_usable(row, 'oriented_volume_m3') for row in rows):
            ambiguity = _proxy_ambiguity(rows, spec['direction'], metric, 'oriented_volume_m3')
            if ambiguity:
                result['metric_ambiguity'] = {**ambiguity,
                    'reason': 'book body-plane area and observed volume orderings disagree'}
                spec = {**spec, 'tentative_metric': True}
                result['tentative_metric'] = True
    ordered = sorted(rows, key=lambda row: row[metric], reverse=spec['direction'] == 'max')
    best, second = ordered[0][metric], ordered[1][metric]
    gap, margin = abs(best - second), _margin(metric, kind, best, second)
    result.update(margin_to_next=gap, required_margin=margin,
                  tie_ids=sorted(row['id'] for row in rows
                                 if abs(row[metric] - best) <= _margin(metric, 'tie', row[metric], best)))
    result['interpretation'] = ('contextual_set_comparison' if spec['form'] == 'adjective' else
                                'pair_comparison' if spec['form'] == 'comparative' and len(pool_ids) == 2 else
                                'set_comparison')
    if gap <= margin:
        if spec.get('relax_once') and metric not in ('height_m', 'tabletop_height_m'):
            # The absolute floor is retained; ordinal errors and ambiguous semantics never qualify.
            lower = max(.0005 if metric.endswith('_m3') else .005 if metric.endswith('_m2') else .03,
                        .05 * max(abs(best), abs(second)))
            if gap > lower and not result.get('metric_ambiguity'):
                result.update(dynamic_threshold={'old_margin': margin, 'new_margin': lower,
                    'trigger': 'planner stop after unchanged, valid bound measurements'}, tentative_metric=True)
                spec = {**spec, 'tentative_metric': True}
                winner = ordered[0]['id']
                return finish(None, {oid: 'pass' if oid == winner else 'fail' for oid in pool_ids})
        return finish('size extreme is tied or not clearly separated at the declared tolerance')
    winner = ordered[0]['id']
    return finish(None, {oid: 'pass' if oid == winner else 'fail' for oid in pool_ids})
