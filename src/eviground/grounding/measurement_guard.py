"""Check explicit comparisons before a grounding plan eliminates target candidates."""
import math
import re

from eviground.categories.retrieval import retrieval_category
from eviground.grounding.evidence_followup import EvidenceFollowup
from eviground.query.size_language import category_pattern


_SMALLER = {'smaller', 'smallest', 'shorter', 'shortest', 'narrower', 'narrowest'}
_LARGER = {'larger', 'largest', 'bigger', 'biggest', 'taller', 'tallest', 'wider', 'widest'}
_ORDINALS = {word: rank for rank, word in enumerate(
    ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'), 1)}
_POSITION = {'relation:left': (1, -1), 'relation:right': (1, 1),
             'relation:front': (2, 1), 'relation:back': (2, -1)}
_EXTREME = re.compile(r'\b(?:far\s+(left|right)|(left|right|front|back)most)\b', re.I)
_SIDE = re.compile(r'\b(?:(?:top|bottom|upper|lower)[ -]+(left|right)|'
                   r'(?:on|at|to)\s+(?:the\s+)?(left|right))\b', re.I)
_OBSERVER_OPENING = re.compile(r"""^\s*['"]?(?:facing |looking (?:at |towards? |from )|while (?:facing|looking) )""", re.I)
_LOOK_AT = r'\b(?:fac(?:e|ing)|look(?:ing)?\s+(?:at|towards?))\s+'
_ROOM_CENTER = re.compile(r'\b(?:stand(?:ing)?|sit(?:ting)?)\s+in\s+(?:the\s+)?'
                          r'(?:middle|cent(?:er|re))\s+of\s+(?:the\s+)?room\b', re.I)


def observer_frame_constraints(description, references, observer=None):
    """Return the roles supported by an exact observer phrase or unambiguous source wording."""
    observer = observer or {}
    origin, facing = (observer.get(role) or {} for role in ('origin', 'facing'))
    origin_phrase = origin.get('phrase') or ''
    facing_phrase = facing.get('phrase') or ''
    explicit_center = bool(_ROOM_CENTER.search(description))
    origin_ids = None
    if origin_phrase and origin_phrase.casefold() in description.casefold():
        if origin.get('kind') == 'room_center':
            explicit_center = True
        elif origin.get('kind') == 'category':
            origin_ids = references.get(retrieval_category(origin.get('category', '')), set())

    # BigModel sometimes returns only "the window" as the facing quote, which
    # parser validation drops. Recover the literal verb+noun phrase when unique.
    matches = []
    opener = bool(_OBSERVER_OPENING.search(description))
    for name in references:
        if not name:
            continue
        pattern = _LOOK_AT + r'(?:(the|a|an|all|both)\s+)?(' + category_pattern(name) + r')\b'
        for match in re.finditer(pattern, description, re.I):
            before = description[:match.start()]
            if ((opener and before.strip(" \t\n'\"").casefold() in {'', 'while'})
                    or re.search(r'\b(?:you|stand(?:ing)?|sit(?:ting)?)\b', before, re.I)):
                matches.append((match.start(), match.end(), name, match[1], match[2]))
    longest = {}
    for start, end, name, determiner, noun in sorted(matches):
        longest[start] = (name, determiner, noun)
    facing_ids = None
    facing_group = False
    intrinsic = facing.get('reference_frame') == 'intrinsic'
    if len(longest) == 1 and not intrinsic:
        category, determiner, noun = next(iter(longest.values()))
        facing_ids = references[category]
        facing_group = (determiner or '').casefold() in {'all', 'both'} or noun.casefold() != category
    elif facing_phrase and facing_phrase.casefold() in description.casefold():
        facing_ids = references.get(retrieval_category(facing.get('category', '')), set())
        facing_group = bool(facing.get('group'))
    return explicit_center, origin_ids, facing_ids, facing_group, intrinsic


def matching_observer_frame(frame, constraints):
    center, origin_ids, facing_ids, facing_group, intrinsic = constraints
    if intrinsic:
        return False
    origin, facing = set(frame.get('origin_ids', [])), set(frame.get('facing_ids', []))
    if center and (frame.get('origin_ids') != []):
        return False
    if origin_ids is not None and (not origin or not origin <= origin_ids):
        return False
    if facing_ids is not None and (not facing or not facing <= facing_ids
                                  or (len(facing) < 2 if facing_group else len(facing) != 1)):
        return False
    return not (origin & facing)


def requested_frame(constraints, bound, measurements):
    """Reuse one matching recorded frame or construct one from unambiguous roles."""
    center, origin_ids, facing_ids, facing_group, intrinsic = constraints
    if intrinsic:
        return None
    recorded = set()
    for measurement in measurements:
        frame = measurement.get('selected_observer_frame') or {}
        if frame.get('facing_ids') and matching_observer_frame(frame, constraints):
            recorded.add((tuple(sorted(frame.get('origin_ids', []))), tuple(sorted(frame['facing_ids']))))
    if len(recorded) == 1:
        origin, facing = next(iter(recorded))
        return {'origin_ids': list(origin), 'facing_ids': list(facing)}

    origin = [] if center else sorted((bound & origin_ids) or origin_ids) if origin_ids else None
    facing = sorted((bound & facing_ids) or facing_ids) if facing_ids else []
    if (origin is None or (not center and len(origin) != 1) or not facing
            or (facing_group and len(facing) < 2)
            or (not facing_group and len(facing) != 1)
            or (facing_group and not set(facing) <= bound)):
        return None
    frame = {'origin_ids': origin, 'facing_ids': facing}
    return frame if matching_observer_frame(frame, constraints) else None


def size_rows(measurement, metric):
    for row in measurement.get('objects', []):
        dx, dy, dz = row['extent_xyz_m']
        value = {'height_m': dz, 'horizontal_short_extent_m': min(dx, dy),
                 'vertical_face_area_m2': max(dx, dy) * dz,
                 'volume_m3': row['observed_box_volume_m3']}[metric]
        yield row['id'], value


def comparison_ordinal(quote, kind):
    operator = {'relation:nearest': 'nearest|closest',
                'relation:farthest': 'farthest|furthest'}.get(kind, re.escape(kind))
    ordinal_words = '|'.join(_ORDINALS)
    match = re.search(rf'\b(?:(?P<word>{ordinal_words})|(?P<number>[1-9]\d*)(?:st|nd|rd|th))'
                      rf'[\s-]+(?:{operator})\b', quote, re.I)
    if match:
        return int(match['number']) if match['number'] else _ORDINALS[match['word'].lower()]
    return None


def measured_rank_range(values, group, winner, direction):
    value = values[winner]
    ties = {rival for rival in group - {winner}
            if math.isclose(values[rival], value, rel_tol=0.05, abs_tol=0.0001)}
    rank = 1 + sum(direction * (values[rival] - value) > 0
                   for rival in group - {winner} - ties)
    return rank, rank + len(ties)


class ComparisonContradiction(ValueError):
    """Cited measurements place a proposed target at a different rank."""
    def __init__(self, message, *, condition_id, focus_ids, anchor_id, ordinal, supported_id):
        super().__init__(message)
        self.condition_id = condition_id
        self.focus_ids = focus_ids
        self.anchor_id = anchor_id
        self.ordinal = ordinal
        self.supported_id = supported_id


def check_measured_exclusions(requirements, allowed, plan, observations, reference_bindings, target_categories,
                              *, observer=None):
    """Reject unsupported comparisons without choosing or removing targets."""
    nodes = {node['id']: node for node in plan['nodes']}
    measurements = {row['evidence_id']: row for row in observations
                    if row.get('tool') == 'measure' and row.get('evidence_id')
                    and not row.get('no_evidence_acquired')}
    target_categories = {retrieval_category(name) for name in target_categories}
    references = {retrieval_category(binding['category']): {row['id'] for row in binding['candidates']}
                  for binding in reference_bindings}
    description = next((row.get('source_quote') or '' for row in requirements if row['id'] == 'q0'), '')
    frame_constraints = observer_frame_constraints(description, references, observer)
    explicit_viewer = (_OBSERVER_OPENING.search(description)
                       or frame_constraints[0] or frame_constraints[1] is not None
                       or frame_constraints[2] is not None
                       or any(row['condition'].startswith('viewpoint:') for row in requirements))

    for node in nodes.values():
        excluded = set(node['excluded_target_ids'])
        if node['status'] != 'completed' or not excluded:
            continue
        members = set(node['bound_object_ids']) & allowed
        survivors = members - excluded
        if len(survivors) != 1:
            continue
        winner = next(iter(survivors))
        group = survivors | excluded
        cited = [measurements[key] for key in node['evidence_ids'] if key in measurements]
        bound = set(node['bound_object_ids'])
        pending = list(node['depends_on'])
        visited = set()
        while pending:
            key = pending.pop()
            if key not in visited:
                visited.add(key)
                bound.update(nodes[key]['bound_object_ids'])
                pending.extend(nodes[key]['depends_on'])

        for requirement in requirements:
            if requirement['id'] not in node['condition_ids']:
                continue
            quote = requirement.get('source_quote') or ''
            if not quote or re.search(r"\b(?:not|no|never|without)\b|n['’]t\b", quote, re.I):
                continue
            role = requirement.get('target_role') or {}
            category = retrieval_category(role.get('category', ''))
            if category and category not in target_categories:
                continue
            kind, metric = requirement['condition'], requirement.get('metric')
            extreme = _EXTREME.search(quote)
            side = _SIDE.search(quote)
            if kind in {'relation', 'relation:other', 'appearance', 'unhandled'}:
                literal = extreme or side
                if literal and (category or not requirement.get('reference_role')):
                    kind = 'relation:' + next(term.casefold() for term in literal.groups() if term)
            anchor_role = requirement.get('reference_role') or {}
            anchor_names = anchor_role.get('categories') or [anchor_role.get('category', '')]
            anchors = set().union(*(references.get(retrieval_category(name), set()) for name in anchor_names)) - group
            bound_anchors = bound & anchors
            anchor = next(iter(bound_anchors)) if len(bound_anchors) == 1 else None
            checks = []
            context = ''
            relative = False
            compare_rank = True
            ordinal = None
            followup_ids, followup_frame, followup_tool = None, None, 'measure'
            if (kind in _SMALLER | _LARGER and metric in
                    {'height_m', 'horizontal_short_extent_m', 'vertical_face_area_m2', 'volume_m3'}
                    and requirement.get('scope_kind') in {'explicit_pair', 'full_category'}):
                count = requirement.get('member_count')
                if count is not None and len(group) != count:
                    raise ValueError(f"Comparison {requirement['id']} ({quote!r}) requires {count} members; "
                                     f"the proposal compares IDs {sorted(group)}. Bind that group before excluding.")
                direction = -1 if kind in _SMALLER else 1
                ordinal = comparison_ordinal(quote, kind)
                followup_ids = group
                for measurement in cited:
                    values = dict(size_rows(measurement, metric))
                    if group <= values.keys():
                        # Overall size is ambiguous when footprint and volume disagree.
                        orderings = [values]
                        if metric == 'volume_m3':
                            orderings.append({row['id']: row['observed_footprint_m2']
                                              for row in measurement['objects']})
                        checks.append(orderings)
                context = metric
            elif (kind in ('relation:nearest', 'relation:farthest') and category
                    and len(anchor_names) == 1 and anchors):
                direction = -1 if kind == 'relation:nearest' else 1
                ordinal = comparison_ordinal(quote, kind)
                if anchor is not None:
                    followup_ids = group | {anchor}
                metrics = ('xy_center_m', 'xy_box_gap_m')
                collected = {candidate: set() for candidate in group}
                for measurement in cited:
                    pairs = {row['object_id']: row for row in measurement.get('directed_pair_measurements', [])
                             if row['reference_id'] == anchor}
                    for candidate in group & pairs.keys():
                        collected[candidate].add(tuple(pairs[candidate][field] for field in metrics))
                    if group <= pairs.keys():
                        checks.append([{key: pairs[key][field] for key in group}
                                       for field in metrics])
                # Distances are fixed within this episode; merge split comparisons
                # only when every candidate uses the same reference and rows agree.
                if not checks and all(len(values) == 1 for values in collected.values()):
                    ordered = {candidate: next(iter(values)) for candidate, values in collected.items()}
                    checks.append([{candidate: values[index] for candidate, values in ordered.items()}
                                   for index in range(len(metrics))])
                context = f'one bound reference from IDs {sorted(anchors)} and common-reference distance'
            elif kind in _POSITION and (extreme or explicit_viewer):
                axis, direction = _POSITION[kind]
                compare_rank = bool(extreme)
                relative = not compare_rank and bool(anchors or (side and axis == 1))
                context = 'rows from a single observer frame grounded in the description'
                _, _, facing_ids, facing_group, _ = frame_constraints
                if facing_ids is not None:
                    context += (f' facing {"a bound group" if facing_group else "one reference"} '
                                f'from IDs {sorted(facing_ids)}')
                for measurement in cited:
                    frame = measurement.get('selected_observer_frame') or {}
                    if not matching_observer_frame(frame, frame_constraints):
                        continue
                    values = {row[0]: row[axis] for row in frame.get('rows_left_to_right', [])}
                    measured_anchors = anchors & set(frame.get('facing_ids', []))
                    measured_anchor = anchor
                    if measured_anchor is None and len(measured_anchors) == 1:
                        measured_anchor = next(iter(measured_anchors))
                    if group <= values.keys() and (not relative or not anchors or measured_anchor in values):
                        offset = values[measured_anchor] if relative and anchors else 0
                        checks.append([{key: value - offset for key, value in values.items()}
                                       if relative else values])
                if not checks:
                    followup_ids, followup_tool = group, None
                    frame = requested_frame(frame_constraints, bound, measurements.values())
                    if frame:
                        frame_anchors = anchors & set(frame['facing_ids'])
                        if not relative or not anchors or anchor is not None or len(frame_anchors) == 1:
                            followup_ids = group | set(frame['origin_ids']) | set(frame['facing_ids'])
                            if relative and anchor is not None:
                                followup_ids.add(anchor)
                            followup_frame, followup_tool = frame, 'measure'
            elif (kind in ('relation:above', 'relation:below') and category and anchors
                  and re.match(r'^(?:directly |just )?(?:above|below)\b', quote, re.I)):
                direction = 1 if kind == 'relation:above' else -1
                relative = True
                context = f'joint height measurement relative to a bound reference from IDs {sorted(anchors)}'
                if anchor is not None:
                    followup_ids = group | {anchor}
                for measurement in cited:
                    values = {row['id']: row['center_xyz_m'][2] for row in measurement['objects']}
                    if group <= values.keys() and anchor in values:
                        checks.append([{key: value - values[anchor] for key, value in values.items()}])
            else:
                continue

            if ordinal is not None and ordinal > 1:
                ancestor_exclusions = set().union(*(nodes[key]['excluded_target_ids'] for key in visited
                    if nodes[key]['status'] == 'completed' and requirement['id'] not in nodes[key]['condition_ids']))
                unbound = allowed - group - ancestor_exclusions
                if len(group) < ordinal or unbound:
                    raise EvidenceFollowup(
                        f"Comparison {requirement['id']} ({quote!r}) requires rank {ordinal} in the "
                        f"full bound group. Compared IDs {sorted(group)}; still unbound {sorted(unbound)}. "
                        "Establish the group with evidence before excluding.",
                        focus_ids=sorted(group | unbound | ({anchor} if anchor is not None else set())),
                        tool=None, condition_id=requirement['id'])
            if not checks:
                message = (f"Comparison {requirement['id']} ({quote!r}) lacks cited {context} for IDs "
                           f"{sorted(group)}. Keep rivals active or measure them together and cite the observation.")
                if followup_ids:
                    raise EvidenceFollowup(message, focus_ids=sorted(followup_ids), tool=followup_tool,
                                           frame=followup_frame, condition_id=requirement['id'])
                raise EvidenceFollowup(message, focus_ids=sorted(group), tool=None, condition_id=requirement['id'])
            if relative:
                if all(direction * orderings[0][winner] < -0.02 for orderings in checks):
                    raise ComparisonContradiction(
                        f"Comparison {requirement['id']} ({quote!r}) contradicts cited {context}: "
                        f"ID {winner} has signed offset {checks[0][0][winner]:.4g} m. "
                        "Keep the comparison open and check the reference and frame.",
                        condition_id=requirement['id'], focus_ids=sorted(group), anchor_id=anchor,
                        ordinal=None, supported_id=None)
                continue
            if ordinal is not None and ordinal > 1:
                metrics = [values for orderings in checks for values in orderings]
                ranks = [measured_rank_range(values, group, winner, direction) for values in metrics]
                if all(rank == (ordinal, ordinal) for rank in ranks):
                    continue
                if all(not first <= ordinal <= last for first, last in ranks):
                    orders = [sorted(((candidate, values[candidate]) for candidate in group),
                                     key=lambda row: (-direction * row[1], row[0])) for values in metrics]
                    proposed = orders[0][ordinal - 1][0]
                    supported = all(measured_rank_range(values, group, proposed, direction) == (ordinal, ordinal)
                                    for values in metrics)
                    raise ComparisonContradiction(
                        f"Comparison {requirement['id']} ({quote!r}) contradicts cited {context}: "
                        f"ID {winner} is not rank {ordinal} in measured ID/value orders {orders}. "
                        "Keep rivals active and revise the comparison.",
                        condition_id=requirement['id'], focus_ids=sorted(group), anchor_id=anchor,
                        ordinal=ordinal, supported_id=proposed if supported else None)
                raise ComparisonContradiction(
                    f"Comparison {requirement['id']} ({quote!r}) has ambiguous rank {ordinal} "
                    f"for ID {winner} in cited {context}. Keep rivals active and resolve the ambiguity.",
                    condition_id=requirement['id'], focus_ids=sorted(group), anchor_id=anchor,
                    ordinal=ordinal, supported_id=None)
            # A clear contradiction needs agreement across the relevant metrics and observations.
            for rival in sorted(excluded) if compare_rank else ():
                if all(direction * (values[rival] - values[winner]) > 0
                       and not math.isclose(values[rival], values[winner], rel_tol=0.05, abs_tol=0.0001)
                       for orderings in checks for values in orderings):
                    values = checks[0][0]
                    raise ComparisonContradiction(
                        f"Comparison {requirement['id']} ({quote!r}) contradicts cited {context}: "
                        f"ID {winner} has {values[winner]:.4g}, ID {rival} has {values[rival]:.4g}. "
                        "Retain the rival or revise the comparison group with evidence.",
                        condition_id=requirement['id'], focus_ids=sorted(group), anchor_id=anchor,
                        ordinal=ordinal, supported_id=None)
