"""Recognize opposing visual findings that need a joint view of the candidates."""
from collections import defaultdict
import re


_PREDICATES = {
    'corner': r'\b(?:in|at) (?:a |the )?corner\b',
    'right side': r'\b(?:to|on|at) (?:its|their|the) (?:far )?right\b',
    'left side': r'\b(?:to|on|at) (?:its|their|the) (?:far )?left\b',
    'top right': r'\btop[- ]right\b',
    'top left': r'\btop[- ]left\b',
    'in front': r'\b(?:out )?in front\b',
    'behind': r'\bbehind\b',
    'on top': r'\b(?:on top of|stacked (?:on|above))\b',
}
_OPPOSITE = {'top right': 'top left', 'top left': 'top right'}
_NEGATION = re.compile(r"\b(?:not|never|no)\b|\b\w+n['’]t\b", re.I)
_UNCERTAINTY = re.compile(
    r"\b(?:cannot|unclear|unknown|occluded|possibly|might|insufficient|appear\w*|seem\w*)\b"
    r"|\b(?:not|barely) (?:visible|clear|shown|identifiable)\b"
    r"|\bnot (?:enough|reveal|show|confirm)\b"
    r"|\b(?:can't|could not|unable to|hard to) (?:see|tell|determine|confirm|identify)\b", re.I
)
_MENTION = re.compile(r'\((?:object |id )?(\d+)\)|\b(?:object|id)\s*#?(\d+)\b', re.I)


def _finding_signals(finding, predicates, marked_ids, candidates):
    for clause in re.split(r'[.;]|\b(?:but|while|whereas)\b', finding['observation'], flags=re.I):
        if _UNCERTAINTY.search(clause):
            continue
        mentioned = {int(groups[0] or groups[1]) for groups in _MENTION.findall(clause)}
        refs = (mentioned & marked_ids) - candidates
        for criterion, pattern in predicates.items():
            match = re.search(pattern, clause, re.I)
            if match:
                yield criterion, not _NEGATION.search(clause[:match.start()]), refs
            elif criterion in _OPPOSITE:
                opposite = re.search(_PREDICATES[_OPPOSITE[criterion]], clause, re.I)
                if opposite and not _NEGATION.search(clause[:opposite.start()]):
                    yield criterion, False, refs


def unresolved_visual_conflict(reports, views, candidate_ids, proposed_id, *, description=''):
    """Return candidates and references to reinspect together, without selecting a winner."""
    candidates = set(candidate_ids)
    if proposed_id not in candidates:
        return None
    predicates = {key: pattern for key, pattern in _PREDICATES.items()
                  if not description or re.search(pattern, description, re.I)}
    if not predicates:
        return None
    signals = defaultdict(lambda: defaultdict(list))
    grouped_conflicts = set()
    for index, report in enumerate(reports):
        for group in report['groups']:
            if group['assessment'] == 'conflicting':
                grouped_conflicts.update((a, b) for a in group['finding_ids'] for b in group['finding_ids'] if a != b)
        for finding in report['findings']:
            if finding['status'] != 'observed':
                continue
            marked = set(views.get(finding['image_id'], {}).get('view', {}).get('marked_ids', []))
            covered = candidates & marked & set(finding['object_ids'])
            for criterion, positive, refs in _finding_signals(finding, predicates, marked, candidates):
                for object_id in covered:
                    signals[criterion][object_id].append((index, finding, positive, refs))
    for criterion, by_candidate in signals.items():
        winner = by_candidate.get(proposed_id, [])
        for rival_id in sorted(candidates - {proposed_id}):
            rival = by_candidate.get(rival_id, [])
            # Require opposite preferences for the same predicate. Two positive
            # sightings, or one image that could not see a candidate, are not enough.
            for a in (row for row in winner if row[2]):
                for b in (row for row in rival if row[2]):
                    if (a[1]['image_id'] == b[1]['image_id']
                            or (a[3] and b[3] and not a[3] & b[3])):
                        continue
                    negatives = [row for row in (_opposing_negative(a, b, rival, grouped_conflicts),
                                                 _opposing_negative(b, a, winner, grouped_conflicts))
                                 if row is not None]
                    if len(negatives) != 2 and not any(
                            all((row[1]['finding_id'], event[1]['finding_id']) in grouped_conflicts
                                for event in (a, b)) for row in negatives):
                        continue
                    fact_rows = [a, b, *negatives]
                    refs = set().union(*(row[3] for row in fact_rows)) - {proposed_id, rival_id}
                    followup = _joint_followup(winner, rival, views, max(row[0] for row in fact_rows),
                                               {proposed_id, rival_id} | refs)
                    if followup and followup[0]:
                        continue
                    if followup:
                        fact_rows.extend(followup[1])
                    conflict = {
                        'candidate_ids': sorted([proposed_id, rival_id]),
                        'reference_ids': sorted(refs),
                        'finding_ids': list(dict.fromkeys(row[1]['finding_id'] for row in fact_rows)),
                        'evidence_ids': sorted({row[1]['evidence_id'] for row in fact_rows}),
                        'criterion': criterion,
                    }
                    if followup:
                        conflict['refuted'] = True
                    return conflict
    return None


def _opposing_negative(same_view_positive, contrary_positive, rows, grouped_conflicts):
    for row in rows:
        if (not row[2] and row[0] == same_view_positive[0]
                and (not row[3] or not contrary_positive[3] or row[3] & contrary_positive[3])
                and (row[1]['image_id'] != contrary_positive[1]['image_id']
                     or (row[1]['finding_id'], contrary_positive[1]['finding_id']) in grouped_conflicts)):
            return row
    return None


def _joint_followup(winner, rival, views, after, required_ids):
    """A later contextual image must provide distinct answers about both candidates."""
    signs = defaultdict(lambda: [dict(), dict()])
    for side, observations in enumerate((winner, rival)):
        for index, finding, positive, _ in observations:
            image_id = finding['image_id']
            view = views.get(image_id, {}).get('view', {})
            if (index >= after and view.get('context') == 'scene'
                    and required_ids <= set(view.get('marked_ids', []))):
                signs[index, image_id][side][positive] = (index, finding, positive, set())
    votes = defaultdict(dict)
    for (index, _), (winner_signs, rival_signs) in signs.items():
        for supports_winner in (True, False):
            if winner_signs.keys() == {supports_winner} and rival_signs.keys() == {not supports_winner}:
                votes[index][supports_winner] = [winner_signs[supports_winner], rival_signs[not supports_winner]]
    if votes:
        latest = votes[max(votes)]
        if len(latest) == 1:
            return next(iter(latest.items()))
    return None
