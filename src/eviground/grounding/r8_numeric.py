"""Execute a source-bound comparison after the model binds its pool and observer."""
import re

from eviground.grounding.evidence import retrieval_category
from eviground.api.response_schemas import obj, array, nullable, enum, REASON, TEXT

INSTRUCTIONS = '''
r4 numeric_choice is optional. Use null unless one target height or lateral ordering can resolve
the choice AFTER you bind all the other conditions. Supply only IDs that satisfy those remaining
conditions in eligible_ids, with remaining_conditions_checked=true. This pool is your visual/language
hypothesis, not an annotation. Never include unrelated objects just to make a numeric extremum win.
For left/right, copy the exact frame_id from observer_frames whose origin and facing fit the description.
Frame IDs contain the reference instance ID; never count array positions or substitute another reference.
For tallest/shortest, compare the TARGET's height, not a reference's height or a horizontal extent.
source_quote must be the exact original phrase expressing this comparison. Use leftmost/rightmost
only for an explicit extreme or a two-object left/right choice, not arbitrary left-of relations.
The executor sorts measured values in the bound pool. It will preserve the original decision if the
binding is missing, unsupported or tied. Do not reverse a computed ordering in the explanation.
'''


def choice_schema(allowed_ids, frames):
    return nullable(obj({'predicate': enum('leftmost', 'rightmost', 'tallest', 'shortest'),
        'source_quote': TEXT, 'eligible_ids': array({'type': 'integer', 'enum': allowed_ids}),
        'frame_id': nullable({'type': 'string', 'enum': [f['frame_id'] for f in frames]}) if frames
                       else {'type': 'null'},
        'remaining_conditions_checked': {'type': 'boolean'}, 'binding_reason': REASON}))


def execute_choice(decision, query, parsed, payload, allowed_ids):
    claim = decision.get('numeric_choice')
    if not isinstance(claim, dict) or claim.get('remaining_conditions_checked') is not True:
        return decision, {'applied': False, 'reason': 'No checked comparison binding'}
    quote = claim.get('source_quote', '')
    pool = claim.get('eligible_ids', [])
    if (not quote.strip() or quote.casefold() not in query.casefold() or len(pool) < 2
            or len(set(pool)) != len(pool) or any(type(i) is not int or i not in allowed_ids for i in pool)):
        return decision, {'applied': False, 'reason': 'Invalid source quote or candidate pool'}
    predicate = claim.get('predicate')
    if predicate in ('leftmost', 'rightmost'):
        side = 'left' if predicate == 'leftmost' else 'right'
        extreme = bool(re.search(r'\b(?:'+side+r'most|(?:far|furthest|farthest)\s+'+side+r')\b', quote, re.I))
        if not re.search(r'\b'+side+r'\b', quote, re.I) and not extreme:
            return decision, {'applied': False, 'reason': 'Direction not supported by quote'}
        frame = next((f for f in payload.get('observer_frames', [])
                      if f.get('frame_id') == claim.get('frame_id')), None)
        if (not extreme and len(pool) != 2) or frame is None:
            return decision, {'applied': False, 'reason': 'Requires explicit extreme/pair and a bound observer frame'}
        values = {r[0]: r[1] for r in frame['rows_left_to_right'] if r[0] in pool}
    elif predicate in ('tallest', 'shortest'):
        word = r'\b(?:tallest|taller)\b' if predicate == 'tallest' else r'\b(?:shortest|shorter)\b'
        target_categories = {retrieval_category(c) for c in parsed.get('Target', [])}
        target_bound = any(c.get('condition_type') == 'size' and c.get('source_quote')
            and c['source_quote'].casefold() in quote.casefold()
            and retrieval_category((c.get('target_role') or {}).get('category', '')) in target_categories
            and not c.get('reference_role') for c in parsed.get('condition_records', []))
        if not re.search(word, quote, re.I) or not target_bound:
            return decision, {'applied': False, 'reason': 'Height comparison is not bound to the target noun'}
        values = {o['id']: o['height_m'] for o in payload['objects'] if o['id'] in pool}
    else:
        return decision, {'applied': False, 'reason': 'Unsupported numeric predicate'}
    if set(values) != set(pool):
        return decision, {'applied': False, 'reason': 'Missing measurements'}
    ordered = sorted(values, key=lambda i: (values[i], i), reverse=predicate in ('rightmost', 'tallest'))
    if abs(values[ordered[0]]-values[ordered[1]]) <= 1e-6:
        return decision, {'applied': False, 'reason': 'Tied extremum; ID order is not evidence'}
    winner = ordered[0]
    proof = {'applied': True, 'predicate': predicate, 'quote': quote,
             'bound_pool': pool, 'frame_id': claim.get('frame_id'),
             'ordered_values': [[i, values[i]] for i in ordered],
             'before_id': decision['predicted_id'], 'after_id': winner,
             'binding_status': 'model_hypothesis_not_independent_verification'}
    result = {**decision, 'predicted_id': winner,
        'reason': f'Executed {predicate} over the model-bound pool {pool}: ID {winner} has value {values[winner]}. '
                  +claim['binding_reason'],
        'evidence': [{'source_quote': query, 'object_ids': ordered,
                      'observation': f'{predicate}: '+str(proof['ordered_values'])}]+decision.get('evidence', []),
        'unresolved': decision.get('unresolved', [])+[
            'Comparison pool and observer frame are model-bound hypotheses, not certified facts.']}
    return result, proof
