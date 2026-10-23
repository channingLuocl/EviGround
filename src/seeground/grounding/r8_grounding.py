"""Query-conditioned r8 evidence without evaluation labels or learned cutoffs."""
from pathlib import Path
import re

import numpy as np

from seeground.categories.categories_v9 import category_fields
from seeground.grounding.evidence import distance_pair_metrics, match_retrieval_categories, retrieval_category

REVISION = 'predicted-category-grounding-v9-r8-r2'
INSTRUCTIONS = '''
r8 evidence policy:
target_category_bindings distinguishes primary Top-1 matches from alternative-only matches.
These are priors, not a hard filter: compare both groups, and prefer the primary match only when
the language, appearance and spatial evidence are otherwise comparable. A low-ranked hypothesis
needs visible category support before a convenient spatial position can justify it. Keep the same
object's target/reference category interpretations compatible. Unrelated classifier alternatives
are omitted from this query's compact table; the frozen source retains the complete Top-5.
Read isolated_target_views to check shape/color of each candidate. Their tiles have independent
cameras and scales: tile order, apparent size and image left/right are NOT scene relations.
Use numeric_reference_distances and numeric_target_orders for arithmetic and ordering rather than
estimating from pictures or reciting coordinates in the wrong order. These measurements are not
automatic predicates: bind the right reference and the clause's comparison pool first. Preserve
other required conditions and negation; 'closest' is not a license to ignore them.
Scene cameras are inspection cameras. Infer a speaker frame from the described observer location
and facing direction before applying left/right. State the frame assumption; never equate world X
or picture left with speaker left. AABB dimensions alone do not reveal a bed's head or foot.
Anonymous supplemental instances remain geometry-only; their labels are not GT categories.
'''

REVIEW_INSTRUCTIONS = '''
Review the proposed scene decision against the ORIGINAL description and supplied measurements/images.
The proposal may be wrong. Re-evaluate all allowed targets and return a complete corrected decision
using the same schema. Keep a supported proposal; replace a contradicted one. Do not treat its
explanation as evidence. Check noun attachment, target/reference identity, visual category support,
comparison-group membership and every mandatory clause before choosing.
Check numeric comparisons explicitly against numeric_target_orders: a smaller height cannot become
'taller' by substituting width, and an ascending coordinate list cannot be reversed without a
justified observer frame. Do not confuse 'over' with 'nearest' or support with higher center Z alone.
Speaker right is perpendicular to speaker forward, never the same axis. A bed's long axis is not
automatically world Y. Infer head/foot from visible structure, and preserve uncertainty if unavailable.
Classifier-covered scene objects are not a complete vocabulary of physical things: a reference
absent from category labels may still be visible. Seek it in the original scene images; never invent
its ID or location. Compare alternatives rather than returning null solely because a label is absent.
Source_quote must be the COMPLETE original description verbatim. Choose only an allowed ID or null.
'''


def target_head_hint(query, categories):
    """Recognize only a leading category noun, not subjects in later clauses."""
    match = re.match(r'^\s*(?:please\s+)?(?:(?:select|choose|pick|find|locate)\s+)?'
                     r'(?:the|a|an|this|that)\s+(.+?)\s+'
                     r'(?:in|on|with|under|above|below|beneath|at|that|which)\b', query, re.I)
    if not match or re.search(r',\s*(?:the|a|an|this|that)\s+', query[match.end():], re.I):
        return None
    name = retrieval_category(match.group(1))
    if not name or name not in {retrieval_category(c) for c in categories}:
        return None
    return {'category': name, 'source_quote': match.group(1)}


def matching_hypotheses(names, item):
    return [h for h in item.get('category_hypotheses', [])
            if match_retrieval_categories(names, [{**item, 'category_hypotheses': [h]}])]


def generic_target_pool(names):
    from seeground.grounding.v4_policy import r8_generic_target_enabled
    return (r8_generic_target_enabled() and bool(names)
            and all(retrieval_category(name) in ('object', 'thing', 'item') for name in names))


def query_evidence(parsed, objects, candidates):
    targets = parsed.get('Target', [])
    anchors = parsed.get('Anchor', [])
    generic = generic_target_pool(targets)
    bindings = []
    for item in candidates:
        matches = matching_hypotheses(targets, item)
        bindings.append({'id': item['bbox_id'], 'matched_hypotheses': matches,
                         'primary_match': any(h['rank'] == 1 for h in matches)})
    references = match_retrieval_categories(anchors, objects)
    distances = []
    for reference in references:
        rows = []
        for item in candidates:
            if item['bbox_id'] == reference['bbox_id']:
                continue
            metrics = distance_pair_metrics(reference, item)
            rows.append([item['bbox_id'], *[round(metrics[key], 4) for key in
                        ('horizontal_center', 'horizontal_gap', 'center', 'aabb_gap')]])
        distances.append({'reference_id': reference['bbox_id'],
            'columns': ['target_id', 'xy_center_m', 'xy_box_gap_m', 'center_3d_m', 'box_gap_3d_m'],
            'rows': sorted(rows, key=lambda row: (row[3], row[0]))})
    orders = {}
    for key, index in (('center_x', 0), ('center_y', 1), ('center_z', 2),
                       ('extent_x', 3), ('extent_y', 4), ('height', 5)):
        orders[key+'_ascending'] = [[o['bbox_id'], round(float(o['bbox_3d'][index]), 4)]
            for o in sorted(candidates, key=lambda o: (o['bbox_3d'][index], o['bbox_id']))]
    table = []
    for item in objects:
        fields = category_fields(item)
        if fields and not generic:
            fields['category_hypotheses'] = [h for h in fields['category_hypotheses']
                if h['rank'] == 1 or matching_hypotheses(targets+anchors,
                    {**item, 'category_hypotheses': [h]})]
        table.append({'id': item['bbox_id'], 'category': item['target'], **fields,
            'center_xyz_m': [round(float(v), 4) for v in item['bbox_3d'][:3]],
            'extent_xyz_m': [round(float(v), 4) for v in item['bbox_3d'][3:]]})
    result = {'objects': table, 'target_category_bindings': bindings,
              'numeric_reference_distances': distances, 'numeric_target_orders': orders}
    if generic:
        result['generic_target_visual_pool'] = (
            'The parsed target is a nonspecific object/thing/item, not a classifier category. '
            'All classifier-covered instances are eligible. Resolve the referent from the original '
            'description, appearance and relations. A literal object label or matching rank is not '
            'evidence for this generic target. Original ranked class hypotheses remain fallible priors.')
    return result


def camera_eye(lower, upper, direction):
    """Fit every bounding-box corner within the renderer's 65-degree FOV."""
    at = (lower+upper)/2
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, -direction)
    corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])-at
    half_fov = np.tan(np.deg2rad(65/2))
    radial = np.maximum(np.abs(corners@right), np.abs(corners@up))
    distance = max(float(np.max(corners@direction + 1.12*radial/half_fov)), .3)
    return at+distance*direction, at


def isolated_target_views(candidates, xyz, rgb, ids, out):
    # Reuse the existing depth-tested isolated renderer and its explicit provenance.
    from seeground.grounding.active_diagnostic import render_action
    result = []
    for offset in range(0, len(candidates), 6):
        group = candidates[offset:offset+6]
        view = render_action({'focus_ids': [o['bbox_id'] for o in group],
            'azimuth_deg': 315, 'elevation_deg': 30, 'context': 'isolated'},
            candidates, xyz, rgb, ids, Path(out), 1000+offset//6)
        result.append({**view, 'kind': 'isolated_target_views'})
    return result
