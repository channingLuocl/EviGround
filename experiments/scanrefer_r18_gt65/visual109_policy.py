"""Fixed, answer-independent candidate coverage for one independent retry."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from null_visual_recall import null_reason

POLICY = {
    'name': 'all_instances_atlas_binding_v1',
    'dynamic_tool_actions': 3,
    'max_focus_ids': 12,
    'overview_images': 3,
    'atlas_tile_px': 300,
    'atlas_columns': 4,
    'atlas_instances_per_page': 12,
    'atlas_azimuth_deg': 315,
    'atlas_elevation_deg': 30,
    'atlas_point_radius_px': 2,
    'atlas_order': 'ascending instance ID; every classifier-covered instance',
    'atlas_category_labels': False,
    'atlas_satisfies_dynamic_inspection': False,
}

INSTRUCTIONS = '''Independent visual coverage and referent-binding policy:
First survey EVERY ID in the atlas and the three room views before narrowing candidates.
All classifier-covered instances are allowed, including nonmatching category hypotheses.
Parser categories and Top-5 matches are fallible priors, never hard exclusions.
Atlas tiles are independently scaled; tile order, tile size and page position have no spatial meaning.
The atlas supplies a coarse shape/color survey, not proof of a fine attribute or spatial relation.

Bind the entity the utterance asks to locate separately from its reference, support, contained object,
and visible parts. Keep the full noun phrase and its modifiers together. A conspicuous object named
inside a relational clause is not automatically the target. Neither first nor last noun is a rule.
For a described set/row or an object's part, compare the visible instance boundaries with the full
description; do not substitute a salient accessory or component merely because its category matches.
Reconsider the parsed target if another noun phrase better accounts for the whole utterance.

Before a nonnull finish, dynamically inspect the proposed target with its strongest plausible rival;
include relevant references when they distinguish the pair. Choose isolated for shape/color/category
and scene for attachment, support or part/whole ambiguity. Use measure when the deciding clause
requires distances, support, height or order. The initial atlas does not satisfy this focused check.
In reason/evidence explicitly state the target noun phrase, bound reference IDs, best rival ID and
the clause/evidence that distinguishes them. If no rival is plausible, state why. Missing attributes
remain unknown. Keep abstention available when the essential binding cannot be supported.
Every inspect or measure has at most 12 distinct focus_ids; split wider comparisons across the
remaining three actions. The output must be one action object matching the supplied schema.
'''


def recover_candidates(candidates, objects):
    """Only the supplied classifier-covered objects can enter the target pool."""
    covered = [o for o in objects if 'category_hypotheses' in o]
    recovered = sorted(covered, key=lambda o: o['bbox_id'])
    original = [o['bbox_id'] for o in candidates]
    effective = [o['bbox_id'] for o in recovered]
    if len(effective) != len(set(effective)):
        raise ValueError('Classifier instance IDs must be unique')
    return recovered, {
        'policy': POLICY['name'],
        'fallback_applied': set(effective) != set(original),
        'original_candidate_ids': original,
        'effective_candidate_ids': effective,
        'source': 'all_classifier_covered_scene_instances',
        'ground_truth_used_for_recall': False,
        'reason': 'Category matches are priors; all classifier-covered instances remain allowed',
    }


def parse_action(content):
    """Do not salvage a nested object from a malformed top-level array."""
    action = json.loads(content)
    if not isinstance(action, dict):
        raise ValueError('ReAct response must be a single top-level JSON object')
    return action


def candidate_atlas(objects, xyz, rgb, ids, out):
    """Deterministic survey of every allowed object; no query or evaluation inputs."""
    from run import render

    covered = sorted((o for o in objects if 'category_hypotheses' in o), key=lambda o: o['bbox_id'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tile_size = POLICY['atlas_tile_px']
    page_size = POLICY['atlas_instances_per_page']
    azimuth, elevation = np.deg2rad([POLICY['atlas_azimuth_deg'], POLICY['atlas_elevation_deg']])
    direction = np.array([np.cos(elevation)*np.cos(azimuth),
                          np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    views = []
    font = ImageFont.load_default(size=16)
    for offset in range(0, len(covered), page_size):
        group = covered[offset:offset+page_size]
        columns = min(POLICY['atlas_columns'], len(group))
        rows = (len(group)+columns-1)//columns
        canvas = Image.new('RGB', (columns*tile_size, rows*tile_size), (245, 245, 245))
        panels = []
        for index, obj in enumerate(group):
            oid = obj['bbox_id']
            mask = ids == oid
            tile = Image.new('RGB', (tile_size, tile_size), (245, 245, 245))
            panel = {'id': oid, 'marked_ids': [], 'views_rendered': 0}
            if mask.any():
                box = np.asarray(obj['bbox_3d'], dtype=float)
                at = box[:3]
                eye = at + max(float(np.linalg.norm(box[3:])), .5)*1.5*direction
                path = out / f'visual109_atlas_instance_{oid}.png'
                stats = render(xyz[mask], rgb[mask], path, eye=eye, at=at,
                               size=tile_size, point_radius=POLICY['atlas_point_radius_px'])
                with Image.open(path) as rendered:
                    tile.paste(rendered)
                panel.update(image=str(path), marked_ids=[oid], views_rendered=1,
                             visible_pixels=stats['visible_pixels'])
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, tile_size, 45), fill='white')
            draw.text((6, 3), f'ID {oid} | independent scale', font=font, fill='black')
            draw.text((6, 24), 'shape/color survey' if mask.any() else 'NO INSTANCE POINTS',
                      font=font, fill='black')
            canvas.paste(tile, ((index % columns)*tile_size, (index // columns)*tile_size))
            panels.append(panel)
        path = out / f'visual109_atlas_page_{offset//page_size+1:02d}.png'
        canvas.save(path)
        views.append({'image': str(path), 'kind': 'all_instance_survey_atlas',
            'context': 'isolated', 'focus_ids': [o['bbox_id'] for o in group],
            'marked_ids': [i for panel in panels for i in panel['marked_ids']],
            'azimuth_deg': POLICY['atlas_azimuth_deg'], 'elevation_deg': POLICY['atlas_elevation_deg'],
            'layout': 'ascending IDs; independently scaled tiles; no shared spatial frame',
            'panels': panels, 'views_rendered': sum(p['views_rendered'] for p in panels),
            'provenance': 'Observed RGB points only; no class label or evaluation answer'})
    return views
