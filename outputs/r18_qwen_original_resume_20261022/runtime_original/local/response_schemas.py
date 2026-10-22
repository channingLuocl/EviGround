"""Strict output contracts for v6.2 OpenAI-compatible production stages."""


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


def enum(*values):
    return {'type': 'string', 'enum': list(values)}


def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


TEXT = {'type': 'string'}
REASON = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
INTEGER = {'type': 'integer'}
NUMBER = {'type': 'number'}
VERDICT = enum('yes', 'no', 'uncertain')
VISIBILITY = enum('visible', 'occluded', 'low_detail', 'unknown')
STATUS = enum('pass', 'fail', 'unknown')
ROLE = enum('target', 'reference')
ATTRIBUTE = obj({'category': TEXT, 'role': ROLE, 'phrase': TEXT})
RELATION = obj({'target_category': TEXT, 'anchor_category': TEXT,
    'anchor_categories': array(TEXT),
    'relation': enum('nearest', 'farthest', 'near', 'left', 'right', 'front',
                     'back', 'above', 'below', 'underneath', 'between', 'other'),
    'scope': enum('', 'side', 'relative_anchor', 'relative_anchors'), 'phrase': TEXT})
OBSERVER = obj({'origin': obj({'kind': enum('room_center', 'category', 'unspecified'),
    'category': TEXT, 'phrase': TEXT}), 'facing': obj({'category': TEXT,
    'group': {'type': 'boolean'}, 'phrase': TEXT})})


def parser_schema(mode):
    fields = {'Target': array(TEXT), 'Anchor': array(TEXT)}
    if mode != 'baseline':
        fields.update(Relations=array(RELATION), Viewpoint=TEXT)
    if mode == 'diagnostic':
        fields.update(Observer=OBSERVER, Attributes=array(ATTRIBUTE), Unhandled=array(TEXT))
    return obj(fields)


ATTRIBUTES = obj({'Attributes': array(ATTRIBUTE)})


def planner_schema(known_ids, target_ids):
    focus = {'type': 'array', 'items': {'type': 'integer', 'enum': sorted(set(known_ids))},
             'minItems': 1, 'maxItems': 6}
    schema = obj({'action': enum('evidence', 'finish', 'stop'), 'reason': REASON,
        'dimension': nullable(enum('viewpoint', 'position', 'distance', 'appearance', 'size')),
        'predicted_id': {'type': ['integer', 'null'], 'enum': sorted(set(target_ids)) + [None]},
        'inspect': nullable({'type': 'boolean'}), 'focus_ids': nullable(focus),
        'azimuth_deg': nullable({'type': 'number', 'minimum': 0, 'exclusiveMaximum': 360}),
        'elevation_deg': nullable({'type': 'number', 'minimum': 10, 'maximum': 75}),
        'context': nullable(enum('scene', 'isolated'))})
    from v4_policy import v73_enabled
    if v73_enabled():
        schema['properties']['inspect']['description'] = 'Only for size/position/distance evidence. Null for appearance (always renders), viewpoint, finish and stop.'
        schema['properties']['context']['description'] = 'Only for appearance evidence. Null for other dimensions and finish/stop; geometry inspection uses a fixed comparison panel.'
        schema['properties']['predicted_id']['description'] = 'Only for finish. Null for evidence and stop.'
    return schema


ATTRIBUTE_CHECK = obj({'attribute': TEXT, 'verdict': VERDICT,
                       'visibility': VISIBILITY, 'reason': REASON})
APPEARANCE_CHECK = obj({'id': INTEGER, 'category': TEXT, 'role': ROLE,
    'category_verdict': VERDICT, 'verdict': VERDICT, 'visibility': VISIBILITY,
    'reason': REASON, 'attribute_checks': array(ATTRIBUTE_CHECK)})
APPEARANCE = obj({'checks': array(APPEARANCE_CHECK)})
RGB_REVIEW = obj({'id': INTEGER, 'category_verdict': VERDICT,
    'visibility': VISIBILITY, 'verdict': VERDICT, 'reason': REASON,
    'attribute_checks': array(ATTRIBUTE_CHECK)})
CATEGORY_REVIEW = obj({'id': INTEGER, 'verdict': VERDICT, 'reason': REASON})
INSTANCE_REVIEW = obj({'TargetIDs': array(INTEGER), 'AnchorIDs': array(INTEGER),
    'UnknownIDs': array(INTEGER), 'Observations': array(obj({'id': INTEGER, 'appearance': TEXT}))})

VISUAL_REPORT = obj({
    'observations': array(obj({'id': INTEGER,
        'completeness': enum('supported', 'suspected_incomplete', 'unknown'),
        'visible_surfaces': array(TEXT), 'reason': REASON,
        'condition_checks': array(obj({'source_quote': TEXT, 'status': STATUS, 'reason': REASON}))})),
    'metric_choices': array(obj({'source_quote': TEXT, 'category': TEXT, 'word': TEXT,
        'metric': enum('oriented_volume_m3', 'oriented_footprint_m2', 'body_plane_area_m2'),
        'reason': REASON})),
    'relations': array(obj({'candidate_id': INTEGER, 'anchor_id': nullable(INTEGER),
        'source_quote': TEXT, 'status': STATUS, 'reason': REASON})),
    'contradictions': array(obj({'candidate_ids': array(INTEGER), 'source_quote': TEXT,
        'evidence_type': enum('visible_geometry', 'measurement'), 'reason': REASON})),
    'status': enum('consistent', 'contradiction', 'unknown'), 'reason': REASON,
    'findings': array(obj({
        'finding': enum('confirms_exclusion', 'refutes_exclusion', 'contradicts_selected', 'unknown'),
        'candidate_id': INTEGER, 'event_id': nullable(INTEGER), 'source_quote': TEXT,
        'target_role': enum('target'), 'predicate_status': STATUS,
        'evidence_type': enum('visible_geometry', 'measurement'), 'reason': REASON,
        'observation_ids': array(INTEGER), 'measurement_keys': array(TEXT),
        'condition_checks': array(obj({'source_quote': TEXT, 'status': STATUS, 'reason': REASON})),
        'comparison': nullable(obj({'key': TEXT, 'operator': enum('lt', 'gt'),
            'value': NUMBER, 'reference_id': INTEGER, 'reference_key': TEXT}))}))})


def schema_kwargs(name, schema):
    from qwen_api import api_provider
    from v4_policy import v62_enabled
    if v62_enabled() and api_provider() == 'openai':
        return {'response_schema': {'name': name, 'strict': True, 'schema': schema}}
    return {}
