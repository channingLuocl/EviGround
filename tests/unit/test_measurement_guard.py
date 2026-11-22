"""BigModel comparisons use actual measured groups rather than asserted ranks."""
from copy import deepcopy

import pytest

from eviground.grounding.evidence_followup import EvidenceFollowup
from eviground.grounding.measurement_guard import ComparisonContradiction, check_measured_exclusions


def check(requirement, measured, *, quote, result=8, rival=0, anchor=None, additional_rivals=(),
          unbound=(), second_measurement=None):
    node = {'id': 'n0', 'condition_ids': ['q1'], 'depends_on': [],
            'status': 'completed', 'bound_object_ids': [result] + ([anchor] if anchor else []),
            'excluded_target_ids': [rival, *additional_rivals], 'evidence_ids': ['initial', 'measurement']}
    bindings = [{'category': 'chair', 'candidates': [{'id': anchor}]}] if anchor else []
    observations = [{'tool': 'measure', 'evidence_id': 'measurement', **measured}]
    if second_measurement:
        observations.append({'tool': 'measure', 'evidence_id': 'second_measurement', **second_measurement})
        node['evidence_ids'].append('second_measurement')
    check_measured_exclusions([{'id': 'q1', 'source_quote': quote,
                                'target_role': {'category': 'couch'}, **requirement}],
                              {result, rival, *additional_rivals, *unbound}, {'nodes': [node]}, observations, bindings,
                              ['couch'])


def test_far_right_requires_observer_frame_and_does_not_use_world_coordinates():
    requirement = {'condition': 'relation:right'}
    measurement = {'objects': [{'id': 0, 'center_xyz_m': [1, 2, 0]},
                               {'id': 8, 'center_xyz_m': [-1, -2, 0]}]}
    with pytest.raises(ValueError, match='lacks cited.*observer frame'):
        check(requirement, measurement, quote='on the far right')
    measurement['selected_observer_frame'] = {'rows_left_to_right': [[0, -1, 2], [8, 0.5, -2]]}
    check(requirement, measurement, quote='on the far right')
    with pytest.raises(ValueError, match='contradicts.*ID 0'):
        check(requirement, measurement, quote='on the far right', result=0, rival=8)


def test_size_contradiction_needs_agreement_and_keeps_the_pair():
    requirement = {'condition': 'smallest', 'metric': 'volume_m3', 'scope_kind': 'explicit_pair', 'member_count': 2}
    measurement = {'objects': [{'id': 0, 'extent_xyz_m': [1, 2, 1.25], 'observed_box_volume_m3': 2.5,
                                'observed_footprint_m2': 2},
                               {'id': 8, 'extent_xyz_m': [1, 1.4, 0.5], 'observed_box_volume_m3': 0.7,
                                'observed_footprint_m2': 1.4}]}
    check(requirement, measurement, quote='the smallest of the two couches')
    with pytest.raises(ValueError, match='contradicts.*ID 0 has 2.5'):
        check(requirement, measurement, quote='the smallest of the two couches', result=0, rival=8)
    conflict = deepcopy(measurement)
    conflict['objects'][0]['observed_footprint_m2'] = 1
    check(requirement, conflict, quote='the smallest of the two couches', result=0, rival=8)
    with pytest.raises(ValueError, match='requires 2 members'):
        check(requirement, measurement, quote='the smallest of the two couches', additional_rivals=[9])


def test_nearest_requires_measuring_both_targets_to_the_same_bound_reference():
    requirement = {'condition': 'relation:nearest', 'reference_role': {'categories': ['chair']}}
    measurement = {'directed_pair_measurements': [
        {'object_id': 0, 'reference_id': 3, 'xy_center_m': 4, 'xy_box_gap_m': 3},
        {'object_id': 8, 'reference_id': 4, 'xy_center_m': 2, 'xy_box_gap_m': 1},
    ]}
    with pytest.raises(EvidenceFollowup, match='lacks cited.*common-reference distance') as error:
        check(requirement, measurement, quote='closest to the chair', anchor=3)
    assert set(error.value.focus_ids) == {0, 3, 8}
    assert error.value.condition_id == 'q1'
    measurement['directed_pair_measurements'][1]['reference_id'] = 3
    check(requirement, measurement, quote='closest to the chair', anchor=3)
    with pytest.raises(ValueError, match='contradicts'):
        check(requirement, measurement, quote='closest to the chair', result=0, rival=8, anchor=3)


@pytest.mark.parametrize('kind, quote, distances', [
    ('relation:farthest', 'second furthest from the chair', [4.4949, 3.1137, 2.4111]),
    ('relation:nearest', '2nd-closest to the chair', [1, 2, 3]),
])
def test_ordinal_distance_uses_measured_rank_in_full_bound_group(kind, quote, distances):
    requirement = {'condition': kind, 'reference_role': {'categories': ['chair']}}
    measurement = {'directed_pair_measurements': [
        {'object_id': candidate, 'reference_id': 3, 'xy_center_m': distance, 'xy_box_gap_m': distance - 0.5}
        for candidate, distance in zip([0, 8, 9], distances)]}
    check(requirement, measurement, quote=quote, anchor=3, additional_rivals=[9])
    with pytest.raises(ComparisonContradiction, match='contradicts.*ID 0 is not rank 2') as error:
        check(requirement, measurement, quote=quote, anchor=3, result=0, rival=8, additional_rivals=[9])
    assert (error.value.supported_id, error.value.anchor_id, error.value.ordinal) == (8, 3, 2)
    with pytest.raises(ValueError, match=r'full bound group.*still unbound \[10\]'):
        check(requirement, measurement, quote=quote, anchor=3, additional_rivals=[9], unbound=[10])
    with pytest.raises(ValueError, match='full bound group'):
        check(requirement, measurement, quote=quote, anchor=3, unbound=[9])
    with pytest.raises(ValueError, match='lacks cited'):
        check(requirement, {'directed_pair_measurements': measurement['directed_pair_measurements'][:2]},
              quote=quote, anchor=3, additional_rivals=[9])
    check(requirement, {'directed_pair_measurements': measurement['directed_pair_measurements'][:2]},
          quote=quote, anchor=3, additional_rivals=[9],
          second_measurement={'directed_pair_measurements': measurement['directed_pair_measurements'][1:]})

    measurement['directed_pair_measurements'][1]['xy_box_gap_m'] = distances[0] - 0.5
    with pytest.raises(ValueError, match='ambiguous rank 2'):
        check(requirement, measurement, quote=quote, anchor=3, additional_rivals=[9])


def test_size_ordinal_uses_bounded_measured_rank():
    requirement = {'condition': 'tallest', 'metric': 'height_m', 'scope_kind': 'full_category'}
    measurement = {'objects': [
        {'id': candidate, 'extent_xyz_m': [1, 1, height], 'observed_box_volume_m3': height}
        for candidate, height in [(0, 4), (9, 3), (8, 2), (10, 1)]]}
    check(requirement, measurement, quote='the third tallest couch',
          additional_rivals=[9, 10])
    with pytest.raises(ComparisonContradiction, match='ID 0 is not rank 3') as error:
        check(requirement, measurement, quote='the third tallest couch', result=0, rival=8, additional_rivals=[9, 10])
    assert error.value.supported_id == 8
