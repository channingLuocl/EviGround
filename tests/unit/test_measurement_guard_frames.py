"""Checks for binding measured axes to the observer in the source wording."""

import pytest

from eviground.grounding.evidence_followup import EvidenceFollowup
from eviground.grounding.measurement_guard import check_measured_exclusions


def check(description, condition, quote, frame, *, winner, rival, observer=None):
    requirements = [{'id': 'q0', 'source_quote': description, 'condition': 'complete description'},
                    {'id': 'q1', 'source_quote': quote, 'condition': condition}]
    node = {'id': 'n0', 'condition_ids': ['q1'], 'depends_on': [], 'status': 'completed',
            'bound_object_ids': [winner], 'excluded_target_ids': [rival],
            'evidence_ids': ['observation_1']}
    observations = [{'tool': 'measure', 'evidence_id': 'observation_1', 'selected_observer_frame': frame}]
    refs = [{'category': 'window', 'candidates': [{'id': n} for n in (6, 12, 46)]}]
    check_measured_exclusions(requirements, {winner, rival}, {'nodes': [node]}, observations, refs,
                              ['pillow'], observer=observer)


def test_top_right_rejects_reversed_facing_and_checks_signed_side():
    description = 'Facing the window, choose the blue pillow at the top right'
    positions = [[48, -0.7332, 1.8659], [41, 0.7332, 1.3806]]
    wrong_frame = {'origin_ids': [6, 12, 46], 'facing_ids': [41, 48], 'rows_left_to_right': positions}
    with pytest.raises(EvidenceFollowup, match='facing one reference') as missing:
        check(description, 'relation:right', 'at the top right', wrong_frame, winner=48, rival=41)
    assert missing.value.tool is None and missing.value.condition_id == 'q1'

    matching_frame = {'origin_ids': [], 'facing_ids': [46], 'rows_left_to_right': positions}
    with pytest.raises(ValueError, match='signed offset -0.7332'):
        check(description, 'relation:right', 'at the top right', matching_frame, winner=48, rival=41)
    check(description, 'relation:right', 'at the top right', matching_frame, winner=41, rival=48)


def test_literal_leftmost_survives_generic_parser_label():
    description = 'Facing the window, the leftmost file cabinet.'
    with pytest.raises(EvidenceFollowup, match='observer frame'):
        check(description, 'relation', 'the leftmost', None, winner=31, rival=6)
    frame = {'origin_ids': [], 'facing_ids': [46], 'rows_left_to_right': [[6, -2, 1], [31, 3, 1]]}
    with pytest.raises(ValueError, match='contradicts cited.*ID 31'):
        check(description, 'relation', 'the leftmost', frame, winner=31, rival=6)
    check(description, 'relation', 'the leftmost', frame, winner=6, rival=31)


def test_missing_candidate_in_an_existing_compatible_frame_requests_joint_measurement():
    description = 'Standing in the middle of the room facing the window, choose the leftmost pillow.'
    frame = {'origin_ids': [], 'facing_ids': [46], 'rows_left_to_right': [[6, -2, 1]]}
    with pytest.raises(EvidenceFollowup) as missing:
        check(description, 'appearance', 'leftmost', frame, winner=6, rival=31)
    assert missing.value.details() == {'tool': 'measure', 'focus_ids': [6, 31, 46],
                                       'frame': {'origin_ids': [], 'facing_ids': [46]}, 'condition_id': 'q1'}


def test_explicit_room_center_and_plural_facing_require_matching_roles():
    description = 'Standing in the middle of the room facing the windows, choose the far right pillow.'
    frame = {'origin_ids': [6], 'facing_ids': [46], 'rows_left_to_right': [[31, -2, 1], [41, 3, 1]]}
    with pytest.raises(EvidenceFollowup, match='facing a bound group'):
        check(description, 'relation:right', 'the far right', frame, winner=41, rival=31)
    check(description, 'relation:right', 'the far right', {**frame, 'origin_ids': [], 'facing_ids': [6, 46]},
          winner=41, rival=31)
