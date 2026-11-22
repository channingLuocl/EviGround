"""Conflicting grounded visual findings keep relevant rivals visible for reinspection."""
from eviground.grounding.visual_conflicts import unresolved_visual_conflict


def finding(name, image, ids, text, evidence='initial', status='observed'):
    return {'finding_id': name, 'image_id': image, 'object_ids': ids,
            'observation': text, 'evidence_id': evidence, 'status': status}


def report(*findings, conflicts=()):
    return {'findings': findings, 'groups': [{'assessment': 'conflicting', 'finding_ids': ids}
                                            for ids in conflicts]}


def view(ids, context=None):
    return {'view': {'marked_ids': ids, 'context': context}}


def test_opposite_door_findings_keep_desk_and_both_rivals():
    first = report(
        finding('f01', 'a', [2], 'Door 2 has a wood desk (8) to its right.'),
        finding('f02', 'a', [3, 13], 'Doors 3 and 13 do not have a wood desk to their right.'),
        finding('f03', 'b', [3], 'Door 3 has a wood desk (8) to its right side.'),
        finding('f04', 'b', [2], 'Door 2 does not have a wood desk immediately to its right side.'),
        conflicts=[['f01', 'f04'], ['f02', 'f03']],
    )
    views = {'a': view([2, 3, 8, 13]), 'b': view([2, 3, 8, 13])}
    args = ([2, 3, 13], 3)
    query = 'Its the door with a wood desk to its right.'
    expected = {'candidate_ids': [2, 3], 'reference_ids': [8],
                'finding_ids': ['f03', 'f01', 'f04', 'f02'], 'evidence_ids': ['initial'],
                'criterion': 'right side'}
    assert unresolved_visual_conflict([first], views, *args, description=query) == expected

    views['c'] = view([3, 8], 'scene')
    narrow = report(finding('f05', 'c', [3, 8], 'Door 3 has a wood desk (8) to its right.', 'observation_2'))
    assert unresolved_visual_conflict([first, narrow], views, *args, description=query) == expected

    views['d'] = view([2, 3, 8], 'scene')
    joint = report(
        finding('f06', 'd', [2], 'Door 2 has a wood desk (8) to its right.', 'observation_3'),
        finding('f07', 'd', [3], 'Door 3 does not have a wood desk (8) to its right.', 'observation_3'),
    )
    result = unresolved_visual_conflict([first, narrow, joint], views, *args, description=query)
    assert result['refuted'] is True
    assert result['evidence_ids'] == ['initial', 'observation_3']
    assert unresolved_visual_conflict([first, narrow, joint], views, [2, 3, 13], 2, description=query) is None


def test_cross_batch_shelf_contradiction_needs_a_joint_view():
    broad = report(
        finding('f01', 'a', [12], 'Object 12 is a shelf located in the corner of the room.'),
        finding('f02', 'a', [13, 22, 23, 35, 36, 40],
                'Objects 13, 22, 23, 35, 36, and 40 are not shelves located in the corner.'),
    )
    paired_split = report(
        finding('f03', 'b', [12, 13, 22, 23, 35],
                'Objects 12, 13, 22, 23, and 35 are not shelves located in the corner.', 'observation_1'),
        finding('f04', 'c', [36], 'Object 36 is a shelf located in the corner of the room.', 'observation_1'),
    )
    views = {'a': view([12, 13, 22, 23, 35, 36, 40]), 'b': view([12, 13, 22, 23, 35], 'scene'),
             'c': view([36, 40], 'scene')}
    conflict = unresolved_visual_conflict([broad, paired_split], views, [12, 13, 22, 23, 35, 36, 40], 12,
                                          description='This shelf is in the corner.')
    assert conflict['candidate_ids'] == [12, 36]
    assert conflict['criterion'] == 'corner'


def test_two_sightings_and_unseen_candidates_are_not_a_contradiction():
    broad = report(
        finding('f01', 'a', [12], 'Object 12 is in the corner.'),
        finding('f02', 'a', [36], 'Cannot tell if object 36 is in the corner.'),
    )
    other = report(
        finding('f03', 'b', [36], 'Object 36 is in the corner.'),
        finding('f04', 'b', [12], 'Object 12 is not in the corner.', status='unknown'),
        conflicts=[['f01', 'f03']],
    )
    assert unresolved_visual_conflict([broad, other], {'a': view([12, 36]), 'b': view([12, 36])},
                                      [12, 36], 12, description='The shelf in the corner.') is None


def test_coordinator_group_links_opposite_top_positions_to_both_pillows():
    evidence = [
        finding('f01', 'a', [48], 'Object 48 is a blue pillow located at the top right when facing the window.'),
        finding('f03', 'b', [41], 'Object 41 is a blue pillow located at the top right when facing the window.'),
        finding('f04', 'b', [48], 'Object 48 is a blue pillow located at the top left when facing the window.'),
    ]
    views = {'a': view([41, 48]), 'b': view([41, 48])}
    query = 'Facing the window, choose the blue pillow at the top right'
    conflict = unresolved_visual_conflict([report(*evidence, conflicts=[['f01', 'f03', 'f04']])],
                                          views, [41, 48], 48, description=query)
    assert conflict['candidate_ids'] == [41, 48]
    assert unresolved_visual_conflict([report(*evidence)], views, [41, 48], 48, description=query) is None
