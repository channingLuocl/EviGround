"""Synthetic semantic audit, not a 3DVG benchmark or an ambiguity classifier.

Run from any directory with Python 3; stdout is the reproducible JSON result.
The old experiment remains unchanged. Centers, headings and all candidate IDs
below are hand-built fixtures, not ScanNet observations or human labels.
"""

import json
import math

from explore_horizontal_viewpoints import choose_candidate, projection


ANCHOR = {"id": 0, "center": [0, 0, 0]}


def objects(points):
    return [{"id": i + 1, "center": [x, y, 0]} for i, (x, y) in enumerate(points)]


def admissible(candidates, anchor, heading, relation, epsilon=1e-6):
    """Center-based strict half-plane convention; not full natural language."""
    sign = -1 if relation == "left" else 1
    return sorted(
        obj["id"] for obj in candidates
        if sign * projection(obj["center"], anchor["center"], heading) > epsilon
    )


def summarize(sets):
    if not any(sets):
        return "no_support"
    if not all(sets):
        return "partial_no_support"
    if any(len(ids) > 1 for ids in sets):
        return "within_view_nonunique"
    if len({ids[0] for ids in sets}) == 1:
        return "invariant_singleton"
    return "cross_view_conflict"


def main():
    fixtures = [
        ("opposite_directions", [(-1, 0), (1, 0)], [90, 270],
         [[1], [2]], "cross_view_conflict"),
        ("direction_supplied", [(-1, 0), (1, 0)], [90],
         [[1]], "invariant_singleton"),
        ("two_objects_on_left", [(-2, 0), (-1, 0)], [90],
         [[1, 2]], "within_view_nonunique"),
        ("no_object_on_left", [(1, 0), (2, 0)], [90],
         [[]], "no_support"),
        ("tied_objects_on_left", [(-1, -1), (-1, 1)], [90],
         [[1, 2]], "within_view_nonunique"),
        ("some_directions_have_no_support", [(-2, 0), (-1, 0)], [90, 270],
         [[1, 2], []], "partial_no_support"),
        ("on_relation_boundary", [(0, -1), (0, 1)], [90],
         [[]], "no_support"),
    ]
    results = []
    for name, points, headings, expected_sets, expected_state in fixtures:
        candidates = objects(points)
        sets = [admissible(candidates, ANCHOR, h, "left") for h in headings]
        state = summarize(sets)
        assert sets == expected_sets, name
        assert state == expected_state, name
        results.append({
            "case": name, "points_xy": points, "headings_deg": headings,
            "admissible_ids": sets, "state": state,
            "legacy_winner_ids": [
                choose_candidate(candidates, ANCHOR, "left", h)["id"]
                for h in headings
            ],
        })

    # Independent invariance checks: rigid transforms must not change relations.
    candidates = objects([(-2, 0), (1, 2), (3, -1)])
    base = admissible(candidates, ANCHOR, 90, "left")
    moved = [{**o, "center": [o["center"][0] + 7, o["center"][1] - 4, 0]}
             for o in candidates]
    assert admissible(moved, {"id": 0, "center": [7, -4, 0]}, 90, "left") == base
    angle = math.radians(37)
    rotated = [{**o, "center": [
        o["center"][0] * math.cos(angle) - o["center"][1] * math.sin(angle),
        o["center"][0] * math.sin(angle) + o["center"][1] * math.cos(angle), 0
    ]} for o in candidates]
    assert admissible(rotated, ANCHOR, 127, "left") == base
    assert admissible(candidates, ANCHOR, 270, "right") == base
    assert admissible(list(reversed(candidates)), ANCHOR, 90, "left") == base

    # Shrinking an externally justified heading set can only shrink the union.
    full = {i for h in [90, 270] for i in admissible(candidates, ANCHOR, h, "left")}
    assert set(base) <= full
    print(json.dumps({
        "scope": "synthetic semantic audit only; no real-data performance claim",
        "assumption": "known anchor, complete candidates, strict center half-plane left-of",
        "fixtures_passed": len(results), "invariance_checks_passed": 5,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
