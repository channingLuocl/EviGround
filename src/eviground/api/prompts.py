"""Concise contracts for instruction models with limited prompt-following capacity."""

import os


def prompt_profile():
    value = os.getenv("EVIGROUND_PROMPT_PROFILE", "standard").strip()
    if value not in ("standard", "compact"):
        raise ValueError("EVIGROUND_PROMPT_PROFILE must be standard or compact")
    return value


PARSER = """Parse the user's object description into the requested JSON schema.
Target: the requested object category, singular, without color or size words.
Anchor: reference object categories, including references inside subordinate clauses.
Relations: preserve which noun each clause modifies. Use exact source phrases.
Use left/right/front/back/above/below/underneath/between for those relations;
near for next to; nearest/farthest only for explicit distance superlatives.
For between two beds, anchor_categories=["bed","bed"]. Unsupported relations
belong in Unhandled; never replace on, attached, middle or negation with near.
Observer: origin is where the viewer stands or sits; facing is what they look at.
Do not equate an observer's origin with what they face. Unspecified fields use
kind="unspecified", category="", phrase="", group=false. Viewpoint stays ""
unless the description supplies one. Attributes contain only actual appearance
or size modifiers, with category, role=target|reference, and an exact phrase.
Do not put ordinary noun phrases or spatial clauses into Attributes.
Preserve negation as a complete quote in Unhandled. Keep other positive clauses.
If attachment is uncertain, quote it in Unhandled rather than invent a relation.
Scene categories are fallible classifier hypotheses, not answers. Preserve the
described category even if absent. Syntax hints are language hints, not evidence.
Use empty arrays when absent. Do not repeat entries or invent object IDs.
Return only the JSON object, no commentary."""


PLANNER = """Locate the object in description. The original description overrides the parser.
allowed_target_ids is the eligible pool. Object IDs identify physical instances.
Classifier category hypotheses are alternatives, not simultaneous true names.
Ranks and gaps are uncalibrated priors; confirm alternatives visually.

Choose one action per turn:
There is no fixed action order. Never repeat an acquired observation; a rejected
duplicate supplies no new evidence. Read the returned tool results before acting.
The text after the images contains current observations and controller feedback;
use these results to revise task status and choose a different useful action.
inspect: focus_ids contains 1-8 known IDs; include relevant references as well
as competing targets. context="isolated" compares shape/color/category ONLY:
each tile has its own scale and camera, so tile order and size carry NO spatial
information. context="scene" checks support, grouping and relative placement.
Angles may be null (defaults 315 degrees azimuth, 30 degrees elevation), or
azimuth in [0,360), elevation in [10,75].
measure: focus_ids includes plausible targets AND the relevant reference group.
Request at most 8 objects per action; split larger comparisons across tool calls.
Use measured heights for top/bottom/stack order, dimensions for wider/narrower,
pair distances for near/nearest, overlap and vertical gaps for on/under, and a
supported observer frame for left/right. A visual guess does not replace a
discriminating numeric comparison when measure is available. frame=null unless
the description supports origin_ids/facing_ids. Empty origin_ids means room
center, not an arbitrary camera. The object faced is not the observer origin.
finish: predicted_id is one allowed ID or null. Compare the strongest rival,
explain observed distinguishing evidence, and name actual reference IDs.

Initial views and scene inspections are arbitrary cameras, not speaker left/right.
Basic coordinates are meters. Smaller bottom_z is lower; never reverse numeric
order to fit a guess. A bed's box does not identify its head or foot.
Measurement tables use columns and rows: each row follows the column order.
Bind references and qualifiers before comparing targets. All clauses must hold
for the SAME assignment: an object on a desk must use that support desk later.
Unknown appearance does not rule out a rival. Request missing discriminating
evidence while tools remain; inspecting isolated targets cannot prove proximity
to an absent reference. Do not simply restate the query as an observation.
Inspect alternative-only category targets before finish when the policy requires it.

For inspect/measure: predicted_id=null, evidence=[]. For inspect/finish: frame=null.
For finish: focus_ids=[], context=null, both angles=null. Keep reason under 60
words and evidence at most 4 entries. Each source_quote is the full description
verbatim, object_ids are known IDs, observation describes acquired evidence.
Use unresolved for missing clauses. When monitor_requirements is supplied, use
at most one entry per requirement. Never repeat an unresolved entry.
Return one JSON object, not a list."""
