"""Deterministic, auditable normalization of a structured NR3D query."""

import re
from eviground.categories.retrieval import canonical_category, retrieval_category
from eviground.query.size_language import SIZE_WORDS, category_pattern, size_specs

RELATION_TYPES = {
    "nearest": "distance_min",
    "farthest": "distance_max",
    "near": "near",
    "left": "left_of",
    "right": "right_of",
    "front": "front_of",
    "back": "back_of",
    "above": "above",
    "below": "below",
    "between": "between",
}
VERTICAL_WORDS = {
    "over": "above",
    "above": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
}
CONTAINMENT_PATTERN = "\\b(?:includes?|including|contains?|containing)\\b[^.!?;]*"
DISTANCE_METADATA_KEYS = (
    "distance_metric",
    "metric",
    "distance_scope",
    "target_id",
    "target_instance_id",
    "anchor_id",
    "anchor_instance_id",
    "tie_tolerance_m",
)


def _category(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    canonical = canonical_category(value).strip()
    normalized = retrieval_category(canonical).strip()
    if not normalized and canonical in {"wall", "floor", "ground", "ceiling"}:
        return canonical
    return normalized


def _raw_category(value):
    return value.strip() if isinstance(value, str) else ""


def _quote(query, phrase):
    return (
        isinstance(phrase, str)
        and bool(phrase.strip())
        and (phrase.casefold() in query.casefold())
    )


def intrinsic_facing_condition(query, category):
    """Retain a quoted object side without inferring its physical orientation."""
    category = _category(category)
    if not category:
        return None
    pattern = (
        "\\b(?:facing|looking\\s+(?:at|towards?))\\s+(?:the\\s+)?(?P<part>front|backs?|head|foot)\\s+of\\s+(?:the\\s+)?(?P<objects>"
        + re.escape(category)
        + "s?)\\b"
    )
    match = re.search(pattern, query, re.I)
    if not match:
        return None
    return {
        "phrase": match.group(),
        "category": category,
        "part": (
            "back"
            if match["part"].casefold().startswith("back")
            else match["part"].casefold()
        ),
        "group": match["objects"].casefold() != category.casefold(),
    }


def full_category_size_comparison(query, parsed):
    """Recover a complete size request with its original word and quoted scope."""
    targets = parsed.get("Target", [])
    targets = [targets] if isinstance(targets, str) else targets
    if (
        not isinstance(targets, list)
        or len(targets) != 1
        or (not isinstance(targets[0], str))
        or (not targets[0].strip())
        or parsed.get("Relations")
    ):
        return None
    observer = parsed.get("Observer", {})
    if isinstance(observer, dict):
        origin = observer.get("origin", {})
        facing = observer.get("facing", {})
        explicit_origin = isinstance(origin, dict) and (
            origin.get("kind") in ("room_center", "category")
            or _quote(query, origin.get("phrase", ""))
        )
        explicit_facing = isinstance(facing, dict) and (
            facing.get("category") or _quote(query, facing.get("phrase", ""))
        )
        if explicit_origin or explicit_facing:
            return None
    category = canonical_category(targets[0])
    words = "|".join(sorted(SIZE_WORDS, key=len, reverse=True))
    noun = category_pattern(category)
    syntax_enabled = parsed.get("query_syntax", {}).get("backend") == "spacy"
    scene_scope = "(?P<scene_scope>\\s+in\\s+the\\s+room)?" if syntax_enabled else ""
    match = re.fullmatch(
        "\\s*(?:(?:choose|select|find|pick|click)\\s+|it\\s+is\\s+)?(?:the\\s+)?(?P<word>"
        + words
        + ")\\s+(?:(?P<scope>of\\s+the\\s+(?P<two>two\\s+)?)?(?P<category>"
        + noun
        + ")|(?P<paired_category>"
        + noun
        + ")\\s+of\\s+the\\s+(?P<end_two>two))"
        + scene_scope
        + "\\s*[.!?]?\\s*",
        query,
        re.I,
    )
    if (
        not match
        or category != _category(targets[0])
        or (
            match["scope"]
            and (not match["two"])
            and (match["category"].casefold() == category)
        )
    ):
        return None
    anchors = parsed.get("Anchor", [])
    anchors = [anchors] if isinstance(anchors, str) else anchors
    attributes = parsed.get("Attributes", [])
    if not isinstance(attributes, list) or any(
        (
            not isinstance(attribute, dict)
            or attribute.get("role") not in (None, "target")
            or _category(attribute.get("category", "")) != category
            or (not _quote(query, attribute.get("phrase", "")))
            for attribute in attributes
        )
    ):
        return None
    if anchors and (
        not isinstance(anchors, list)
        or any(
            (
                not isinstance(name, str)
                or not re.fullmatch(noun, canonical_category(name))
                for name in anchors
            )
        )
        or parsed.get("Viewpoint")
        or any((attribute.get("role") == "reference" for attribute in attributes))
    ):
        return None
    pair = bool(match["two"] or match["end_two"])
    return {
        "category": targets[0],
        "role": "target",
        "phrase": match["word"],
        "comparison_scope": "explicit_pair" if pair else "full_category",
        "scope_phrase": query.strip(),
        **({"required_member_count": 2} if pair else {}),
        **(
            {
                "source_phrase": query.strip(),
                "source_span": [len(query) - len(query.lstrip()), len(query.rstrip())],
                "scene_scope": (
                    "supplied_scene" if match.groupdict().get("scene_scope") else None
                ),
                **(
                    {
                        "scene_scope_assumption": "in the room refers to the supplied NR3D scene"
                    }
                    if match.groupdict().get("scene_scope")
                    else {}
                ),
            }
            if syntax_enabled
            else {}
        ),
    }


def _attribute_constraint(attribute, query, target_category):
    phrase = attribute.get("phrase", "") if isinstance(attribute, dict) else ""
    category = (
        _category(attribute.get("category", "")) if isinstance(attribute, dict) else ""
    )
    if not _quote(query, phrase) or not category:
        return None
    if re.search(CONTAINMENT_PATTERN, phrase, re.I):
        return None
    text = phrase.casefold()
    if category != target_category or attribute.get("role") == "reference":
        return None
    specs = size_specs(phrase, category, query=query, role="target")
    if specs:
        spec = specs[0]
        prefix = (
            "height"
            if spec["metric"] == "height_m"
            else (
                "length"
                if spec["metric"] == "horizontal_oriented_long_extent_m"
                else "size"
            )
        )
        kind = prefix + "_" + spec["direction"]
        if spec["interpretation"] == "descriptive_attribute":
            kind = "size_descriptive_attribute"
        return {
            "type": kind,
            "target": "target",
            "reference": None,
            "metric": spec["metric"],
            "phrase": phrase,
            "confidence": "lexical",
            "value": phrase,
            "remaining_attribute_phrase": re.sub(
                "\\b(?:"
                + "|".join((re.escape(item["word"]) for item in specs))
                + ")\\b",
                "",
                phrase,
                flags=re.I,
            ).strip(),
            "size_conditions": specs,
            **{
                key: attribute[key]
                for key in ("comparison_scope", "scope_phrase", "required_member_count")
                if key in attribute
            },
        }
    if re.search("\\b(on the floor|on floor|on the ground|grounded)\\b", text):
        return {
            "type": "bottom_z_near_floor",
            "target": "target",
            "reference": None,
            "metric": "bottom_z_m",
            "phrase": phrase,
            "confidence": "lexical",
        }
    return {
        "type": "attribute",
        "target": "target",
        "reference": None,
        "value": phrase,
        "phrase": phrase,
        "confidence": "lexical",
    }


def rewrite_query(query, parsed):
    """Compile parsed fields into stable constraints without choosing an object ID."""
    if not isinstance(query, str) or not isinstance(parsed, dict):
        raise ValueError("query and parsed query are required")
    raw_targets = parsed.get("Target", [])
    raw_targets = [raw_targets] if isinstance(raw_targets, str) else raw_targets
    targets = [
        {
            "category": _category(value),
            "raw_category": _raw_category(value),
            "role": "target",
            "phrase": value,
        }
        for value in raw_targets
        if _category(value)
    ]
    target_category = targets[0]["category"] if len(targets) == 1 else ""
    raw_references = parsed.get("Anchor", [])
    raw_references = (
        [raw_references] if isinstance(raw_references, str) else raw_references
    )
    references = []

    def add_reference(value, role="anchor", phrase=""):
        category = _category(value)
        if not category:
            return
        key = (category, role)
        if not any(((item["category"], item["role"]) == key for item in references)):
            references.append(
                {
                    "category": category,
                    "raw_category": _raw_category(value),
                    "role": role,
                    "phrase": phrase or value,
                }
            )

    for value in raw_references or []:
        add_reference(value)
    observer = (
        parsed.get("Observer") if isinstance(parsed.get("Observer"), dict) else {}
    )
    origin = observer.get("origin", {})
    facing = observer.get("facing", {})
    origin = origin if isinstance(origin, dict) else {}
    facing = facing if isinstance(facing, dict) else {}
    intrinsic_facing = intrinsic_facing_condition(query, facing.get("category", ""))
    if intrinsic_facing:
        facing = {
            **facing,
            "phrase": intrinsic_facing["phrase"],
            "group": intrinsic_facing["group"],
            "reference_frame": "intrinsic",
            "orientation_part": intrinsic_facing["part"],
        }
    for role, entry in (("observer_origin", origin), ("observer_facing", facing)):
        if entry.get("category"):
            add_reference(entry["category"], role, entry.get("phrase", ""))
    constraints = []
    unresolved = []
    if intrinsic_facing:
        unresolved.append(
            {
                "phrase": intrinsic_facing["phrase"],
                "reason": "object-intrinsic facing orientation is unavailable; object orientation is not inferred",
            }
        )
    for relation in parsed.get("Relations", []) or []:
        if not isinstance(relation, dict):
            unresolved.append({"phrase": "", "reason": "invalid relation object"})
            continue
        relation_type = relation.get("relation")
        relation_type = VERTICAL_WORDS.get(relation_type, relation_type)
        phrase = relation.get("phrase", "")
        target = _category(relation.get("target_category", ""))
        raw_references = (
            relation.get("anchor_categories", [])
            if relation_type == "between"
            else [relation.get("anchor_category", "")]
        )
        references_for_relation = [_category(value) for value in raw_references]
        directional = relation_type in ("left", "right", "front", "back")
        if directional and relation.get("reference_frame", "observer") != "observer":
            unresolved.append(
                {
                    "phrase": phrase,
                    "reason": "directional reference frame is unavailable; object orientation is not inferred",
                }
            )
            continue
        side_direction = directional and relation.get("scope") == "side"
        if (
            relation_type not in RELATION_TYPES
            or not target
            or (
                any((not reference for reference in references_for_relation))
                and (not side_direction)
            )
            or (not _quote(query, phrase))
            or (relation_type == "between" and len(references_for_relation) != 2)
        ):
            unresolved.append(
                {"phrase": phrase, "reason": "relation is incomplete or unquoted"}
            )
            continue
        for raw_reference in raw_references:
            add_reference(raw_reference, "anchor", phrase)
        constraint = {
            "type": RELATION_TYPES[relation_type],
            "target": target,
            "scope": relation.get("scope", ""),
            "phrase": phrase,
            "confidence": "lexical",
        }
        if directional:
            constraint["reference_frame"] = relation.get("reference_frame", "observer")
        if relation_type in ("nearest", "farthest", "near"):
            constraint.update(
                {
                    key: relation[key]
                    for key in DISTANCE_METADATA_KEYS
                    if key in relation
                }
            )
        if relation_type == "between":
            constraint.update(
                {
                    "references": references_for_relation,
                    "raw_reference_categories": [
                        _raw_category(value) for value in raw_references
                    ],
                    "raw_target_category": _raw_category(
                        relation.get("target_category", "")
                    ),
                }
            )
        else:
            constraint.update(
                {
                    "reference": references_for_relation[0],
                    "raw_target_category": _raw_category(
                        relation.get("target_category", "")
                    ),
                    "raw_reference_category": _raw_category(raw_references[0]),
                }
            )
        constraints.append(constraint)
    for attribute in parsed.get("Attributes", []) or []:
        if isinstance(attribute, dict) and attribute.get("role") == "reference":
            continue
        compiled = _attribute_constraint(attribute, query, target_category)
        if compiled:
            compiled["polarity"] = attribute.get("polarity", "positive")
            if attribute.get("negation_quote"):
                compiled["negation_quote"] = attribute["negation_quote"]
                compiled["executor"] = "negative_color"
            if (
                compiled.get("size_conditions")
                and parsed.get("query_syntax", {}).get("backend") == "spacy"
            ):
                from eviground.query.validation import size_conditions

                words = {spec["word"] for spec in compiled["size_conditions"]}
                compiled["size_conditions"] = [
                    spec
                    for spec in size_conditions(parsed, target_category)
                    if spec["word"] in words
                ]
            constraints.append(compiled)
        elif (
            isinstance(attribute, dict)
            and _quote(query, attribute.get("phrase"))
            and re.search(CONTAINMENT_PATTERN, attribute["phrase"], re.I)
        ):
            unresolved.append(
                {
                    "phrase": attribute["phrase"],
                    "reason": "containment/composition requires an unsupported relation predicate",
                }
            )
        elif isinstance(attribute, dict) and attribute.get("phrase"):
            unresolved.append(
                {
                    "phrase": attribute["phrase"],
                    "reason": "attribute category is not uniquely bound",
                }
            )
    for match in re.finditer(CONTAINMENT_PATTERN, query, re.I):
        phrase = match.group().rstrip()
        if not any(
            (phrase.casefold() in item["phrase"].casefold() for item in unresolved)
        ):
            unresolved.append(
                {
                    "phrase": phrase,
                    "reason": "containment/composition requires an unsupported relation predicate",
                }
            )
    for phrase in parsed.get("Unhandled", []) or []:
        if _quote(query, phrase):
            unresolved.append(
                {"phrase": phrase, "reason": "parser marked condition unsupported"}
            )
    for diagnostic in parsed.get("diagnostic_missing", []) or []:
        if not isinstance(diagnostic, str):
            continue
        if diagnostic.startswith("unsupported relation: "):
            phrase = diagnostic[len("unsupported relation: ") :]
            reason = "relation rejected by structure validation"
        elif diagnostic.startswith("unhandled condition: "):
            phrase = diagnostic[len("unhandled condition: ") :]
            reason = "condition marked unsupported by parser"
        elif diagnostic.startswith("appearance condition not represented: "):
            phrase = diagnostic[len("appearance condition not represented: ") :]
            reason = "appearance condition omitted from Attributes; role not inferred"
        elif diagnostic.startswith("immediate proximity has no calibrated predicate: "):
            phrase = diagnostic[
                len("immediate proximity has no calibrated predicate: ") :
            ]
            reason = "immediate proximity has no calibrated predicate"
        else:
            continue
        if _quote(query, phrase) and (
            not any((item["phrase"] == phrase for item in unresolved))
        ):
            unresolved.append({"phrase": phrase, "reason": reason})
    if re.search(
        "\\b(on the floor|on floor|on the ground|grounded)\\b", query, re.I
    ) and (not any((c["type"] == "bottom_z_near_floor" for c in constraints))):
        phrase = re.search(
            "\\b(on the floor|on floor|on the ground|grounded)\\b", query, re.I
        ).group()
        constraints.append(
            {
                "type": "bottom_z_near_floor",
                "target": "target",
                "reference": None,
                "metric": "bottom_z_m",
                "phrase": phrase,
                "confidence": "lexical_recovered",
            }
        )
    from eviground.grounding.settings import evidence_inventory

    return {
        "query": query,
        "target": targets,
        "references": references,
        "constraints": constraints,
        "unresolved": unresolved,
        "query_syntax": parsed.get("query_syntax"),
        "condition_coverage": parsed.get("condition_coverage", []),
        **{"evidence_inventory": evidence_inventory(query, parsed)},
        "language_reference": {
            "type": "observer",
            "origin": dict(origin),
            "facing": dict(facing),
            "capture_camera_independent": True,
        },
        "policy": "deterministic lexical normalization; no target ID selection",
    }
