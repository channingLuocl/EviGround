"""V7 source-bound box proxies and tentative, unweighted evidence selection.

No evaluation answers enter this module. Box/context preferences never delete
candidates or certify unresolved predicates; verified contradictions still veto.
"""

import re
from eviground.categories.retrieval import retrieval_category

COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "grey",
    "gray",
    "brown",
    "yellow",
    "orange",
    "purple",
    "pink",
}


def recover_part_category(query, parsed):
    match = re.search(
        "\\b(?:section|part|piece)s?\\s+of\\s+(?:the\\s+|a\\s+)?([a-z]+)", query, re.I
    )
    if not match or len(parsed.get("Target", [])) != 1:
        return
    original = parsed["Target"][0]
    mother = match[1]
    if retrieval_category(original) not in (
        "section",
        "part",
        "piece",
        retrieval_category(mother),
    ):
        return
    parsed["Target"] = [mother]
    if len(re.findall("\\b" + re.escape(mother) + "s?\\b", query, re.I)) == 1:
        parsed["Anchor"] = [
            a
            for a in parsed.get("Anchor", [])
            if retrieval_category(a) != retrieval_category(mother)
        ]
    parsed["category_recovery"] = {
        "original": match[0].split()[0],
        "canonical": retrieval_category(mother),
        "source_quote": match[0],
        "reason": "explicit part-of mother category",
    }
    for row in parsed.get("Attributes", []):
        if retrieval_category(row.get("category", "")) == retrieval_category(original):
            row["category"] = mother
    for row in parsed.get("Relations", []):
        if retrieval_category(row.get("target_category", "")) == retrieval_category(
            original
        ):
            row["target_category"] = mother


def bind_negative_attributes(query, parsed, exclusions):
    from eviground.query.query_graph import polarity
    from eviground.query.attributes import attribute_units

    rows = []
    for attribute in parsed.get("Attributes", []):
        for phrase in attribute_units(attribute.get("phrase", "")):
            value = polarity(query, phrase, exclusions, attribute=True)
            if value == "positive" or (
                value == "exclude" and phrase.casefold() in COLORS
            ):
                row = {**attribute, "phrase": phrase, "polarity": value}
                clause = next(
                    (
                        c
                        for c in exclusions
                        if c["source_span"][0]
                        <= query.casefold().find(phrase.casefold())
                        < c["source_span"][1]
                    ),
                    None,
                )
                if value == "exclude":
                    if not clause or not re.fullmatch(
                        "(?:not|except)\\s+(?:the\\s+)?"
                        + re.escape(phrase)
                        + "(?:\\s+one)?[.!?]?",
                        clause["source_quote"],
                        re.I,
                    ):
                        continue
                if value == "exclude" and clause:
                    row["negation_quote"] = clause["source_quote"]
                    if len(parsed.get("Target", [])) == 1 and re.fullmatch(
                        "(?:not|except)\\s+(?:the\\s+)?"
                        + re.escape(phrase)
                        + "\\s+one[.!?]?",
                        clause["source_quote"],
                        re.I,
                    ):
                        row.update(
                            category=parsed["Target"][0],
                            role="target",
                            binding_source="quoted contrastive one coreference",
                        )
                if row not in rows:
                    rows.append(row)
    for match in re.finditer(
        "\\b(?:not|except)\\s+(?:the\\s+)?(" + "|".join(sorted(COLORS)) + ")\\s+one\\b",
        query,
        re.I,
    ):
        if len(parsed.get("Target", [])) == 1 and (
            not any(
                (
                    r["phrase"].casefold() == match[1].casefold()
                    and r["polarity"] == "exclude"
                    for r in rows
                )
            )
        ):
            rows.append(
                {
                    "category": parsed["Target"][0],
                    "role": "target",
                    "phrase": match[1],
                    "polarity": "exclude",
                    "negation_quote": match[0],
                    "binding_source": "quoted contrastive one coreference",
                }
            )
    parsed["Attributes"] = rows
    covered = {r["negation_quote"] for r in rows if r.get("negation_quote")}
    parsed["_executed_negations"] = sorted(covered)
    parsed["Unhandled"] = [
        p
        for p in parsed.get("Unhandled", [])
        if p.rstrip(".!?") not in {c.rstrip(".!?") for c in covered}
    ]
    if covered and all((c["source_quote"] in covered for c in exclusions)):
        parsed["diagnostic_missing"] = [
            m
            for m in parsed.get("diagnostic_missing", [])
            if m != "negation/exclusion is unsupported"
            and (not any((m == "unhandled condition: " + c for c in covered)))
        ]


def box_size_spec(spec, category, query):
    spec = {
        **spec,
        "measurement_policy": "provided_instance_box",
        "metric_choice_source": "declared category/word semantics",
    }
    category = retrieval_category(category)
    if spec["word"] in {
        "tall",
        "tallest",
        "taller",
        "short",
        "shortest",
        "shorter",
        "high",
        "highest",
        "low",
        "lowest",
    }:
        explicit_tabletop = re.search(
            "\\b(?:tabletop|table top|desktop|desk top)\\b.*\\b(?:floor|ground)\\b",
            query,
            re.I,
        )
        metric = (
            "tabletop_height_m"
            if explicit_tabletop and category in ("table", "desk")
            else "height_m"
        )
        if metric == "tabletop_height_m":
            spec.pop("measurement_policy")
    elif spec["word"] == "longer":
        metric = "horizontal_long_extent_m"
    elif spec["word"] in {"wider", "widest", "narrower", "narrowest"}:
        metric = "horizontal_short_extent_m"
    elif category in ("door", "window"):
        metric = "vertical_box_face_area_m2"
    elif category in ("whiteboard", "blackboard", "mirror"):
        metric = "box_plane_area_m2"
    elif category in ("table", "desk") and (not re.search("\\bvolume\\b", query, re.I)):
        metric = "footprint_m2"
    elif category == "book":
        metric = "volume_m3"
    else:
        metric = "volume_m3"
    return {
        **spec,
        "metric": metric,
        "tentative_metric": True,
        "metric_reason": "provided instance box "
        + metric
        + "; observed geometry proxy, not completed physical object",
    }
