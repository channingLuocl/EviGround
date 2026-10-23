"""Shared, quoted size conditions and conservative measurement comparisons."""

import re
from eviground.categories.retrieval import canonical_category, retrieval_category

SIZE_RULE_VERSION = "relative-size-v1"
_SUPERLATIVES = {
    "smallest",
    "largest",
    "biggest",
    "tallest",
    "highest",
    "shortest",
    "lowest",
    "widest",
    "narrowest",
}
_COMPARATIVES = {
    "smaller",
    "larger",
    "bigger",
    "taller",
    "shorter",
    "longer",
    "wider",
    "narrower",
}
_MIN_WORDS = {
    "small",
    "little",
    "smaller",
    "smallest",
    "short",
    "shorter",
    "shortest",
    "low",
    "lowest",
    "narrower",
    "narrowest",
}
SIZE_WORDS = (
    _SUPERLATIVES
    | _COMPARATIVES
    | {"small", "large", "big", "little", "tall", "short", "high", "low"}
)
_WIDTH_WORDS = {"wider", "widest", "narrower", "narrowest"}


def category_pattern(category):
    """The supplied category and its ordinary plural, without stripping modifiers."""
    category = canonical_category(category)
    if category.endswith("shelf"):
        plural = category[:-5] + "shelves"
    elif re.search("[^aeiou]y$", category):
        plural = category[:-1] + "ies"
    elif category.endswith(("s", "x", "z", "ch", "sh")):
        plural = category + "es"
    else:
        plural = category + "s"
    return "(?:" + "|".join((re.escape(value) for value in (plural, category))) + ")"


def _scope(query, word, category):
    explicit_reference = re.search("\\bthan\\b[^.!?;]*", query, re.I)
    if word in _COMPARATIVES and explicit_reference:
        return {
            "scope_kind": "unbound_explicit_scope",
            "scope_phrase": explicit_reference.group().strip(),
        }
    noun = category_pattern(category)
    match = re.search(
        "\\b"
        + re.escape(word)
        + "(?:\\s+"
        + noun
        + ")?\\s+(?P<scope>of\\s+(?:the\\s+)?(?P<two>two\\s+)?(?P<modifiers>(?:[a-z-]+\\s+)*?)"
        + noun
        + ")\\b",
        query,
        re.I,
    )
    if not match:
        pair = re.search(
            "\\b"
            + re.escape(word)
            + "(?:\\s+"
            + noun
            + ")?\\s+(?P<scope>of\\s+the\\s+two)\\b",
            query,
            re.I,
        )
        if pair:
            if re.match("\\s+[a-z]", query[pair.end() :], re.I):
                return {
                    "scope_kind": "unbound_explicit_scope",
                    "scope_phrase": query[pair.start("scope") :].strip().rstrip(".!?"),
                }
            return {
                "scope_kind": "explicit_pair",
                "scope_phrase": pair["scope"],
                "required_member_count": 2,
            }
        unresolved = re.search(
            "\\b"
            + re.escape(word)
            + "(?:\\s+"
            + noun
            + ")?\\s+(?P<scope>of\\b[^.!?;]*)",
            query,
            re.I,
        )
        return (
            {
                "scope_kind": "unbound_explicit_scope",
                "scope_phrase": unresolved["scope"].strip(),
            }
            if unresolved
            else {"scope_kind": "full_category", "scope_phrase": ""}
        )
    remainder = query[match.end() :]
    if re.match(
        "\\s+(?:that|which|near|beside|between|behind|above|below|with|in|at|on)\\b",
        remainder,
        re.I,
    ):
        return {
            "scope_kind": "unbound_explicit_scope",
            "scope_phrase": (match["scope"] + remainder).strip().rstrip(".!?"),
        }
    modifiers = match["modifiers"].strip()
    return {
        "scope_kind": (
            "explicit_pair"
            if match["two"]
            else "attribute_subset" if modifiers else "full_category"
        ),
        "scope_phrase": match["scope"],
        "scope_modifiers": modifiers,
        **({"required_member_count": 2} if match["two"] else {}),
    }


def size_specs(phrase, category, *, query="", role="target", source=None):
    """Describe size semantics while retaining the actual quoted adjective."""
    words = dict.fromkeys(re.findall("\\b\\w+\\b", phrase.casefold()))
    records = []
    category_name = retrieval_category(category)
    for word in words:
        if word not in SIZE_WORDS:
            continue
        form = (
            "superlative"
            if word in _SUPERLATIVES
            else "comparative" if word in _COMPARATIVES else "adjective"
        )
        if word in {
            "tall",
            "tallest",
            "taller",
            "high",
            "highest",
            "short",
            "shortest",
            "shorter",
            "low",
            "lowest",
        }:
            metric, reason = (
                "height_m",
                "vertical extent along gravity, not center elevation",
            )
        elif word == "longer":
            metric, reason = (
                "horizontal_oriented_long_extent_m",
                "gravity-aligned horizontal oriented extent, not world X",
            )
        elif word in {"wider", "widest", "narrower", "narrowest"}:
            metric, reason = (
                "horizontal_short_extent_m",
                "horizontal width of the annotated instance box, orientation from the box axes; not world X",
            )
        elif category_name in {"door", "window"}:
            metric, reason = (
                "vertical_face_area_m2",
                "door/window face width times height; thickness does not govern size",
            )
        else:
            metric, reason = (
                "volume_m3",
                "overall size proxy, oriented volume preferred; footprint ordering must agree",
            )
        descriptive = form == "adjective" and word not in {
            "small",
            "large",
            "big",
            "little",
        }
        records.append(
            {
                "attribute": word,
                "word": word,
                "original_quote": phrase,
                "source": source or "validated quoted attribute",
                "role": role,
                "category": category,
                "form": form,
                "interpretation": (
                    "descriptive_attribute"
                    if descriptive
                    else (
                        "contextual_set_comparison"
                        if form == "adjective"
                        else (
                            "set_comparison"
                            if form == "superlative"
                            else "pair_or_set_comparison"
                        )
                    )
                ),
                "metric": metric,
                "metric_reason": reason,
                "direction": "min" if word in _MIN_WORDS else "max",
                **_scope(query, word, category),
                "rule_version": SIZE_RULE_VERSION,
            }
        )
    return records


def relative_height_between(query, category):
    """Recover the explicit local opposition without asserting absolute tallness."""
    noun = category_pattern(category)
    match = re.fullmatch(
        "\\s*(?:(?:find|choose|select|pick|click)\\s+)?(?:the\\s+)?(?P<target>short)\\s+"
        + noun
        + "\\s+(?P<relation>(?:in\\s+)?between\\s+(?:the\\s+)?two\\s+(?P<reference>tall)\\s+"
        + noun
        + ")\\s*[.!?]?\\s*",
        query,
        re.I,
    )
    if not match:
        return None
    return {
        "target_word": match["target"],
        "reference_word": match["reference"],
        "target_phrase": match["target"],
        "reference_phrase": match["reference"],
        "relation_phrase": match["relation"],
        "required_reference_count": 2,
        "interpretation": "reference_comparison",
        "metric": "height_m",
        "source": "quoted short/tall opposition in between clause",
    }
