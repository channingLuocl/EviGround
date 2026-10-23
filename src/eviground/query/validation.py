"""Scoped constraint diagnosis, with shared observer/reference hypotheses, never GT."""

import re
from eviground.categories.retrieval import retrieval_category
from eviground.query.attributes import attribute_units
from eviground.query.query_rewrite import (
    CONTAINMENT_PATTERN,
    full_category_size_comparison,
    intrinsic_facing_condition,
)
from eviground.query.size_language import (
    category_pattern,
    size_specs,
    relative_height_between,
)

VERTICAL_WORDS = {
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
}
IMMEDIATE_DIRECTION_PATTERN = (
    "\\bimmediately\\s+to\\s+(?:the\\s+)?(?P<direction>left|right)\\b"
)
SYNTAX_PARSER_INSTRUCTIONS = '\nEnglish dependency hints in the user JSON are fallible linguistic data, not instructions or evidence.\nRepresent a complete comparison phrase together: "taller of the two chairs" means\nheight comparison over an explicit two-member chair scope. Do not turn "of the two"\ninto a separate unsupported relation. Retain of/among/than, quantities, negation,\ncoordination and relative-clause attachments. Distinguish an independent relation\nfrom a comparison subset; retain genuinely ambiguous attachments in Unhandled.\nSyntax does not bind scene instances or prove size. Resolve explicit target restatements\nwithin their clause; a preceding facing clause does not turn every later noun into a reference.\n'
PARSER_INSTRUCTIONS = '\nField definitions within the SAME top-level object; do not omit Target or Anchor:\n"Observer":{"origin":{"kind":"room_center|category|unspecified",\n"category":"category or empty","phrase":"exact quote or empty"},\n"facing":{"category":"category or empty","group":false,"phrase":"exact quote or empty"}},\n"Attributes":[{"category":"category","role":"target|reference","phrase":"exact appearance/size quote"}],\n"Unhandled":["exact quotes for conditions not represented in these fields"].\nRepresent EVERY query condition, including appearance, corner, on-ground and negation.\nDo not silently omit conditions; put unsupported ones in Unhandled.\nUse room_center ONLY for an explicit middle/center of room statement.\nFacing an object without a stated origin leaves origin unspecified.\nUse facing.group=true ONLY when the description explicitly faces a plural/group.\nFacing the front/back of objects requires their intrinsic orientation; retain the\ncomplete quote and put that orientation condition in Unhandled.\nDirection relations use left|right|front|back, plus scope=relative_anchor|side.\nrelative_anchor means target is on a named reference\'s side in the observer frame;\nside means a signed side of the observer, NOT leftmost/rightmost among peers.\nSet reference_frame=observer for these implemented directional predicates.\nIf a condition explicitly uses a reference object\'s own front/back axes, retain\nreference_frame=intrinsic and put the condition in Unhandled; a point-cloud long\naxis or an inspection camera does not establish the object\'s front direction.\nUse between with anchor_categories for a target located between two named references.\nRepresent smallest/largest/biggest/tallest/shortest/highest/lowest and the\ncomparatives larger/smaller/bigger/taller/shorter as Attributes of the noun they\nmodify, with target or reference role as quoted; do not force them onto the target.\nAdjective phrases such as tall skinny are Attributes, never Relations. Preserve every modifier.\nBind attributes to the noun they describe; reference colors/materials do not describe the target.\nWhen target and reference share a category, supply their distinct attribute roles.\nUse other+Unhandled for unsupported extremes, object-intrinsic axes, ordinal or unsupported relations.\nIncludes/contains a named object is a containment/composition relation, not appearance;\nuse other+Unhandled because this relation has no implemented predicate.\nSeparate facing references from target relations; include their categories in Anchor.\nResolve pronouns to an actual category in the description, never use a pronoun as a category.\nEvery relation and observer phrase must be a nonempty EXACT quote.\nFor Observer use the full standing/facing clause; do not invent the speaker\'s position.\n'


def quoted(query, phrase):
    return (
        isinstance(phrase, str)
        and bool(phrase.strip())
        and (phrase.casefold() in query.casefold())
    )


def _room_center_anchor(name):
    normalized = (
        re.sub("[_-]+", " ", name.strip().casefold()) if isinstance(name, str) else ""
    )
    return normalized in ("room", "room center", "center of room", "middle of room")


def _explicit_room_center_origin(query, observer):
    origin = observer.get("origin", {}) if isinstance(observer, dict) else {}
    if not isinstance(origin, dict) or origin.get("kind") != "room_center":
        return False
    phrase = origin.get("phrase", "")
    return (
        quoted(query, phrase)
        and re.search("\\b(middle|cent(?:er|re))\\b", phrase, re.I)
        and re.search("\\broom\\b", phrase, re.I)
    )


def _complete_observer_coreference_clause(query, observer, relation):
    origin = observer.get("origin", {})
    facing = observer.get("facing", {})
    phrases = (
        origin.get("phrase", ""),
        facing.get("phrase", ""),
        relation.get("phrase", ""),
    )
    if not all((isinstance(phrase, str) and phrase for phrase in phrases)):
        return False
    pattern = (
        "\\s*(?:when\\s+)?"
        + re.escape(phrases[0])
        + "[\\s,]+"
        + re.escape(phrases[1])
        + "[\\s,]+"
        + re.escape(phrases[2])
        + "[.!?]*\\s*"
    )
    return bool(re.fullmatch(pattern, query, re.I))


def recover_distance_relations(query, parsed):
    """Recover only a complete, simple distance clause with explicit bindings."""
    parsed = dict(parsed)
    raw_relations = parsed.get("Relations", [])
    if not isinstance(raw_relations, list) or any(
        (not isinstance(r, dict) for r in raw_relations)
    ):
        return parsed
    relations = list(raw_relations)
    targets = parsed.get("Target") if isinstance(parsed.get("Target"), list) else []
    anchors = parsed.get("Anchor") if isinstance(parsed.get("Anchor"), list) else []
    if len(targets) != 1 or len(anchors) != 1:
        return parsed
    if any(
        (
            isinstance(r, dict) and r.get("relation") in ("nearest", "farthest")
            for r in relations
        )
    ):
        return parsed
    target, anchor = (name.strip() for name in (targets[0], anchors[0]))
    if not target or not anchor:
        return parsed
    pattern = (
        "(?:it is\\s+)?(?:the\\s+)?"
        + re.escape(target)
        + "s?\\s+(?:is\\s+)?(?P<operator>nearest|closest|farthest|furthest)\\s+(?:away\\s+)?(?:to\\s+|from\\s+)?(?:the\\s+)?"
        + re.escape(anchor)
        + "s?[.!?]?"
    )
    match = re.fullmatch(pattern, query.strip(), re.I)
    if match:
        kind = (
            "nearest"
            if match["operator"].lower() in ("nearest", "closest")
            else "farthest"
        )
        recovered = {
            "target_category": target,
            "anchor_category": anchor,
            "relation": kind,
            "scope": "",
            "phrase": query.strip(),
        }
        parsed["Relations"] = relations + [recovered]
        parsed["recovered_distance_relations"] = [recovered]
    return parsed


def recover_vertical_relations(query, parsed):
    """Recover one explicit vertical clause when the parser omitted or mislabeled it."""
    parsed = dict(parsed)
    raw_relations = parsed.get("Relations", [])
    if not isinstance(raw_relations, list) or any(
        (not isinstance(r, dict) for r in raw_relations)
    ):
        return parsed
    relations = [dict(r) for r in raw_relations]
    targets = parsed.get("Target") if isinstance(parsed.get("Target"), list) else []
    anchors = parsed.get("Anchor") if isinstance(parsed.get("Anchor"), list) else []
    anchors = (
        list(dict.fromkeys(anchors))
        if all((isinstance(a, str) for a in anchors))
        else anchors
    )
    if len(targets) != 1 or not anchors:
        return parsed

    def embedded_inverse(text, anchor_name, source_phrase=None):
        if not isinstance(anchor_name, str) or not anchor_name.strip():
            return None
        anchor_pattern = re.escape(anchor_name.strip()) + "s?"
        matches = list(
            re.finditer(
                "(?P<subject>\\b"
                + anchor_pattern
                + ")\\s+(?P<operator>right\\s+above|directly\\s+above|above|over|below|directly\\s+under|under|underneath|beneath)\\s+(?P<object>it|them|the target)\\b",
                text,
                re.I,
            )
        )
        if source_phrase is not None:
            quote = source_phrase.strip().rstrip(".!?").casefold()
            matches = [match for match in matches if quote in match.group().casefold()]
            if len(matches) != 1:
                return None
        if not matches:
            return None
        match = matches[0]
        if re.search(
            "\\b(?:above|over|below|under|underneath|beneath)\\b",
            text[: match.start("subject")],
            re.I,
        ):
            return None
        operator = (
            re.search(
                "\\b(above|over|below|under|underneath|beneath)\\b",
                match["operator"],
                re.I,
            )
            .group()
            .lower()
        )
        relation = "below" if operator in ("above", "over") else "above"
        start = match.start("subject")
        return {"relation": relation, "start": start, "end": match.end()}

    unresolved_short_quote = False
    for relation in relations:
        anchor_value = relation.get("anchor_category")
        if "anchor_category" in relation and (not isinstance(anchor_value, str)):
            continue
        phrase = relation.get("phrase", "")
        if not isinstance(phrase, str):
            continue
        anchor_name = (
            anchor_value.strip()
            if isinstance(anchor_value, str) and anchor_value.strip()
            else anchors[0] if len(anchors) == 1 else None
        )
        if not isinstance(anchor_name, str) or not anchor_name.strip():
            continue
        inverse = embedded_inverse(phrase, anchor_name)
        if not inverse and re.fullmatch(
            "(?:directly\\s+)?(?:above|over|below|under|underneath|beneath)\\s+(?:it|them|the target)[.!?]?",
            phrase.strip(),
            re.I,
        ):
            inverse = embedded_inverse(query, anchor_name, source_phrase=phrase)
            if not inverse:
                relation["relation"] = "other"
                unresolved_short_quote = True
                continue
        if inverse:
            relation["relation"] = inverse["relation"]
            relation["scope"] = "relative_anchor"
            relation["target_category"] = relation.get("target_category") or targets[0]
            if not anchor_value:
                relation["anchor_category"] = anchor_name
            relation["source_subject"] = anchor_name
            relation["source_object"] = targets[0]
            continue
        match = re.search(
            "\\b(?:right\\s+above|directly\\s+above|above|over|below|directly\\s+under|under|underneath|beneath)\\b",
            phrase,
            re.I,
        )
        if match:
            relation["relation"] = VERTICAL_WORDS[
                re.search(
                    "\\b(above|over|below|under|underneath|beneath)\\b",
                    match.group(),
                    re.I,
                )
                .group()
                .lower()
            ]
            relation["scope"] = "relative_anchor"
            relation["target_category"] = relation.get("target_category") or targets[0]
            if not anchor_value:
                relation["anchor_category"] = anchor_name
    if unresolved_short_quote or any(
        (r.get("relation") in ("above", "below") for r in relations)
    ):
        parsed["Relations"] = relations
        return parsed
    if len(anchors) != 1:
        parsed["Relations"] = relations
        return parsed
    inverse = embedded_inverse(query, anchors[0])
    if inverse:
        phrase = query[inverse["start"] : inverse["end"]].strip().rstrip(".,!?")
        relations.append(
            {
                "target_category": targets[0],
                "anchor_category": anchors[0],
                "relation": inverse["relation"],
                "scope": "relative_anchor",
                "phrase": phrase,
            }
        )
        parsed["Relations"] = relations
        return parsed
    vertical = re.search(
        "\\b(?:right\\s+above|directly\\s+above|above|over|below|directly\\s+under|under|underneath|beneath)\\b",
        query,
        re.I,
    )
    if not vertical:
        parsed["Relations"] = relations
        return parsed
    operator = (
        re.search(
            "\\b(above|over|below|under|underneath|beneath)\\b", vertical.group(), re.I
        )
        .group()
        .lower()
    )
    tail = query[vertical.end() :]
    tail_match = re.match("\\s+(?:the|a|an)\\s+[A-Za-z][A-Za-z -]*", tail)
    end = vertical.end() + (tail_match.end() if tail_match else 0)
    phrase = query[vertical.start() : end].strip().rstrip(".,!?")
    relations.append(
        {
            "target_category": targets[0],
            "anchor_category": anchors[0],
            "relation": "above" if operator in ("above", "over") else "below",
            "scope": "relative_anchor",
            "phrase": phrase,
        }
    )
    parsed["Relations"] = relations
    return parsed


def recover_between_relations(query, parsed):
    """Recover a two-reference horizontal relation omitted by the parser."""
    parsed = dict(parsed)
    raw_relations = parsed.get("Relations", [])
    if not isinstance(raw_relations, list) or any(
        (not isinstance(r, dict) for r in raw_relations)
    ):
        return parsed
    relations = [dict(r) for r in raw_relations]
    targets = parsed.get("Target") if isinstance(parsed.get("Target"), list) else []
    anchors = parsed.get("Anchor") if isinstance(parsed.get("Anchor"), list) else []
    if len(targets) != 1 or len(anchors) < 2:
        parsed["Relations"] = relations
        return parsed
    for relation in relations:
        if relation.get("relation") == "between":
            names = relation.get("anchor_categories")
            if (names is None or (isinstance(names, list) and len(names) != 2)) and len(
                anchors
            ) >= 2:
                relation["anchor_categories"] = [
                    name.strip()
                    for name in anchors[:2]
                    if isinstance(name, str) and name.strip()
                ]
            relation.setdefault("scope", "relative_anchors")
            parsed["Relations"] = relations
            return parsed
    match = re.search("\\bbetween\\b", query, re.I)
    if not match:
        parsed["Relations"] = relations
        return parsed
    anchor_categories = [
        name.strip() for name in anchors[:2] if isinstance(name, str) and name.strip()
    ]
    if len(anchor_categories) != 2:
        parsed["Relations"] = relations
        return parsed
    phrase = query[match.start() :].strip().rstrip(".,!?")
    relations.append(
        {
            "target_category": targets[0],
            "anchor_categories": anchor_categories,
            "relation": "between",
            "scope": "relative_anchors",
            "phrase": phrase,
        }
    )
    parsed["Relations"] = relations
    parsed["recovered_between_relations"] = [relations[-1]]
    return parsed


def covered_attribute_request(query, parsed):
    targets = parsed.get("Target", [])
    targets = [targets] if isinstance(targets, str) else targets
    if len(targets) != 1 or parsed.get("Anchor") or parsed.get("Relations"):
        return False
    words = []
    allowed = set(
        "small smallest smaller large largest larger big biggest bigger little skinny narrow wide thin thick tall tallest taller high highest short shortest shorter low lowest dark light black white red blue green gray grey brown wooden mounted square round".split()
    )
    for attribute in parsed.get("Attributes", []):
        if retrieval_category(attribute.get("category", "")) != retrieval_category(
            targets[0]
        ):
            return False
        tokens = attribute.get("phrase", "").casefold().split()
        if not tokens or not set(tokens).issubset(allowed):
            return False
        words.extend(tokens)
    if not words:
        return False
    match = re.fullmatch(
        "\\s*(?:(?:choose|select|find)\\s+|it is\\s+)?(?:the\\s+)?(?P<attributes>[a-z ]+)\\s+"
        + re.escape(targets[0])
        + "\\s*[.!?]?\\s*",
        query,
        re.I,
    )
    return bool(
        match and sorted(match["attributes"].casefold().split()) == sorted(words)
    )


def attribute_role(parsed, attribute):
    """Legacy category-only bindings are usable only for a single language role."""
    if attribute.get("role") in ("target", "reference"):
        return attribute["role"]
    category = retrieval_category(attribute.get("category", ""))
    targets = parsed.get("Target", [])
    targets = [targets] if isinstance(targets, str) else targets
    references = parsed.get("Anchor", [])
    references = [references] if isinstance(references, str) else list(references)
    for relation in parsed.get("Relations", []):
        if isinstance(relation, dict):
            names = (
                relation.get("anchor_categories")
                if relation.get("relation") == "between"
                else [relation.get("anchor_category", "")]
            )
            references.extend(names if isinstance(names, list) else [])
    observer = parsed.get("Observer", {})
    if isinstance(observer, dict):
        references.extend(
            (
                entry.get("category", "")
                for entry in observer.values()
                if isinstance(entry, dict)
            )
        )
    target_match = category in {retrieval_category(name) for name in targets}
    reference_match = category in {retrieval_category(name) for name in references}
    if target_match != reference_match:
        return "target" if target_match else "reference"
    return None


def _source_attribute_bindings(query, parsed, phrase):
    """Find direct noun attachments without collapsing repeated language roles."""
    if not quoted(query, phrase):
        return set()
    recovery = parsed.get("category_recovery")
    if recovery:
        part = re.search(
            "\\b" + re.escape(recovery["source_quote"]) + "\\b", query, re.I
        )
        occurrences = list(re.finditer(re.escape(phrase), query, re.I))
        if (
            part
            and len(occurrences) == 1
            and (occurrences[0].end() <= part.start())
            and re.fullmatch(
                "(?:\\s+(?:big|small|large|largest|smallest|brown|white|round))*\\s*",
                query[occurrences[0].end() : part.start()],
                re.I,
            )
        ):
            return {(recovery["canonical"], "target")}
    target_names = parsed.get("Target", [])
    target_names = [target_names] if isinstance(target_names, str) else target_names
    anchor_names = parsed.get("Anchor", [])
    anchor_names = [anchor_names] if isinstance(anchor_names, str) else anchor_names
    names = target_names + anchor_names
    targets = {retrieval_category(name) for name in target_names}
    observer_spans = []
    for entry in parsed.get("Observer", {}).values():
        if isinstance(entry, dict) and quoted(query, entry.get("phrase", "")):
            observer_spans.extend(
                (
                    (match.start(), match.end())
                    for match in re.finditer(re.escape(entry["phrase"]), query, re.I)
                )
            )
    mentions = []
    for name in names:
        category = retrieval_category(name)
        if not category:
            continue
        for match in re.finditer(
            "\\b" + category_pattern(category) + "\\b", query, re.I
        ):
            key = (match.start(), match.end(), category)
            if key not in mentions:
                mentions.append(key)
    syntax_enabled = parsed.get("query_syntax", {}).get("backend") == "spacy"
    if syntax_enabled:
        mentions = [
            (a, b, c)
            for a, b, c in mentions
            if not any(
                (
                    left <= a and b <= right and ((left, right) != (a, b))
                    for left, right, _ in mentions
                )
            )
        ]
    attachments = []
    direct_connector = "(?:with|is|has|that\\s+(?:is|has))\\s+(?:(?:the|a|an)\\s+)?"
    spatial_words = "\\b(?:above|over|below|under|underneath|beneath|between|near|beside|next to|left of|right of|behind|in front of|facing)\\b"
    head_modifiers = _SOURCE_VISUAL_MODIFIERS | {
        "small",
        "smallest",
        "smaller",
        "large",
        "largest",
        "larger",
        "big",
        "biggest",
        "bigger",
        "little",
        "skinny",
        "narrow",
        "thin",
        "wide",
        "thick",
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
    }
    for attr_match in re.finditer(re.escape(phrase), query, re.I):
        embedded_nouns = [
            (a, b, c)
            for a, b, c in mentions
            if attr_match.start() <= a and b <= attr_match.end()
        ]
        for start, end, category in mentions:
            between_tokens = re.findall(
                "\\b\\w+\\b", query[attr_match.end() : start].casefold()
            )
            before_noun = (
                attr_match.end() <= start
                and all((token in head_modifiers for token in between_tokens))
                and (not re.search("[^a-z\\s-]", query[attr_match.end() : start], re.I))
            )
            gap = query[end : attr_match.start()]
            after_noun = (
                end <= attr_match.start()
                and (not embedded_nouns)
                and (not re.search(spatial_words, phrase, re.I))
                and bool(
                    re.fullmatch("\\s+" + direct_connector, gap, re.I)
                    or (
                        re.fullmatch("\\s+", gap)
                        and re.match(direct_connector + "\\S", phrase, re.I)
                    )
                )
            )
            if any(
                (
                    attr_match.end() <= noun_start
                    and all(
                        (
                            token in head_modifiers
                            for token in re.findall(
                                "\\b\\w+\\b",
                                query[attr_match.end() : noun_start].casefold(),
                            )
                        )
                    )
                    and (
                        not re.search(
                            "[^a-z\\s-]", query[attr_match.end() : noun_start], re.I
                        )
                    )
                    for noun_start, _, _ in mentions
                )
            ):
                after_noun = False
            prefix_tokens = re.findall(
                "\\b\\w+\\b", query[attr_match.start() : start].casefold()
            )
            if prefix_tokens and prefix_tokens[0] in {
                "the",
                "a",
                "an",
                "another",
                "other",
            }:
                prefix_tokens = prefix_tokens[1:]
            embedded_head = (
                embedded_nouns == [(start, end, category)]
                and all((token in head_modifiers for token in prefix_tokens))
                and (not re.search(spatial_words, query[end : attr_match.end()], re.I))
            )
            if not (before_noun or after_noun or embedded_head):
                continue
            commands = list(
                re.finditer(
                    "\\b(?:choose|select|find|pick|click)\\b", query[:start], re.I
                )
            )
            clause_start = commands[-1].end() if commands else 0
            if syntax_enabled:
                restatements = list(
                    re.finditer(
                        "\\bthe\\s+(?:correct|target)\\s+(?:object|"
                        + category_pattern(category)
                        + ")\\s+is\\s+",
                        query[clause_start:start],
                        re.I,
                    )
                )
                if category in targets and restatements:
                    clause_start += restatements[-1].end()
            prefix = query[clause_start:start]
            earlier_target = any(
                (clause_start <= a < start and c in targets for a, _, c in mentions)
            )
            in_reference_clause = bool(
                re.search(
                    "\\b(?:facing|standing|looking|above|over|below|under|underneath|beneath|between|near|beside|next to|left of|right of|behind|in front of|with)\\b",
                    prefix,
                    re.I,
                )
            )
            role = (
                "target"
                if category in targets
                and (not earlier_target)
                and (not in_reference_clause)
                else "reference"
            )
            if (
                role == "reference"
                and category in targets
                and (not any((a <= start and end <= b for a, b in observer_spans)))
            ):
                earlier_nonobserver_target = any(
                    (
                        clause_start <= a < start
                        and c in targets
                        and (
                            not any(
                                (
                                    left <= a and b <= right
                                    for left, right in observer_spans
                                )
                            )
                        )
                        for a, b, c in mentions
                    )
                )
                spatial_reference = re.search(
                    "\\b(?:above|over|below|under|underneath|beneath|between|near|beside|next to|left of|right of|behind|in front of|with)\\b",
                    prefix,
                    re.I,
                )
                if not earlier_nonobserver_target and (not spatial_reference):
                    attachments.append((category, "target"))
            attachments.append((category, role))
    bindings = set(attachments)
    if (
        not bindings
        and len(targets) == 1
        and re.search(
            "\\bthe (?:correct|target) one\\s+(?:is|has)\\s+" + re.escape(phrase),
            query,
            re.I,
        )
    ):
        bindings.add((next(iter(targets)), "target"))
    return bindings


_SOURCE_VISUAL_MODIFIERS = {
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
    "rainbow",
    "wooden",
    "metal",
    "metallic",
    "plastic",
    "glass",
    "striped",
    "dotted",
    "checkered",
    "speckled",
    "smooth",
    "rough",
    "textured",
    "folded",
    "vertical",
    "horizontal",
    "round",
    "square",
    "rectangular",
    "dark",
    "light",
}


def _recover_surface_attributes(query, parsed, attributes):
    """Recover an omitted visual property only when it names a clear object part."""
    parts = "(?:top|bottom|front|back|side|surface|panel|upholstery|fabric)"
    modifier = "|".join(sorted(_SOURCE_VISUAL_MODIFIERS, key=len, reverse=True))
    pattern = re.compile(
        "\\bwith\\s+(?:the\\s+|a\\s+|an\\s+)?(?P<phrase>(?:"
        + modifier
        + ")\\s+"
        + parts
        + ")\\b",
        re.I,
    )
    existing = {
        (
            retrieval_category(item.get("category", "")),
            item.get("phrase", "").casefold(),
        )
        for item in attributes
        if isinstance(item, dict)
    }
    missing = []
    for match in pattern.finditer(query):
        phrase = match.group("phrase")
        bindings = _source_attribute_bindings(query, parsed, phrase)
        if len(bindings) != 1:
            missing.append("appearance condition role not represented: " + phrase)
            continue
        category, role = next(iter(bindings))
        key = (retrieval_category(category), phrase.casefold())
        if key not in existing:
            attributes.append(
                {
                    "category": category,
                    "role": role,
                    "phrase": phrase,
                    "binding_source": "quoted surface attribute recovery",
                }
            )
            existing.add(key)
    return (attributes, missing)


def _normalize_relation_scope(query, relation, targets, observer=None):
    relation = dict(relation)
    kind, phrase = (relation.get("relation"), relation.get("phrase", ""))
    if not quoted(query, phrase):
        return relation
    anchor = relation.get("anchor_category")
    facing = observer.get("facing", {}) if isinstance(observer, dict) else {}
    target = targets[0] if len(targets) == 1 else ""
    if (
        kind in ("left", "right")
        and target
        and isinstance(facing, dict)
        and (facing.get("group") is True)
        and _explicit_room_center_origin(query, observer)
        and (
            retrieval_category(target)
            == retrieval_category(facing.get("category", ""))
            == retrieval_category(anchor or "")
        )
        and quoted(query, facing.get("phrase"))
        and re.search("\\b(?:fac(?:e|ing)|look(?:ing)?)\\b", facing["phrase"], re.I)
        and re.fullmatch(
            "(?:the\\s+)?" + re.escape(target) + "\\s+on\\s+(?:the\\s+)?" + kind,
            phrase.strip(),
            re.I,
        )
    ):
        origin_phrase = observer["origin"]["phrase"]
        complete = (
            "\\s*(?:when\\s+)?"
            + re.escape(origin_phrase)
            + "[\\s,]+"
            + re.escape(facing["phrase"])
            + "[\\s,]+"
            + "(?:(?:it(?:'s|\\s+is)|the\\s+correct\\s+one\\s+is)\\s+)?"
            + re.escape(phrase)
            + "[.!?]*\\s*"
        )
        if re.fullmatch(complete, query, re.I):
            relation.update(anchor_category="", anchor_categories=[], scope="side")
            anchor = ""
    if kind in ("above", "below") and isinstance(anchor, str) and anchor.strip():
        relation["scope"] = "relative_anchor"
    elif kind in ("left", "right", "front", "back") and (not anchor):
        extreme = re.search(
            "\\b(?:far|most|furthest|farthest|leftmost|rightmost)\\b", phrase, re.I
        )
        relative = re.search("\\b(?:of|than)\\b", phrase, re.I)
        side = re.search("\\b(?:on|to|at)\\s+(?:the\\s+)?" + kind + "\\b", phrase, re.I)
        adjective = len(targets) == 1 and re.fullmatch(
            "(?:the\\s+)?" + kind + "\\s+" + re.escape(targets[0]) + "[.!?]?",
            phrase.strip(),
            re.I,
        )
        if not extreme and (not relative) and (side or adjective):
            relation["scope"] = "side"
    return relation


def _condition_records(query, parsed, raw_relations, raw_attributes):
    """Keep quoted conditions and role/evidence state together for audit."""
    targets = parsed.get("Target", [])
    targets = (
        [targets]
        if isinstance(targets, str)
        else targets if isinstance(targets, list) else []
    )
    anchors = parsed.get("Anchor", [])
    anchors = (
        [anchors]
        if isinstance(anchors, str)
        else anchors if isinstance(anchors, list) else []
    )
    target_categories = {
        retrieval_category(name)
        for name in targets
        if isinstance(name, str) and retrieval_category(name)
    }
    reference_categories = {
        retrieval_category(name)
        for name in anchors
        if isinstance(name, str) and retrieval_category(name)
    }
    for relation in parsed.get("Relations", []):
        if not isinstance(relation, dict):
            continue
        for key in ("anchor_category", "anchor_categories"):
            names = relation.get(key, []) or []
            names = [names] if isinstance(names, str) else names
            names = names if isinstance(names, list) else []
            reference_categories.update(
                (
                    retrieval_category(name)
                    for name in names
                    if isinstance(name, str) and retrieval_category(name)
                )
            )
    observer = parsed.get("Observer", {})
    if isinstance(observer, dict):
        for role in ("origin", "facing"):
            entry = observer.get(role, {})
            name = entry.get("category", "") if isinstance(entry, dict) else ""
            category = retrieval_category(name) if isinstance(name, str) else ""
            if category:
                reference_categories.add(category)
    records = []

    def add(
        condition_type,
        phrase,
        target_role=None,
        reference_role=None,
        other_role=None,
        binding_status="bound",
        evidence_status="pending",
        reason=None,
    ):
        quote = phrase if quoted(query, phrase) else None
        records.append(
            {
                "condition_type": condition_type,
                "source_quote": quote,
                "reported_phrase": phrase if isinstance(phrase, str) else "",
                "target_role": target_role,
                "reference_role": reference_role,
                "other_role": other_role,
                "binding_status": binding_status,
                "evidence_status": evidence_status,
                "evidence": [],
                **({"reason": reason} if reason else {}),
            }
        )

    for attribute in raw_attributes:
        if not isinstance(attribute, dict):
            continue
        phrase = attribute.get("phrase", "")
        category_name = attribute.get("category", "")
        category = (
            retrieval_category(category_name) if isinstance(category_name, str) else ""
        )
        target_match = category in target_categories
        reference_match = category in reference_categories
        explicit_role = attribute.get("role")
        target_role = reference_role = None
        binding_status = "bound"
        reason = None
        if explicit_role == "target" and target_match:
            target_role = {"category": category}
        elif explicit_role == "reference" and reference_match:
            reference_role = {"category": category}
        elif explicit_role in ("target", "reference"):
            binding_status = "unbound"
            reason = "explicit attribute role does not match a supplied category"
        elif target_match and reference_match:
            binding_status = "ambiguous"
            reason = "attribute category occurs as both target and reference; role was not supplied"
        elif target_match:
            target_role = {"category": category}
        elif reference_match:
            reference_role = {"category": category}
        else:
            binding_status = "unbound"
            reason = "attribute category is not a supplied target or reference"
        phrase_text = phrase.casefold() if isinstance(phrase, str) else ""
        condition_type = (
            "size"
            if re.search(
                "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|tall|high|short|low)\\b",
                phrase_text,
            )
            else "appearance"
        )
        add(
            condition_type,
            phrase,
            target_role,
            reference_role,
            binding_status=binding_status,
            evidence_status="missing" if binding_status != "bound" else "pending",
            reason=reason,
        )
        if (
            condition_type == "size"
            and parsed.get("query_syntax", {}).get("backend") == "spacy"
        ):
            from eviground.query.query_syntax import comparison_hint

            hints = [
                comparison_hint(parsed, word, category)
                for word in re.findall("\\b\\w+\\b", phrase_text)
            ]
            records[-1]["language_comparisons"] = [hint for hint in hints if hint]
            records[-1]["scene_binding_status"] = "pending"
    scope_match = re.search("\\bof the two(?:\\s+[a-z-]+)?\\b", query, re.I)
    if scope_match and re.search(
        "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|small|large|big|little|tall|short)\\b",
        query,
        re.I,
    ):
        add(
            "size_scope",
            scope_match.group(),
            (
                {"category": next(iter(target_categories))}
                if len(target_categories) == 1
                else None
            ),
            binding_status="pending",
            evidence_status="pending",
            reason="exactly two instances must be bound by measured size evidence",
        )
    for relation in raw_relations:
        if not isinstance(relation, dict):
            add(
                "relation",
                "",
                binding_status="unbound",
                evidence_status="missing",
                reason="relation entry is not an object",
            )
            continue
        kind = relation.get("relation", "other")
        target_name = relation.get("target_category", "")
        target = retrieval_category(target_name) if isinstance(target_name, str) else ""
        reference_names = (
            relation.get("anchor_categories", []) or []
            if kind == "between"
            else [relation.get("anchor_category", "")]
        )
        reference_names = (
            [reference_names]
            if isinstance(reference_names, str)
            else reference_names if isinstance(reference_names, list) else []
        )
        reference_names = [
            retrieval_category(name)
            for name in reference_names
            if isinstance(name, str) and retrieval_category(name)
        ]
        phrase = relation.get("phrase", "")
        valid = relation in parsed.get("Relations", [])
        add(
            "relation:" + str(kind),
            phrase,
            {"category": target} if valid and target else None,
            {"categories": reference_names} if valid and reference_names else None,
            binding_status="bound" if valid else "unbound",
            evidence_status="pending" if valid else "unsupported",
            reason=None if valid else "relation failed structural validation",
        )
    for match in re.finditer(IMMEDIATE_DIRECTION_PATTERN, query, re.I):
        bound = []
        for relation in parsed.get("Relations", []):
            phrase = relation.get("phrase", "")
            if relation.get("relation") != match["direction"].lower() or not quoted(
                query, phrase
            ):
                continue
            occurrences = list(re.finditer(re.escape(phrase), query, re.I))
            if len(occurrences) == 1 and (
                occurrences[0].start() < match.end()
                and occurrences[0].end() > match.start()
            ):
                bound.append(relation)
        relation = bound[0] if len(bound) == 1 else {}
        target = retrieval_category(relation.get("target_category", ""))
        reference = retrieval_category(relation.get("anchor_category", ""))
        add(
            "relation:immediate_proximity",
            match.group(),
            target_role={"category": target} if target else None,
            reference_role={"categories": [reference]} if reference else None,
            binding_status="bound" if target and reference else "unbound",
            evidence_status="missing",
            reason="immediate proximity has no calibrated predicate: " + match.group(),
        )
    for coreference in parsed.get("target_coreferences", []):
        if not isinstance(coreference, dict):
            continue
        category = retrieval_category(coreference.get("category", ""))
        phrase = coreference.get("phrase", "")
        if category and quoted(query, phrase):
            add(
                "target_coreference",
                phrase,
                target_role={"category": category},
                evidence_status="pending",
                reason="bound to the unique target; its represented relation supplies the evidence",
            )
    if isinstance(observer, dict):
        for role in ("origin", "facing"):
            entry = observer.get(role, {})
            if isinstance(entry, dict) and entry.get("phrase"):
                category = retrieval_category(entry.get("category", ""))
                add(
                    "viewpoint:" + role,
                    entry.get("phrase"),
                    other_role={"role": role, "category": category or None},
                    binding_status=(
                        "bound"
                        if category or entry.get("kind") == "room_center"
                        else "unbound"
                    ),
                    evidence_status=(
                        "pending"
                        if category or entry.get("kind") == "room_center"
                        else "missing"
                    ),
                )
                if role == "facing" and entry.get("reference_frame") == "intrinsic":
                    add(
                        "viewpoint:intrinsic_orientation",
                        entry["phrase"],
                        other_role={
                            "role": "facing_orientation",
                            "category": category,
                            "part": entry.get("orientation_part"),
                        },
                        binding_status="bound",
                        evidence_status="missing",
                        reason="object-intrinsic facing orientation is unavailable: "
                        + entry["phrase"],
                    )
    for phrase in parsed.get("Unhandled", []):
        add(
            "unhandled",
            phrase,
            binding_status="unbound",
            evidence_status="unsupported",
            reason="parser did not map this condition to a supported evidence field",
        )
    for match in re.finditer(
        "\\b(no|not|nothing|never|without|except)\\b|n['’]t\\b", query, re.I
    ):
        add(
            "negation",
            match.group(),
            binding_status="unbound",
            evidence_status="unsupported",
            reason="negation/exclusion predicate is not implemented",
        )
    for diagnostic in parsed.get("diagnostic_missing", []):
        if not isinstance(diagnostic, str):
            continue
        phrase = None
        condition_type = "coverage_gap"
        binding_status = "unbound"
        if diagnostic.startswith("appearance condition role not represented: "):
            gap = re.fullmatch(
                "appearance condition role not represented: (.+) \\((target|reference) (.+)\\)",
                diagnostic,
            )
            if gap:
                phrase, role, category = gap.groups()
                add(
                    "appearance",
                    phrase,
                    target_role={"category": category} if role == "target" else None,
                    reference_role=(
                        {"category": category} if role == "reference" else None
                    ),
                    binding_status="unbound",
                    evidence_status="missing",
                    reason=diagnostic,
                )
                continue
        elif diagnostic.startswith("appearance condition not represented: "):
            word = diagnostic.split(": ", 1)[1]
            found = re.search("\\b" + re.escape(word) + "\\b", query, re.I)
            phrase = found.group() if found else None
            condition_type = (
                "size"
                if re.search(
                    "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|tall|high|short|low)\\b",
                    word,
                    re.I,
                )
                else "appearance"
            )
        elif diagnostic.startswith("unsupported relation: "):
            phrase = diagnostic[len("unsupported relation: ") :]
            condition_type = "relation"
        elif diagnostic.startswith("unhandled condition: "):
            phrase = diagnostic[len("unhandled condition: ") :]
            condition_type = "unhandled"
        elif diagnostic == "vertical condition not represented":
            found = re.search(
                "\\b(above|over|below|under|underneath|beneath)\\b", query, re.I
            )
            phrase = found.group() if found else None
            condition_type = "relation:vertical"
        elif diagnostic.startswith("size comparison scope not bound: "):
            phrase = diagnostic[len("size comparison scope not bound: ") :]
            condition_type = "size_scope"
        elif diagnostic.startswith(
            "object-intrinsic facing orientation is unavailable: "
        ):
            phrase = diagnostic[
                len("object-intrinsic facing orientation is unavailable: ") :
            ]
            condition_type = "viewpoint:intrinsic_orientation"
        elif diagnostic.startswith("attribute target/reference role is ambiguous: "):
            phrase = diagnostic[len("attribute target/reference role is ambiguous: ") :]
            binding_status = "ambiguous"
            condition_type = (
                "size"
                if re.search(
                    "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|tall|high|short|low)\\b",
                    phrase,
                    re.I,
                )
                else "appearance"
            )
            record = next(
                (
                    item
                    for item in records
                    if item["condition_type"] == condition_type
                    and item["source_quote"]
                    and (item["source_quote"].casefold() == phrase.casefold())
                ),
                None,
            )
            if record:
                record["binding_status"] = "ambiguous"
                record["evidence_status"] = "missing"
                record["reason"] = diagnostic
                continue
        elif diagnostic == "observer origin or facing direction is unspecified":
            found = re.search("\\b(left|right|ahead|behind|front|back)\\b", query, re.I)
            phrase = found.group() if found else None
            condition_type = "viewpoint"
        elif diagnostic == "direction condition not represented":
            found = re.search("\\b(left|right|ahead|behind)\\b", query, re.I)
            phrase = found.group() if found else None
            condition_type = "relation:direction"
        elif diagnostic.startswith("immediate proximity has no calibrated predicate: "):
            continue
        if phrase and any(
            (
                record["source_quote"]
                and record["source_quote"].casefold() == phrase.casefold()
                and (record["evidence_status"] in ("missing", "unsupported"))
                for record in records
            )
        ):
            continue
        add(
            condition_type,
            phrase or "",
            binding_status=binding_status,
            evidence_status="missing",
            reason=diagnostic,
        )
    return records


def validate_structure(query, parsed):
    parsed = dict(parsed)
    from eviground.query.query_syntax import analyze_query

    parsed["query_syntax"] = analyze_query(query)
    parsed.pop("_size_scope_bindings", None)
    parsed.pop("_verified_reference_bindings", None)
    parsed["source_query"] = query
    from eviground.query.noun_binding import recover_part_category

    recover_part_category(query, parsed)
    input_relations = parsed.get("Relations", [])
    preflight_missing = []
    malformed_relations = []
    if not isinstance(input_relations, list):
        preflight_missing.append("relation coverage not supplied by parser")
        malformed_relations = [input_relations]
        parsed["Relations"] = []
    else:
        malformed_relations = [
            relation for relation in input_relations if not isinstance(relation, dict)
        ]
        if malformed_relations:
            preflight_missing.append("parser produced a non-object relation entry")
        parsed["Relations"] = [
            relation for relation in input_relations if isinstance(relation, dict)
        ]
    parsed = recover_between_relations(query, recover_vertical_relations(query, parsed))
    observer_input = parsed.get("Observer", {})
    room_center_origin = _explicit_room_center_origin(query, observer_input)
    if room_center_origin:
        anchors = parsed.get("Anchor", [])
        if isinstance(anchors, list):
            parsed["Anchor"] = [
                name for name in anchors if not _room_center_anchor(name)
            ]
        elif _room_center_anchor(anchors):
            parsed["Anchor"] = []
        relations = []
        for relation in parsed["Relations"]:
            if isinstance(relation, dict):
                relation = dict(relation)
                if relation.get("relation") in (
                    "left",
                    "right",
                    "front",
                    "back",
                ) and _room_center_anchor(relation.get("anchor_category")):
                    relation["anchor_category"] = ""
                    relation["scope"] = "side"
            relations.append(relation)
        parsed["Relations"] = relations
    targets = parsed.get("Target", [])
    targets = [targets] if isinstance(targets, str) else targets
    if isinstance(parsed.get("Attributes"), list):
        parsed["Attributes"] = [
            (
                {
                    key: value
                    for key, value in attribute.items()
                    if key
                    not in (
                        "geometry_attribute",
                        "comparison_scope",
                        "scope_phrase",
                        "required_member_count",
                        "size_semantics",
                    )
                }
                if isinstance(attribute, dict)
                else attribute
            )
            for attribute in parsed["Attributes"]
        ]
    comparison = full_category_size_comparison(query, parsed)
    if comparison and isinstance(parsed.get("Unhandled"), list):
        parsed["Anchor"] = []
        parsed["Attributes"] = [comparison]
        covered = []
        kept = []
        for phrase in parsed["Unhandled"]:
            covered_phrase = (
                quoted(query, phrase)
                if parsed["query_syntax"]["backend"] == "spacy"
                else isinstance(phrase, str)
                and phrase.strip().rstrip(".!?").casefold()
                == query.strip().rstrip(".!?").casefold()
            )
            if covered_phrase:
                covered.append(
                    {
                        "source_quote": phrase,
                        "covered_by": "complete_size_comparison",
                        "source_phrase": query,
                        "scene_binding_status": "pending",
                    }
                )
            else:
                kept.append(phrase)
        parsed["Unhandled"] = kept
        parsed["condition_coverage"] = covered
    if parsed["query_syntax"]["backend"] == "spacy" and isinstance(
        parsed.get("Unhandled"), list
    ):
        kept = []
        for phrase in parsed["Unhandled"]:
            covered = next(
                (
                    hint
                    for hint in parsed["query_syntax"]["comparisons"]
                    if hint["scope_type"] == "explicit_set"
                    and hint["required_member_count"] == 2
                    and isinstance(phrase, str)
                    and quoted(query, phrase)
                    and re.fullmatch(
                        "of\\s+the\\s+two(?:\\s+"
                        + category_pattern(hint["head_category"])
                        + ")?",
                        phrase.strip().rstrip(".!?"),
                        re.I,
                    )
                    and any(
                        (
                            isinstance(attr, dict)
                            and attr.get("role") == "target"
                            and (
                                retrieval_category(attr.get("category", ""))
                                == retrieval_category(hint["head_category"])
                            )
                            and (
                                hint["word"]
                                in re.findall(
                                    "\\b\\w+\\b", attr.get("phrase", "").casefold()
                                )
                            )
                            for attr in parsed.get("Attributes", [])
                        )
                    )
                ),
                None,
            )
            if covered:
                parsed.setdefault("condition_coverage", []).append(
                    {
                        "source_quote": phrase,
                        "covered_by": "explicit_comparison_scope",
                        "source_phrase": covered["source_phrase"],
                        "scene_binding_status": "pending",
                    }
                )
            else:
                kept.append(phrase)
        parsed["Unhandled"] = kept
    if len(targets) == 1:
        match = re.fullmatch(
            "\\s*(?:(?:choose|select|find|pick|click)\\s+|it\\s+is\\s+)?(?:the\\s+)?(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|small|large|big|little|tall|short)\\s+"
            + re.escape(targets[0])
            + "\\s*[.!?]?\\s*",
            query,
            re.I,
        )
        if (
            match
            and (not parsed.get("Anchor"))
            and (not parsed.get("Relations"))
            and isinstance(parsed.get("Unhandled"), list)
            and all((isinstance(p, str) for p in parsed["Unhandled"]))
        ):
            parsed["Attributes"] = [{"category": targets[0], "phrase": match.group(1)}]
            parsed["Unhandled"] = [
                phrase
                for phrase in parsed.get("Unhandled", [])
                if phrase.strip().rstrip(".!?").casefold()
                not in (
                    query.strip().rstrip(".!?").casefold(),
                    match.group(1).casefold(),
                )
            ]
    missing = list(preflight_missing)
    relations = parsed.get("Relations", [])
    if not isinstance(relations, list):
        missing.append("relation coverage not supplied by parser")
        relations = []
    relations = [
        (
            _normalize_relation_scope(query, relation, targets, observer_input)
            if isinstance(relation, dict)
            else relation
        )
        for relation in relations
    ]
    raw_relations = malformed_relations + list(relations)
    valid = []
    for relation in relations:
        if not isinstance(relation, dict):
            missing.append("parser produced a non-object relation entry")
            continue
        kind = relation.get("relation")
        phrase = relation.get("phrase")
        valid_quote = quoted(query, phrase)
        target_name = relation.get("target_category")
        anchor_name = relation.get("anchor_category", "")
        anchor_names = (
            relation.get("anchor_categories", [])
            if kind == "between"
            else [anchor_name]
        )
        valid_quote = (
            valid_quote
            and isinstance(kind, str)
            and isinstance(target_name, str)
            and bool(target_name.strip())
        )
        if anchor_name is not None and (not isinstance(anchor_name, str)):
            valid_quote = False
        if relation.get("anchor_categories") is not None and (
            not isinstance(relation.get("anchor_categories"), list)
        ):
            valid_quote = False
        if kind == "between":
            valid_quote = (
                valid_quote
                and isinstance(anchor_names, list)
                and (len(anchor_names) == 2)
                and all(
                    (
                        isinstance(name, str) and bool(name.strip())
                        for name in anchor_names
                    )
                )
                and bool(re.search("\\bbetween\\b", phrase, re.I))
            )
        elif (
            kind in ("nearest", "farthest", "near", "above", "below")
            or relation.get("scope") == "relative_anchor"
        ):
            valid_quote = (
                valid_quote
                and isinstance(anchor_name, str)
                and bool(anchor_name.strip())
            )
        if kind in ("left", "right", "front", "back"):
            words = {
                "left": "\\bleft\\b",
                "right": "\\bright\\b",
                "front": "\\b(front|ahead)\\b",
                "back": "\\b(back|behind)\\b",
            }
            valid_quote = valid_quote and bool(re.search(words[kind], phrase, re.I))
            valid_quote = valid_quote and relation.get("scope") in (
                "relative_anchor",
                "side",
            )
            valid_quote = valid_quote and (
                not re.search(
                    "\\b(?:far|most|leftmost|rightmost|furthest|farthest)\\b",
                    phrase or "",
                    re.I,
                )
            )
            if relation.get("reference_frame", "observer") != "observer":
                missing.append(
                    "object-intrinsic directional frame is unavailable: " + phrase
                    if quoted(query, phrase)
                    else "unsupported directional reference frame"
                )
                valid_quote = False
        if kind in ("above", "below"):
            valid_quote = valid_quote and bool(
                re.search(
                    "\\b(above|over|below|under|underneath|beneath)\\b", phrase, re.I
                )
            )
        if not valid_quote or kind not in (
            "nearest",
            "farthest",
            "near",
            "left",
            "right",
            "front",
            "back",
            "above",
            "below",
            "between",
        ):
            missing.append(
                "unsupported relation: " + phrase
                if quoted(query, phrase)
                else "parser produced an unquoted relation; not a query condition"
            )
        else:
            valid.append(relation)
    parsed["Relations"] = valid
    observer = parsed.get("Observer", {})
    if not isinstance(observer, dict):
        observer = {}
    origin, facing = (observer.get("origin", {}), observer.get("facing", {}))
    if not isinstance(origin, dict):
        origin = {}
    if not isinstance(facing, dict):
        facing = {}
    if origin.get("kind") == "room_center":
        phrase = origin.get("phrase", "")
        if (
            not quoted(query, phrase)
            or not re.search("\\b(middle|cent(?:er|re))\\b", phrase, re.I)
            or (not re.search("\\broom\\b", phrase, re.I))
        ):
            missing.append("room-center origin has no supporting quote")
            origin = {}
    elif origin.get("kind") == "category":
        if not quoted(query, origin.get("phrase")) or not isinstance(
            origin.get("category"), str
        ):
            missing.append("named origin has no supporting quote/category")
            origin = {}
    else:
        origin = {}
    intrinsic_facing = intrinsic_facing_condition(query, facing.get("category", ""))
    if intrinsic_facing:
        facing = {
            **facing,
            "phrase": intrinsic_facing["phrase"],
            "group": intrinsic_facing["group"],
            "reference_frame": "intrinsic",
            "orientation_part": intrinsic_facing["part"],
        }
    if facing.get("category"):
        phrase = facing.get("phrase", "")
        if (
            not quoted(query, phrase)
            or not re.search("\\b(fac(?:e|ing)|look(?:ing)?)\\b", phrase, re.I)
            or (not isinstance(facing.get("category"), str))
            or (type(facing.get("group", False)) is not bool)
        ):
            missing.append("facing direction has no supporting quote/category")
            facing = {}
        elif facing.get("group") and (
            not re.search(
                "\\b" + re.escape(retrieval_category(facing["category"])) + "s\\b",
                phrase,
                re.I,
            )
        ):
            if not re.search("\\b(group|all|both)\\b", phrase, re.I):
                missing.append("group-facing has no explicit plural/group quote")
                facing = {}
        elif re.fullmatch(
            "(?:facing|looking\\s+(?:at|toward))\\s+(?:the\\s+)?"
            + re.escape(retrieval_category(facing["category"]))
            + "s",
            phrase.strip(),
            re.I,
        ):
            facing = {**facing, "group": True}
        if intrinsic_facing:
            missing.append(
                "object-intrinsic facing orientation is unavailable: "
                + facing["phrase"]
            )
    parsed["Observer"] = {"origin": origin, "facing": facing}
    targets = parsed.get("Target", [])
    targets = [targets] if isinstance(targets, str) else targets
    recovered_observer_coreference = False
    if (
        not targets
        and _explicit_room_center_origin(query, parsed["Observer"])
        and facing.get("group")
        and re.search("\\bthe correct one\\b", query, re.I)
    ):
        matching_relations = [
            relation
            for relation in parsed["Relations"]
            if relation.get("relation") in ("left", "right")
            and relation.get("scope") == "side"
            and (
                retrieval_category(relation.get("target_category", ""))
                == retrieval_category(facing.get("category", ""))
            )
            and _complete_observer_coreference_clause(
                query, parsed["Observer"], relation
            )
        ]
        if len(matching_relations) == 1:
            target_category = matching_relations[0]["target_category"]
            targets = [target_category]
            parsed["Target"] = targets
            parsed["target_binding_recovery"] = {
                "phrase": "the correct one",
                "category": target_category,
                "basis": "explicit room-center frame and matching facing-group directional relation",
            }
            recovered_observer_coreference = True
    target_coreferences = []
    if (
        len(targets) == 1
        and retrieval_category(targets[0])
        not in ("one", "correct one", "it", "object", "thing")
        and origin
        and facing
    ):
        coreference_phrases = list(parsed.get("Unhandled", []))
        if recovered_observer_coreference:
            coreference_phrases.append("the correct one")
        for phrase in coreference_phrases:
            if (
                not isinstance(phrase, str)
                or phrase.strip().rstrip(".!?").casefold() != "the correct one"
            ):
                continue
            coref_quote = phrase.strip().rstrip(".!?")
            coref_start = query.casefold().find(coref_quote.casefold())

            def relation_follows_coreference(relation):
                relation_phrase = relation.get("phrase", "")
                relation_start = query.casefold().find(relation_phrase.casefold())
                if relation_start > coref_start:
                    return True
                relation_end = relation_start + len(relation_phrase)
                return relation_start <= coref_start < relation_end and bool(
                    re.search(
                        "\\b" + re.escape(relation["relation"]) + "\\b",
                        query[coref_start + len(coref_quote) : relation_end],
                        re.I,
                    )
                )

            represented_relation = next(
                (
                    relation
                    for relation in parsed["Relations"]
                    if relation.get("relation") in ("left", "right", "front", "back")
                    and retrieval_category(relation.get("target_category", ""))
                    == retrieval_category(targets[0])
                    and quoted(query, relation.get("phrase"))
                    and relation_follows_coreference(relation)
                ),
                None,
            )
            if represented_relation:
                target_coreferences.append({"phrase": phrase, "category": targets[0]})
    parsed["target_coreferences"] = target_coreferences
    attrs = parsed.get("Attributes")
    if not isinstance(attrs, list) or any((not isinstance(a, dict) for a in attrs)):
        missing.append("attribute coverage not supplied by parser")
        attrs = []
    attrs = [dict(attribute) for attribute in attrs]
    attrs, recovered_attribute_missing = _recover_surface_attributes(
        query, parsed, attrs
    )
    contrast = relative_height_between(query, targets[0]) if len(targets) == 1 else None
    if contrast:
        for role, word in (
            ("target", contrast["target_word"]),
            ("reference", contrast["reference_word"]),
        ):
            if not any(
                (a.get("phrase") == word and a.get("role") == role for a in attrs)
            ):
                attrs.append(
                    {
                        "category": targets[0],
                        "phrase": word,
                        "role": role,
                        "binding_source": "quoted contrast around the two between references",
                    }
                )
        parsed["relative_height_binding"] = contrast
    missing.extend(recovered_attribute_missing)
    raw_attrs = list(attrs)
    bound_categories = set()
    for key in ("Target", "Anchor"):
        names = parsed.get(key, [])
        names = names if isinstance(names, list) else [names]
        bound_categories.update(
            (
                retrieval_category(name)
                for name in names
                if isinstance(name, str) and retrieval_category(name)
            )
        )
    for relation in parsed["Relations"]:
        for key in ("target_category", "anchor_category"):
            name = relation.get(key, "")
            if isinstance(name, str) and retrieval_category(name):
                bound_categories.add(retrieval_category(name))
        relation_anchor_categories = relation.get("anchor_categories", [])
        if not isinstance(relation_anchor_categories, list):
            relation_anchor_categories = []
        for name in relation_anchor_categories:
            if isinstance(name, str) and retrieval_category(name):
                bound_categories.add(retrieval_category(name))
    observer = parsed["Observer"]
    for role in ("origin", "facing"):
        entry = observer.get(role, {})
        name = entry.get("category", "") if isinstance(entry, dict) else ""
        if isinstance(name, str) and retrieval_category(name):
            bound_categories.add(retrieval_category(name))
    valid_attrs = []
    for attribute in attrs:
        source_bindings = _source_attribute_bindings(
            query, parsed, attribute.get("phrase", "")
        )
        source_binding = (
            next(iter(source_bindings)) if len(source_bindings) == 1 else None
        )
        if (
            source_binding
            and retrieval_category(attribute.get("category", "")) in bound_categories
        ):
            source_category, source_role = source_binding
            if (
                retrieval_category(attribute.get("category", "")) != source_category
                or (attribute.get("role") and attribute.get("role") != source_role)
                or attribute_role(parsed, attribute) is None
            ):
                attribute["reported_category"] = attribute.get("category")
                attribute["reported_role"] = attribute.get("role")
                attribute.update(
                    category=source_category,
                    role=source_role,
                    binding_source="direct quoted noun attachment",
                )
        category_name = attribute.get("category")
        category = (
            retrieval_category(category_name) if isinstance(category_name, str) else ""
        )
        target_categories = parsed.get("Target", [])
        target_categories = (
            [target_categories]
            if isinstance(target_categories, str)
            else target_categories
        )
        reference_categories = parsed.get("Anchor", [])
        reference_categories = (
            [reference_categories]
            if isinstance(reference_categories, str)
            else reference_categories
        )
        target_categories = {
            retrieval_category(name)
            for name in target_categories
            if retrieval_category(name)
        }
        reference_categories = {
            retrieval_category(name)
            for name in reference_categories
            if retrieval_category(name)
        }
        for relation in parsed["Relations"]:
            for key in ("anchor_category", "anchor_categories"):
                names = relation.get(key, []) or []
                names = [names] if isinstance(names, str) else names
                names = names if isinstance(names, list) else []
                reference_categories.update(
                    (
                        retrieval_category(name)
                        for name in names
                        if isinstance(name, str) and retrieval_category(name)
                    )
                )
        for role in ("origin", "facing"):
            entry = parsed["Observer"].get(role, {})
            name = entry.get("category", "") if isinstance(entry, dict) else ""
            if isinstance(name, str) and retrieval_category(name):
                reference_categories.add(retrieval_category(name))
        explicit_role = attribute.get("role")
        role_matches = (
            explicit_role == "target"
            and category in target_categories
            or (explicit_role == "reference" and category in reference_categories)
        )
        inferred_role_matches = (
            not explicit_role
            and category in target_categories
            or (not explicit_role and category in reference_categories)
        )
        role_ambiguous = (
            not explicit_role
            and category in target_categories
            and (category in reference_categories)
        )
        if (
            size_specs(attribute.get("phrase", ""), category)
            and category in target_categories
            and (category in reference_categories)
            and (len(source_bindings) != 1)
            and (
                len(
                    list(
                        re.finditer(re.escape(attribute.get("phrase", "")), query, re.I)
                    )
                )
                == 1
            )
        ):
            role_ambiguous = True
        if (
            quoted(query, attribute.get("phrase"))
            and category in bound_categories
            and (
                role_matches
                and (not role_ambiguous)
                or (inferred_role_matches and (not role_ambiguous))
            )
        ):
            if not any(
                (
                    retrieval_category(item.get("category", "")) == category
                    and attribute_role(parsed, item)
                    == attribute_role(parsed, attribute)
                    and (
                        item.get("phrase", "").casefold()
                        == attribute.get("phrase", "").casefold()
                    )
                    for item in valid_attrs
                )
            ):
                valid_attrs.append(attribute)
        elif quoted(query, attribute.get("phrase")) and role_ambiguous:
            missing.append(
                "attribute target/reference role is ambiguous: "
                + attribute.get("phrase", "")
            )
        elif quoted(query, attribute.get("phrase")):
            missing.append(
                "attribute category is not a supplied target/reference: "
                + str(category_name)
            )
            missing.append(
                "appearance condition not represented: " + attribute.get("phrase", "")
            )
    parsed["Attributes"] = valid_attrs
    if len(parsed["Attributes"]) != len(attrs):
        missing.append("unquoted or unbound appearance attribute")
    for phrase in dict.fromkeys((attribute.get("phrase", "") for attribute in attrs)):
        phrase_bindings = _source_attribute_bindings(query, parsed, phrase)
        for unit in attribute_units(phrase):
            bindings = phrase_bindings | _source_attribute_bindings(query, parsed, unit)
            for category, role in sorted(bindings):
                if not any(
                    (
                        unit.casefold()
                        in {
                            part.casefold()
                            for part in attribute_units(attribute.get("phrase", ""))
                        }
                        and retrieval_category(attribute.get("category", ""))
                        == category
                        and (attribute_role(parsed, attribute) == role)
                        for attribute in valid_attrs
                    )
                ):
                    missing.append(
                        "appearance condition role not represented: "
                        + unit
                        + " ("
                        + role
                        + " "
                        + category
                        + ")"
                    )
    covered = " ".join((a["phrase"] for a in parsed["Attributes"]))
    for word in re.findall(
        "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter|longer|large|small|big|little|skinny|narrow|wide|thin|thick|dark|light|black|white|red|blue|green|gr[ae]y|brown|yellow|orange|purple|pink|rainbow|wooden|mounted|folded|vertical)\\b",
        query,
        re.I,
    ):
        if not re.search("\\b" + re.escape(word) + "\\b", covered, re.I):
            missing.append("appearance condition not represented: " + word)
    unhandled = parsed.get("Unhandled")
    if not isinstance(unhandled, list) or any(
        (not isinstance(x, str) for x in unhandled)
    ):
        missing.append("condition coverage not supplied by parser")
        parsed["Unhandled"] = []
    else:
        parsed["Unhandled"] = [p for p in unhandled if quoted(query, p)]
        if len(parsed["Unhandled"]) != len(unhandled):
            missing.append(
                "parser produced unquoted Unhandled entries; these are not query conditions"
            )
        bound_coreference_quotes = {
            item["phrase"].strip().rstrip(".!?").casefold()
            for item in target_coreferences
        }
        parsed["Unhandled"] = [
            p
            for p in parsed["Unhandled"]
            if p.strip().rstrip(".!?").casefold() not in bound_coreference_quotes
        ]
        if recovered_observer_coreference:
            parsed["Unhandled"] = [
                p
                for p in parsed["Unhandled"]
                if p.strip().casefold() != query.strip().casefold()
            ]
        if covered_attribute_request(query, parsed):
            parsed["Unhandled"] = [
                p
                for p in parsed["Unhandled"]
                if p.strip().rstrip(".!?").casefold()
                != query.strip().rstrip(".!?").casefold()
            ]
        missing.extend(("unhandled condition: " + p for p in parsed["Unhandled"]))
    targets = (
        parsed["Target"] if isinstance(parsed["Target"], list) else [parsed["Target"]]
    )
    if any(
        (
            retrieval_category(name) in ("one", "correct one", "it", "object", "thing")
            for name in targets
        )
    ):
        missing.append("target category is an unresolved pronoun or generic noun")
    if re.search("\\b(no|not|nothing|never|without|except)\\b|n['’]t\\b", query, re.I):
        missing.append("negation/exclusion is unsupported")
        from eviground.query.query_graph import retain_positive_predicates

        retain_positive_predicates(query, parsed)
    if any(
        (
            r["relation"] in ("left", "right", "front", "back")
            for r in parsed["Relations"]
        )
    ):
        observer = parsed["Observer"]
        if not observer["origin"] or not observer["facing"]:
            missing.append("observer origin or facing direction is unspecified")
    if re.search(
        "\\b(smallest|largest|biggest|tallest|shortest|highest|lowest|larger|smaller|bigger|taller|shorter)\\b",
        query,
        re.I,
    ):
        scope_match = re.search("\\bof the two(?:\\s+[a-z-]+)?\\b", query, re.I)
        scope_specs = (
            size_conditions(parsed, targets[0])
            if len(targets) == 1 and parsed["query_syntax"]["backend"] == "spacy"
            else [
                spec
                for attr in parsed.get("Attributes", [])
                for spec in size_specs(
                    attr.get("phrase", ""),
                    targets[0] if len(targets) == 1 else "",
                    query=query,
                )
            ]
        )
        if scope_match and (
            not any((spec.get("required_member_count") == 2 for spec in scope_specs))
        ):
            missing.append("size comparison scope not bound: " + scope_match.group())
    direction_query = re.sub(
        "\\bright\\s+(?:above|over|below|under|underneath|beneath)\\b",
        "",
        query,
        flags=re.I,
    )
    if re.search("\\b(left|right|ahead|behind)\\b", direction_query, re.I) and (
        not any(
            (
                r["relation"] in ("left", "right", "front", "back")
                for r in parsed["Relations"]
            )
        )
    ):
        missing.append("direction condition not represented")
    for pattern, kinds, label in (
        (
            "\\b(nearest|closest|farthest|furthest)\\b",
            ("nearest", "farthest"),
            "distance extreme",
        ),
        ("\\b(next to|beside|near)\\b", ("near",), "proximity"),
    ):
        if re.search(pattern, query, re.I) and (
            not any((r["relation"] in kinds for r in parsed["Relations"]))
        ):
            missing.append(label + " condition not represented")
    for match in re.finditer(IMMEDIATE_DIRECTION_PATTERN, query, re.I):
        missing.append(
            "immediate proximity has no calibrated predicate: " + match.group()
        )
    if re.search("\\b(corner|leftmost|rightmost|second|third)\\b", query, re.I):
        missing.append(
            "corner/ordinal/support/extreme conditions require additional predicates"
        )
    if re.search("\\b(above|over|below|under|underneath|beneath)\\b", query, re.I) and (
        not any((r["relation"] in ("above", "below") for r in parsed["Relations"]))
    ):
        missing.append("vertical condition not represented")
    for match in re.finditer(CONTAINMENT_PATTERN, query, re.I):
        missing.append("unsupported containment condition: " + match.group().rstrip())
    covered_nouns = {
        a["phrase"].casefold() + " " + a["category"].casefold()
        for a in parsed["Attributes"]
    }
    missing = [
        item
        for item in missing
        if not (
            item.startswith("unsupported relation: ")
            and item[len("unsupported relation: ") :].casefold() in covered_nouns
        )
    ]
    parsed["diagnostic_missing"] = missing
    from eviground.query.query_graph import retain_positive_predicates

    retain_positive_predicates(query, parsed)
    parsed["condition_records"] = _condition_records(
        query, parsed, raw_relations, raw_attrs
    )
    from eviground.query.query_graph import structure

    previous = parsed.get("query_structure", {}).get("conditions", [])
    parsed["query_structure"] = structure(query, parsed)
    present = {
        (c.get("condition_type"), c.get("source_quote"))
        for c in parsed["query_structure"]["conditions"]
    }
    parsed["query_structure"]["conditions"] += [
        c
        for c in previous
        if c.get("polarity") != "positive"
        and (c.get("condition_type"), c.get("source_quote")) not in present
    ]
    for index, condition in enumerate(parsed["query_structure"]["conditions"]):
        condition["condition_id"] = f"c{index}"
    parsed["diagnostic_missing"].extend(
        (
            "query modifier not represented: " + quote
            for quote in parsed["query_structure"]["uncovered_modifiers"]
        )
    )
    return parsed


def refresh_condition_records(query, parsed):
    """Rebuild the audit ledger after another parser operation updates diagnostics."""
    parsed["condition_records"] = _condition_records(
        query, parsed, parsed.get("Relations") or [], parsed.get("Attributes") or []
    )
    from eviground.query.query_graph import structure

    previous = parsed.get("query_structure", {}).get("conditions", [])
    parsed["query_structure"] = structure(query, parsed)
    present = {
        (c.get("condition_type"), c.get("source_quote"))
        for c in parsed["query_structure"]["conditions"]
    }
    parsed["query_structure"]["conditions"] += [
        c
        for c in previous
        if (c.get("condition_type"), c.get("source_quote")) not in present
    ]
    for index, condition in enumerate(parsed["query_structure"]["conditions"]):
        condition["condition_id"] = f"c{index}"
    return parsed


def size_conditions(parsed, category, role="target"):
    conditions = []
    query = parsed.get("source_query", parsed.get("query_rewrite", {}).get("query", ""))
    for attribute in parsed.get("Attributes", []):
        if (
            retrieval_category(attribute.get("category", ""))
            != retrieval_category(category)
            or attribute_role(parsed, attribute) != role
        ):
            continue
        if attribute.get("polarity") == "exclude":
            continue
        for spec in size_specs(
            attribute.get("phrase", ""),
            category,
            query=query,
            role=role,
            source=attribute,
        ):
            from eviground.query.query_syntax import comparison_hint

            hint = comparison_hint(parsed, spec["word"], category)
            if hint:
                spec["syntax_attachment"] = hint["attachment"]
            if hint:
                spec.update(
                    source_phrase=hint["source_phrase"],
                    source_span=hint["source_span"],
                    syntax_scope_type=hint["scope_type"],
                )
                if hint.get("required_member_count"):
                    spec["required_member_count"] = hint["required_member_count"]
            if attribute.get("source_phrase") == query.strip() and attribute.get(
                "comparison_scope"
            ):
                spec.update(
                    scope_kind=attribute["comparison_scope"],
                    scope_phrase=attribute["scope_phrase"],
                )
                if attribute.get("required_member_count"):
                    spec["required_member_count"] = attribute["required_member_count"]
            if (
                spec.get("scope_kind") == "full_category"
                and spec["form"] != "superlative"
                and (not (hint and hint["attachment"] == "independent"))
            ):
                noun = re.escape(retrieval_category(category))
                dependency = re.search(
                    "\\b"
                    + re.escape(spec["word"])
                    + "\\s+(?:[a-z-]+\\s+)*?"
                    + noun
                    + "s?\\s+(?P<scope>(?:that|which|on|in|at|near|beside|between|behind|above|below|across|next\\s+to|with)\\b[^.!?;,]*)",
                    query,
                    re.I,
                )
                if dependency:
                    spec.update(
                        scope_kind="unbound_explicit_scope",
                        scope_phrase=dependency["scope"].strip(),
                        scope_binding_status="unknown",
                        scope_binding_reason=(
                            "comparison modifier attachment is ambiguous"
                            if hint
                            else "comparison scope depends on an unbound quoted relation"
                        ),
                    )
            if hint and hint["attachment"] == "restrictive_subset":
                spec.update(
                    scope_kind="relation_subset",
                    scope_relation_quotes=[m["text"] for m in hint["modifiers"]],
                    scope_phrase=hint["scope_phrase"],
                )
                if hint.get("required_member_count"):
                    spec["required_member_count"] = hint["required_member_count"]
            if (
                hint
                and hint["attachment"] == "ambiguous"
                and hint.get("modifiers")
                and (not attribute.get("scene_scope_assumption"))
                and (
                    not re.search(
                        "\\band\\s+(?:above|below|near|behind|in front)\\b", query, re.I
                    )
                )
            ):
                spec.update(
                    scope_kind="unbound_explicit_scope",
                    scope_relation_quotes=[m["text"] for m in hint["modifiers"]],
                    scope_binding_reason="independent condition and comparison-subset attachments remain ambiguous",
                )
                spec["scope_hypotheses"] = [
                    {
                        "kind": "full_category",
                        "interpretation": "size with independent quoted relation",
                    },
                    {
                        "kind": "relation_subset",
                        "interpretation": "size within quoted relation members",
                    },
                ]
            if spec["metric"] == "volume_m3":
                if retrieval_category(category) in ("table", "desk"):
                    spec.update(
                        metric="oriented_footprint_m2",
                        metric_reason="table/desk horizontal enclosing area proxy; not exact tabletop area",
                        tentative_metric=True,
                        metric_choice_source="category surface structure; declared measurement policy",
                    )
                else:
                    spec.update(
                        metric="oriented_volume_m3",
                        metric_reason="observed oriented volume proxy; completeness and alternative rankings retained",
                    )
            if spec["metric"] == "height_m" and retrieval_category(category) in (
                "desk",
                "table",
            ):
                spec.update(
                    metric="tabletop_height_m",
                    metric_reason="fitted horizontal tabletop above a supplied local floor; total span is not tabletop height",
                )
            if retrieval_category(category) in (
                "mirror",
                "whiteboard",
                "book",
            ) and spec["metric"] in ("volume_m3", "oriented_volume_m3"):
                spec.update(
                    metric="body_plane_area_m2",
                    tentative_metric=True,
                    metric_reason="observed dominant body-plane area; overall-size semantics remain tentative",
                    metric_choice_source="measured planar structure",
                )
            choice = parsed.get("_metric_choices", {}).get(
                "|".join((role, retrieval_category(category), spec["word"]))
            )
            if choice and spec["metric"] in (
                "oriented_volume_m3",
                "oriented_footprint_m2",
            ):
                spec.update(
                    metric=choice["metric"],
                    metric_reason=choice["reason"],
                    tentative_metric=True,
                    metric_choice_source="structured internal visual observation",
                )
            from eviground.query.noun_binding import box_size_spec

            spec = box_size_spec(spec, category, query)
            if (
                retrieval_category(category) == "book"
                and choice
                and (choice["metric"] == "body_plane_area_m2")
            ):
                spec.update(
                    metric="box_plane_area_m2",
                    metric_choice_source="visual single-book plane interpretation",
                    metric_reason=choice["reason"],
                )
            if parsed.get("category_recovery"):
                corner = re.search(
                    "(?:located\\s+)?in\\s+(?:a|the)\\s+corner\\b", query, re.I
                )
                if corner:
                    spec.update(
                        scope_kind="relation_subset",
                        scope_relation_quotes=[corner[0]],
                        scope_phrase=corner[0],
                    )
            if not any(
                (c["word"] == spec["word"] and c["role"] == role for c in conditions)
            ):
                conditions.append(spec)
    return conditions
