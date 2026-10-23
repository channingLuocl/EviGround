"""Quoted query obligations and a broad local graph; language never deletes IDs."""

import re
from eviground.categories.retrieval import retrieval_category

NEGATION = "\\b(?:no|not|nothing|never|without|except)\\b|n['’]t\\b"


def exclusion_clauses(query, syntax):
    if not re.search(NEGATION, query, re.I):
        return []
    if syntax.get("backend") != "spacy":
        return [
            {
                "source_quote": query,
                "source_span": [0, len(query)],
                "polarity": "unknown",
                "modified_category": None,
            }
        ]
    from eviground.query.query_syntax import _pipeline, _span

    doc = _pipeline()(query)
    clauses = []
    for token in doc:
        if not re.search(NEGATION, token.text, re.I):
            continue
        head = token if token.lower_ in ("without", "except") else token.head
        tokens = list(head.subtree)
        if head.pos_ in ("NOUN", "PROPN", "NUM"):
            tokens = [t for t in tokens if t.i >= token.i]
        span = _span(doc, tokens)
        noun = head if head.pos_ in ("NOUN", "PROPN") else head.head
        category = noun.lemma_ if noun.pos_ in ("NOUN", "PROPN") else None
        predicate = [head, token] + [
            t for t in head.children if t.dep_ in ("amod", "acomp", "attr")
        ]
        predicate = (
            tokens
            if head.pos_ in ("NOUN", "PROPN", "NUM")
            or head.lower_ in ("without", "except")
            else predicate
        )
        clauses.append(
            {
                "source_quote": span["text"],
                "source_span": [span["start"], span["end"]],
                "predicate_spans": [[t.idx, t.idx + len(t)] for t in predicate],
                "polarity": "exclude",
                "modified_category": category,
            }
        )
    return clauses or [
        {
            "source_quote": query,
            "source_span": [0, len(query)],
            "polarity": "unknown",
            "modified_category": None,
        }
    ]


def quote_spans(query, quote):
    return (
        [[m.start(), m.end()] for m in re.finditer(re.escape(quote), query, re.I)]
        if quote
        else []
    )


def polarity(query, quote, exclusions, attribute=False):
    spans = quote_spans(query, quote)
    if not spans:
        return "unknown"
    if len(spans) > 1:
        if not exclusions:
            return "positive"
        return "unknown"
    start, end = spans[0]
    overlapping = [
        c
        for c in exclusions
        if any(
            (
                start < b and end > a
                for a, b in (
                    c.get("predicate_spans", [c["source_span"]])
                    if attribute
                    else [c["source_span"]]
                )
            )
        )
    ]
    if not overlapping:
        return "positive"
    return (
        "exclude"
        if any((c["polarity"] == "exclude" for c in overlapping))
        else "unknown"
    )


def retain_positive_predicates(query, parsed):
    """Keep exclusions in the ledger, outside the existing positive executors."""
    exclusions = exclusion_clauses(query, parsed.get("query_syntax", {}))
    from eviground.query.noun_binding import bind_negative_attributes

    bind_negative_attributes(query, parsed, exclusions)
    parsed["Relations"] = [
        r
        for r in parsed.get("Relations", [])
        if polarity(query, r.get("phrase", ""), exclusions) == "positive"
    ]
    return exclusions
    for field in ("Attributes", "Relations"):
        parsed[field] = [
            row
            for row in parsed.get(field, [])
            if polarity(
                query,
                row.get("phrase", ""),
                exclusions,
                attribute=field == "Attributes",
            )
            == "positive"
        ]
    return exclusions


def structure(query, parsed):
    exclusions = exclusion_clauses(query, parsed.get("query_syntax", {}))
    conditions = []
    for record in parsed.get("condition_records", []):
        quote = record.get("source_quote")
        value = polarity(
            query,
            quote,
            exclusions,
            attribute=record["condition_type"] in ("appearance", "size"),
        )
        role = (
            record.get("target_role")
            or record.get("reference_role")
            or record.get("other_role")
        )
        negative_attribute = record["condition_type"] == "appearance" and any(
            (
                a["phrase"] == quote and a.get("polarity") == "exclude"
                for a in parsed.get("Attributes", [])
            )
        )
        bound = record.get("binding_status") == "bound" and (
            value == "positive" or negative_attribute
        )
        conditions.append(
            {
                **record,
                "condition_id": f"c{len(conditions)}",
                "source_spans": quote_spans(query, quote),
                "polarity": value,
                "modified_role": role,
                "parse_status": "bound" if bound else "unknown",
                **(
                    {"evidence_status": "unsupported", "binding_status": "unknown"}
                    if value != "positive" and (not negative_attribute)
                    else {}
                ),
                **(
                    {"executor": "negative_color", "evidence_status": "pending"}
                    if negative_attribute
                    else {}
                ),
            }
        )
    for clause in exclusions:
        category = retrieval_category(clause.get("modified_category") or "")
        roles = [
            role
            for role, names in (
                ("target", parsed.get("Target", [])),
                ("reference", parsed.get("Anchor", [])),
            )
            if category and category in {retrieval_category(n) for n in names}
        ]
        executed = clause["source_quote"] in parsed.get("_executed_negations", [])
        conditions.append(
            {
                **clause,
                "condition_id": f"c{len(conditions)}",
                "condition_type": "exclusion",
                "parse_status": "unknown",
                "evidence_status": "unsupported",
                "modified_role": (
                    {"role": roles[0], "category": category}
                    if len(roles) == 1
                    else None
                ),
                "reason": "exclusion requires a source-bound predicate; positive evidence cannot execute it",
                **(
                    {
                        "parse_status": "bound",
                        "evidence_status": "pending",
                        "executor": "negative_color",
                        "reason": "source-bound negative color is sent to appearance verification",
                    }
                    if executed
                    else {}
                ),
            }
        )
    covered = {
        i
        for c in conditions
        for start, end in c.get("source_spans", [c.get("source_span", [0, 0])])
        for i in range(start, end)
    }
    covered.update(
        (
            i
            for c in conditions
            for hint in c.get("language_comparisons", [])
            for i in range(*hint["source_span"])
        )
    )
    gaps = [
        t["text"]
        for t in parsed.get("query_syntax", {}).get("dependencies", [])
        if (t["pos"] in ("ADJ", "ADV") or t["dependency"] in ("acl", "amod"))
        and t["text"].casefold() not in ("the", "a", "an", "there", "please")
        and (not any((i in covered for i in range(t["start"], t["end"]))))
    ]
    for quote in dict.fromkeys(gaps):
        conditions.append(
            {
                "condition_id": f"c{len(conditions)}",
                "condition_type": "coverage_gap",
                "source_quote": quote,
                "source_spans": quote_spans(query, quote),
                "polarity": "unknown",
                "modified_role": None,
                "parse_status": "unknown",
                "evidence_status": "missing",
                "reason": "modifier or connector is not represented by a quoted condition",
            }
        )
    return {
        "target_categories": parsed.get("Target", []),
        "reference_categories": parsed.get("Anchor", []),
        "conditions": conditions,
        "uncovered_modifiers": list(dict.fromkeys(gaps)),
        "policy": "quoted language hints only; unknown/exclusion conditions cannot remove candidates",
    }
