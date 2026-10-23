"""Per-attribute visual evidence, independent of grounding and render cameras."""
import re


SIZE_WORDS = {
    'small', 'smallest', 'smaller', 'large', 'largest', 'larger', 'big',
    'biggest', 'bigger', 'little', 'skinny', 'narrow', 'thin', 'tall',
    'tallest', 'taller', 'high', 'highest', 'short', 'shortest', 'shorter',
    'low', 'lowest',
}
GEOMETRIC_SIZE_WORDS = {'smallest', 'largest', 'biggest', 'tallest',
                        'highest', 'shortest', 'lowest'}
_PHYSICAL_SCALE_WORDS = SIZE_WORDS - {'skinny', 'narrow', 'thin'}
_UNCERTAIN_NEGATIVE = re.compile(
    r'\b(?:uncertain|unclear|insufficient detail|not clearly|not enough detail)\b|'
    r'\b(?:cannot|can\'t|could not|unable to)\s+'
    r'(?:confirm|verify|determine|tell|assess|distinguish)\b', re.I)
VISIBILITY_VALUES = {'visible', 'occluded', 'low_detail', 'unmarked',
                     'missing_view', 'unknown'}
_COLORS = {'black', 'white', 'red', 'blue', 'green', 'grey', 'gray',
           'brown', 'yellow', 'orange', 'purple', 'pink'}
_COLOR_COVERAGE = re.compile(
    r'\b(?:all|entire(?:ly)?|whole|uniform(?:ly)?|predominant(?:ly)?|mostly|'
    r'completely|solid|fully|throughout|pure(?:ly)?|majority|dominant(?:ly)?)\b', re.I)
_COLOR_PART = (r'(?:top|tabletop|surface|panel|door|seat|backrest|frame|base|legs?|'
               r'arms?|trim|stripes?|details?|parts?|sections?)')
_MODIFIERS = SIZE_WORDS | {
    'wide', 'thick', 'black', 'white', 'red', 'blue', 'green', 'grey', 'gray',
    'brown', 'yellow', 'orange', 'purple', 'pink', 'wooden', 'metal', 'metallic',
    'plastic', 'glass', 'folded', 'vertical', 'horizontal', 'round', 'rectangular',
}
_TEXTURES = {'stripe', 'stripes', 'striped', 'dotted', 'checkered', 'speckled',
             'patterned', 'smooth', 'rough', 'textured'}
_PART_PHRASES = re.compile(
    r'\bon top\b|\b(?:top|bottom)\s+(?:surface|shelf)\b|'
    r'\b(?:front|back|side)\s+panel\b|\bwood\s+grain\b', re.I)
_REFERENT_RELATION = re.compile(
    r'\b(?:next to|beside|between|against|near|behind|under|underneath|beneath|'
    r'above|below|left of|right of|in front of|on top of|attached to|mounted on|'
    r'hanging on|resting on|facing)\b|'
    r'\b(?:on|in|at|with)\s+(?:the|a|an)\s+'
    r'(?!(?:top|bottom)\s+(?:surface|shelf)\b)', re.I)
_UNSUPPORTED_NEGATION = re.compile(
    r'\b(?:no|not|nothing|never|none|neither|without|except|excluding|'
    r'rather than|other than|instead of)\b|n[\'’]t\b', re.I)


def attribute_units(phrase):
    """Expose literal properties while retaining qualified phrases as conditions."""
    if not isinstance(phrase, str) or not phrase.strip():
        return []
    tokens = re.findall(r'\b\w+\b', phrase)
    modifiers = [word for word in tokens if word.casefold() in _MODIFIERS]
    if modifiers and all(word.casefold() in _MODIFIERS or word.casefold() == 'and'
                         for word in tokens):
        return list(dict.fromkeys(modifiers))
    # Negation and external references cannot be turned into target properties.
    if re.search(r'\b(?:not|nothing|no|none|neither|without|except|excluding|'
                 r'rather than|other than|instead of)\b', phrase, re.I) \
            or _REFERENT_RELATION.search(phrase):
        return [phrase]
    # Named properties are additional checks. They never replace the complete
    # phrase, which preserves shade, part, repetition and other qualifications.
    properties = [word for word in tokens if word.casefold() in
                  (_MODIFIERS - GEOMETRIC_SIZE_WORDS) | _TEXTURES]
    parts = [match.group() for match in _PART_PHRASES.finditer(phrase)]
    return list(dict.fromkeys([phrase, *properties, *parts]))


def required_visual_attributes(phrases):
    """Keep visual modifiers while leaving explicit size extrema to geometry."""
    return list(dict.fromkeys(unit for phrase in phrases for unit in attribute_units(phrase)
                              if unit.casefold() not in GEOMETRIC_SIZE_WORDS))


def attribute_contexts(expected, source_phrases=None):
    """Keep each derived fragment bound to its complete quoted property."""
    qualified, independent = {}, {}
    for phrase in expected if source_phrases is None else source_phrases:
        units = attribute_units(phrase)
        if len(units) > 1 and units[0] == phrase:
            for fragment in units[1:]:
                qualified.setdefault(fragment, []).append(phrase)
        elif source_phrases is not None:
            for unit in units:
                independent.setdefault(unit, []).append(phrase)
    return [{'attribute': name,
             'source_quotes': list(dict.fromkeys(independent.get(name, []) +
                                                 qualified.get(name, []))) or [name],
             'qualified_fragment': name in qualified and name not in independent}
            for name in expected]


def _visibility(row, allow_legacy=False):
    value = row.get('visibility')
    if value in VISIBILITY_VALUES:
        return value
    if value is not None:
        return 'unknown'
    # Legacy category-only traces can retain their conclusion. Attribute
    # evidence always needs an explicit visibility report.
    return ('visible' if allow_legacy and row.get('verdict') in ('yes', 'no')
            else 'unknown')


def _bare_color_scope_conflict(name, source_quotes, reason):
    """A quoted bare color does not require uniform or predominant coverage."""
    color = name.casefold()
    if color not in _COLORS or any(_COLOR_COVERAGE.search(quote) for quote in source_quotes):
        return False
    # The verifier may negate an invented purity/coverage requirement even
    # without naming a matching part. It must not strengthen a bare quote.
    if re.search(r'\b(?:not|isn\'t|aren\'t|without|lack(?:s|ing)?)\s+'
                 r'(?:(?:appear|appears)\s+(?:to\s+be\s+)?|'
                 r'(?:match|satisfy|meet)\s+(?:the\s+)?)?'
                 r'(?:pure(?:ly)?|entirely|uniform(?:ly)?|predominant(?:ly)?|'
                 r'completely|fully|all)\s+[\'\"]?' + color + r'\b', reason, re.I):
        return True
    # Only affirmative evidence about an instance part counts. A clear
    # opposite color ("the top is brown, not white") must remain decisive.
    for clause in re.split(r'[;.!]', reason):
        if re.search(r'\b(?:no|none|neither|without)\b', clause, re.I):
            continue
        if (re.search(r'\b(?:majority|dominant(?:ly)?)\b', reason, re.I)
                and not re.search(r'\b(?:annotation|numeric|ID|overlay)\b|\blabel\s+box\b', clause, re.I)
                and re.search(r'\b(?:has|shows)\s+(?:(?:a|an|some|the)\s+)?'
                              r'(?:(?:significant|visible|clear)\s+)?' + color +
                              r'\s+(?:region|area|portion|patch)\b', clause, re.I)):
            return True
        if re.search(_COLOR_PART + r'\s+(?:surface\s+)?(?:is|are)\s+'
                     r'(?:visibly\s+|clearly\s+)?' + color + r'\b', clause, re.I) or re.search(
                r'\b(?:has|with|shows)\s+(?:a\s+|an\s+|the\s+|some\s+)?' + color +
                r'\s+' + _COLOR_PART + r'\b', clause, re.I):
            return True
        # A verifier may name an unlisted part and then invent whole-object
        # coverage. Preserve that explicit partial match without a part alias.
        if (re.search(r'\b(?:whole|entire)\s+(?:object|body)\b|'
                      r'\b(?:uniform(?:ly)?|predominant(?:ly)?|fully|completely)\b', reason, re.I)
                and not re.search(r'\b(?:annotation|numeric|ID|overlay)\b|\blabel\s+box\b', clause, re.I)
                and re.search(r'\bonly\s+(?:the\s+|a\s+|an\s+)?'
                              r'(?:\w+\s+){1,4}(?:is|are)\s+'
                              r'(?:visibly\s+|clearly\s+)?' + color + r'\b', clause, re.I)):
            return True
    return False


def _negative_reason_is_uncertain(name, reason):
    if _UNCERTAIN_NEGATIVE.search(reason):
        return True
    # An incidental hypothesis ("black top may be misinterpreted, but is
    # distinctly black") does not make the requested blue property unknown.
    properties = ([name] if name.strip() else []) + [
        word for word in re.findall(r'\b\w+\b', name)
        if word.casefold() in _MODIFIERS | _TEXTURES]
    return any(re.search(r'\b(?:may|might|could)\s+be\s+'
                         r'(?:(?:a|an|visibly|clearly)\s+)?[\'\"]?' +
                         re.escape(property_name) + r'\b', reason, re.I)
               for property_name in properties)


def _annotation_used_as_attribute_evidence(name, reason):
    """Reject explicit positive use of artificial marks, not mentions of IDs."""
    annotation = re.compile(
        r'\bannotations?\b|'
        r'\b(?:synthetic|artificial|rendered)\s+(?:ID\s+)?'
        r'(?:overlays?|labels?|numbers?|marks?|boxes?)\b|'
        r'\b(?:ID|instance|numeric)\s+(?:ID\s+)?(?:numbers?|labels?|boxes?)\b|'
        r'\blabel\s+with\s+red\s+[\'\"]?\d+\b', re.I)
    support = re.compile(
        r'\b(?:qualif(?:y|ies)|counts?|satisf(?:y|ies)|meets?|fulfills?|'
        r'supplies?|proves?|establishes?|confirms?|supports?)\b', re.I)
    ignored = re.compile(r'\b(?:ignor(?:e|ed|ing)|disregard(?:ed)?|excluded|'
                         r'not\s+(?:part|evidence))\b', re.I)
    attachment = re.compile(r'\b(?:next\s+to|beside|alongside|near|above|below|'
                            r'around|despite|has|with)\b', re.I)
    previous_annotation = False
    for clause in re.split(r'[;.!?,]|\s+(?:and|but|although|even\s+though|whereas)\s+',
                           reason, flags=re.I):
        marks = list(annotation.finditer(clause))
        # A following sentence may explicitly cite "this white area" as its
        # evidence after identifying the artificial numeric label.
        inherited = previous_annotation
        previous_annotation = bool(marks) and not ignored.search(clause)
        if re.search(r'\b' + re.escape(name) + r'\b', clause, re.I):
            for verb in support.finditer(clause):
                subject = clause[:verb.start()]
                if inherited and re.fullmatch(
                        r'\s*(?:(?:the\s+presence\s+of\s+)?(?:this|that)\s+'
                        r'(?:\w+\s+){0,3}(?:area|label|box|mark|component)\s+)', subject, re.I):
                    return True
                for mark in marks:
                    if (mark.end() <= verb.start() and
                            not attachment.search(clause[:mark.start()]) and
                            re.fullmatch(r'\s*(?:(?:box|label|number|mark)\s+)?'
                                         r'(?:(?:clearly|directly|visibly|itself|alone)\s+)*',
                                         clause[mark.end():verb.start()], re.I)):
                        return True
    return False


def normalize_visual_check(check, expected, source_phrases=None):
    """A generic yes never proves unreported attributes or an invisible view."""
    record = dict(check)
    if source_phrases is None:
        source_phrases = check.get('source_attributes')
    if source_phrases is not None:
        record['source_attributes'] = list(source_phrases)
    record['visual_verdict'] = check.get('visual_verdict', check.get('verdict', 'uncertain'))
    record['visibility'] = _visibility(check, allow_legacy=not expected)
    record['visibility_source'] = check.get('visibility_source',
        'reported' if 'visibility' in check else 'legacy_verdict' if not expected else 'missing')
    record['inspection_status'] = check.get('inspection_status', 'inspected')
    if record['inspection_status'] not in ('inspected', 'uninspected'):
        record['inspection_status'] = 'uninspected'
    # A failed conjunction does not identify which condition failed.
    fallback = check.get('verdict', 'uncertain')
    category_verdict = check.get('category_verdict',
        'uncertain' if expected and fallback == 'no' else fallback)
    if (category_verdict not in ('yes', 'no', 'uncertain') or record['visibility'] != 'visible'
            or record['inspection_status'] == 'uninspected'):
        category_verdict = 'uncertain'
    if (category_verdict == 'no' and isinstance(check.get('reason'), str)
            and _negative_reason_is_uncertain(check.get('category', ''), check['reason'])):
        record['category_visual_verdict'] = 'no'
        category_verdict = 'uncertain'
    record['category_verdict'] = category_verdict
    raw = check.get('attribute_checks', [])
    raw = raw if isinstance(raw, list) else []
    attributes = []
    contexts = {item['attribute']: item for item in attribute_contexts(expected, source_phrases)}
    provenance = record.get('render_provenance', {})
    independent_scale = (
        provenance.get('source_type') == 'real_rgb_projected_instance' or
        (provenance.get('source_type') == 'colored_point_cloud'
         and provenance.get('context') == 'isolated'))
    invalid = any(not isinstance(item, dict) or item.get('attribute') not in expected
                  for item in raw)
    for name in expected:
        matching = [item for item in raw if isinstance(item, dict) and item.get('attribute') == name]
        if (len(matching) != 1 or matching[0].get('verdict') not in ('yes', 'no', 'uncertain')
                or not isinstance(matching[0].get('reason'), str) or not matching[0]['reason'].strip()):
            attributes.append({**contexts[name], 'verdict': 'uncertain', 'visibility': 'unknown',
                               'inspection_status': 'uninspected',
                               'visibility_source': 'missing',
                               'reason': 'attribute was not independently and uniquely verified'})
            continue
        item = dict(matching[0])
        item.update(contexts[name])
        item['inspection_status'] = item.get('inspection_status', 'inspected')
        if invalid or item['inspection_status'] not in ('inspected', 'uninspected'):
            item['inspection_status'] = 'uninspected'
        item['visibility'] = _visibility(item)
        item['visibility_source'] = item.get('visibility_source',
            'reported' if 'visibility' in item else 'missing')
        if (record['visibility'] != 'visible' or item['visibility'] != 'visible' or invalid
                or item['inspection_status'] == 'uninspected'
                or record['inspection_status'] == 'uninspected'):
            item['verdict'] = 'uncertain'
        if (item['verdict'] == 'yes' and not record.get('view_evidence') and any(
                _annotation_used_as_attribute_evidence(name, reason) for reason in
                (item['reason'], check.get('reason', '')) if isinstance(reason, str))):
            # Combined reasons already contain normalized per-view evidence;
            # an invalid view must not erase an earlier genuine surface match.
            item['visual_verdict'] = item.get('visual_verdict', 'yes')
            item['verdict'] = 'uncertain'
            item['evidence_limit'] = 'artificial annotation is not object attribute evidence'
            item['reason'] = 'Invalid annotation evidence: ' + item['reason']
        from eviground.grounding.v4_policy import v7_enabled
        supported_negative = v7_enabled() and re.fullmatch(r'not (?:' + '|'.join(sorted(_COLORS)) + r')', name, re.I)
        if item['verdict'] in ('yes', 'no') and _UNSUPPORTED_NEGATION.search(name) and not supported_negative:
            # An unsupported predicate can remain inside a quoted appearance
            # conjunction; neither visual verdict implements that predicate.
            item['visual_verdict'] = item.get('visual_verdict', item['verdict'])
            item['verdict'] = 'uncertain'
            item['evidence_limit'] = 'negation/exclusion predicate is unsupported'
            item['reason'] = 'Unsupported condition needs its own predicate: ' + item['reason']
        if (item['verdict'] in ('yes', 'no') and independent_scale and
                set(re.findall(r'\b\w+\b', name.casefold())) & _PHYSICAL_SCALE_WORDS):
            # Separately fitted cameras/crops preserve color and silhouette,
            # but cannot establish physical size, even for a confident yes.
            item['visual_verdict'] = item.get('visual_verdict', item['verdict'])
            item['verdict'] = 'uncertain'
            item['evidence_limit'] = 'independently framed views do not establish physical scale'
            item['reason'] = 'Physical-scale evidence unavailable: ' + item['reason']
        if item['verdict'] == 'no' and _negative_reason_is_uncertain(name, item['reason']):
            item['visual_verdict'] = item.get('visual_verdict', 'no')
            item['verdict'] = 'uncertain'
            item['evidence_limit'] = 'negative reason explicitly reports uncertainty'
            item['reason'] = 'Uncertain evidence cannot establish contradiction: ' + item['reason']
        if (item['verdict'] == 'no' and any(
                _bare_color_scope_conflict(name, item['source_quotes'], reason)
                for reason in (item['reason'], check.get('reason', ''))
                if isinstance(reason, str))):
            item['visual_verdict'] = item.get('visual_verdict', 'no')
            item['verdict'] = 'uncertain'
            item['evidence_limit'] = 'bare color coverage is unspecified despite visible matching part'
            item['reason'] = 'Color coverage scope unresolved: ' + item['reason']
        if item['qualified_fragment'] and item['verdict'] == 'no':
            # A bare color/material fragment may have been read as a property
            # of the whole body rather than of the quoted stripe/panel/part.
            # Only its complete property can supply a hard contradiction.
            item['qualified_fragment_verdict'] = 'no'
            item['verdict'] = 'uncertain'
            item['reason'] = 'Fragment contradiction needs its complete quoted property: ' + item['reason']
        attributes.append(item)
    record['attribute_checks'] = attributes
    verdicts = [category_verdict] + [item['verdict'] for item in attributes]
    record['verdict'] = ('no' if 'no' in verdicts else
                         'uncertain' if 'uncertain' in verdicts or invalid else 'yes')
    reason = check.get('reason') if isinstance(check.get('reason'), str) else ''
    if record['verdict'] == 'uncertain' and expected:
        record['reason'] = 'Incomplete or uncertain per-attribute visual evidence: ' + reason
    elif attributes:
        record['reason'] = reason + '; ' + '; '.join(
            item['attribute'] + ': ' + item['reason'] for item in attributes)
    return record


def aggregate_visual_checks(checks, expected, source_phrases=None):
    """Accumulate visible evidence across views without erasing earlier support."""
    if not checks:
        return None
    normalized = [normalize_visual_check(check, expected, source_phrases) for check in checks]

    def combine(verdicts):
        known = {verdict for verdict in verdicts if verdict in ('yes', 'no')}
        return next(iter(known)) if len(known) == 1 else 'uncertain'

    record = dict(normalized[-1])
    if any('render_provenance' in row for row in normalized):
        # The last camera is not the source of every accumulated property.
        # Each view has already passed its own evidence-applicability guard.
        record['render_provenance'] = {
            'source_type': 'combined_visual_views',
            'description': 'normalized per-view evidence; sources retained in view_evidence'}
    record['category_verdict'] = combine(row['category_verdict'] for row in normalized)
    attributes = []
    contexts = {item['attribute']: item for item in
                attribute_contexts(expected, source_phrases if source_phrases is not None
                                   else record.get('source_attributes'))}
    for name in expected:
        rows = [item for check in normalized for item in check['attribute_checks']
                if item['attribute'] == name]
        verdict = combine(item['verdict'] for item in rows)
        attributes.append({**contexts[name], 'verdict': verdict,
                           'inspection_status': ('inspected' if any(
                               item['inspection_status'] == 'inspected' for item in rows)
                               else 'uninspected'),
                           'visibility': 'visible' if verdict != 'uncertain' else 'unknown',
                           'reason': '; '.join(dict.fromkeys(item['reason'] for item in rows))})
    record['attribute_checks'] = attributes
    record['inspection_status'] = (
        'inspected' if any(row['inspection_status'] == 'inspected' for row in normalized)
        else 'uninspected')
    verdicts = [record['category_verdict']] + [item['verdict'] for item in attributes]
    record['verdict'] = ('no' if 'no' in verdicts else
                         'uncertain' if 'uncertain' in verdicts else 'yes')
    record['view_evidence'] = normalized
    return record
