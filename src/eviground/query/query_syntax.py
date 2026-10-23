"""English dependency hints; language structure never certifies scene evidence."""
from functools import lru_cache
from importlib.metadata import version
import os

from eviground.geometry.size_semantics import SIZE_WORDS


MODEL = 'en_core_web_sm'


def syntax_settings():
    backend = os.getenv('EVIGROUND_SYNTAX_BACKEND', 'spacy').strip()
    if backend not in ('spacy', 'none'):
        raise ValueError('EVIGROUND_SYNTAX_BACKEND must be spacy or none')
    from eviground.grounding.v4_policy import settings as v4_settings
    settings = {'backend': backend, 'implementation_version': v4_settings()['version']}
    if backend == 'spacy':
        settings.update(model=MODEL, spacy_version=version('spacy'), model_version=version(MODEL))
    return settings


@lru_cache(maxsize=1)
def _pipeline():
    import spacy
    # Models are installed explicitly; inference never downloads dependencies.
    return spacy.load(MODEL, exclude=['ner'])


def _span(doc, tokens):
    tokens = sorted((t for t in tokens if not t.is_punct), key=lambda t: t.i)
    if not tokens:
        return {'text': '', 'start': 0, 'end': 0}
    start, end = tokens[0].idx, tokens[-1].idx + len(tokens[-1])
    return {'text': doc.text[start:end], 'start': start, 'end': end}


@lru_cache(maxsize=512)
def _analyze(query, v5_scope=False):
    doc = _pipeline()(query)
    chunks = list(doc.noun_chunks)
    comparisons = []
    for token in doc:
        if token.lower_ not in SIZE_WORDS:
            continue
        head = token.head if token.dep_ == 'amod' or (v5_scope and token.dep_ == 'compound') else token
        prepositions = [t for t in head.children if t.lower_ in ('of', 'than', 'among')]
        scope = prepositions[0] if len(prepositions) == 1 else None
        noun = head if head.pos_ in ('NOUN', 'PROPN') and head.lower_ not in SIZE_WORDS else None
        # Keep a contiguous compound category even when the small model tags
        # its final word as an auxiliary (for example, an object named a can).
        # Actual use still requires an exact parsed-category match below.
        if (v5_scope and noun is not None and noun.dep_ == 'compound'
                and noun.head.i == noun.i + 1):
            noun = head = noun.head
        reference = None
        if scope is not None:
            member = next((t for t in scope.children if t.dep_ in ('pobj', 'obj')), None)
            if scope.lower_ == 'than':
                reference = member
                noun = next((t for t in token.head.children if t.dep_ == 'nsubj'), noun)
            elif member is not None and member.pos_ in ('NOUN', 'PROPN'):
                noun = member
        noun_phrase = next((chunk for chunk in chunks if noun is not None and chunk.root == noun), None)
        members = list(scope.subtree) if scope is not None else []
        modifiers = [t for t in (noun.children if noun is not None else [])
                     if t.dep_ in ('prep', 'relcl') and t != scope]
        # Small models sometimes attach the post-set PP to "larger" instead
        # of "windows". The explicit of-set still supplies its quoted scope.
        uncertain_modifiers = []
        if scope is not None and noun is not None:
            uncertain_modifiers = [t for t in head.children if t.dep_ in ('prep', 'relcl')
                                   and t != scope and t.idx > noun.idx and t not in modifiers]
        if v5_scope and scope is None and noun is not None:
            # Recover only the known proximity connector, through at most two
            # advmod levels; this is an attachment hint, never relation proof.
            def next_to(t):
                return (t.lower_ == 'next' and t.dep_ == 'advmod'
                        and any(c.lower_ == 'to' and c.dep_ == 'prep' for c in t.children))

            proximity = [t for t in doc if next_to(t) and
                         (t.head == noun or (t.head.dep_ == 'advmod' and t.head.head == noun))]
            uncertain_modifiers += [t for t in proximity if t not in modifiers]
            # Selection imperatives often attach post-object PPs to the verb.
            # Keep only on/by/next-to; do not reinterpret arbitrary verb PPs.
            verb = noun.head
            if (noun.dep_ in ('dobj', 'obj') and verb.dep_ == 'ROOT'
                    and verb.lemma_.lower() in ('choose', 'find', 'pick', 'select', 'click')
                    and not any(t.dep_ == 'nsubj' for t in verb.children)):
                def spatial_pp(t):
                    member = next((c for c in t.children if c.dep_ in ('pobj', 'obj')
                                   and c.pos_ in ('NOUN', 'PROPN')), None)
                    time_words = ('monday tuesday wednesday thursday friday saturday sunday '
                        'january february march april may june july august september october november december '
                        'morning afternoon evening night noon midnight day week month year weekend birthday').split()
                    return (t.dep_ == 'prep' and t.lower_ in ('on', 'by') and member is not None
                            and member.lemma_.lower() not in time_words)

                uncertain_modifiers += [t for t in verb.children if t.idx > noun.idx
                    and t.sent == noun.sent and
                    (spatial_pp(t) or next_to(t))
                    and t not in modifiers and t not in uncertain_modifiers]
        modifiers += uncertain_modifiers
        counts = [t for t in (noun.children if noun is not None else []) if t.dep_ == 'nummod']
        if scope is not None:
            counts += [t for t in scope.children if t.dep_ == 'pobj' and t.pos_ == 'NUM']
        numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                   'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
        count_values = [int(t.text) if t.text.isdigit() else numbers.get(t.lower_)
                        for t in counts]
        count_values = [n for n in count_values if n is not None]
        antecedent = None
        antecedent_modifiers = []
        if v5_scope and scope is None and noun is not None:
            # A unique earlier existential group supplies a quoted count and
            # membership clause, rather than the entire scene category.
            groups = [t for t in doc if t.i < token.i and t.sent != token.sent
                      and t.lemma_ == noun.lemma_ and t.head.lemma_ == 'be'
                      and any(c.dep_ == 'expl' and c.lower_ == 'there' for c in t.head.children)]
            counted = []
            for group in groups:
                values = [int(c.text) if c.text.isdigit() else numbers.get(c.lower_)
                          for c in group.children if c.dep_ == 'nummod']
                if len(values) == 1 and values[0] is not None:
                    counted.append((group, values[0]))
            if len(counted) == 1:
                antecedent, count = counted[0]
                count_values = [count]
                antecedent_modifiers = [t for t in antecedent.children if t.dep_ in ('prep', 'relcl', 'acl')]
        phrase_tokens = list(token.head.subtree if reference is not None else head.subtree)
        if v5_scope and uncertain_modifiers:
            phrase_tokens += [t for modifier in uncertain_modifiers for t in modifier.subtree]
        phrase = _span(doc, phrase_tokens)
        record = {'word': token.lower_, 'source_phrase': phrase['text'],
                  'source_span': [phrase['start'], phrase['end']],
                  'word_span': [token.idx, token.idx + len(token)],
                  'head_category': (' '.join([t.lemma_ for t in noun.children if t.dep_ == 'compound'
                                             and not (v5_scope and t.lower_ in SIZE_WORDS)]
                                             + [noun.lemma_]) if noun is not None else ''),
                  'noun_phrase': noun_phrase.text if noun_phrase is not None else '',
                  'reference_category': reference.lemma_ if reference is not None else None,
                  'scope_phrase': _span(doc, antecedent.subtree if antecedent is not None else members)['text'],
                  'scope_type': ('reference_comparison' if scope is not None and scope.lower_ == 'than'
                                 else 'explicit_set' if scope is not None or antecedent is not None else 'category'),
                  'required_member_count': (count_values[0] if len(count_values) == 1
                      and ((scope is not None and scope.lower_ != 'than') or antecedent is not None) else None),
                  'member_binding_status': 'pending_scene_binding',
                  'modifiers': [_span(doc, t.subtree) for t in
                                (antecedent_modifiers if antecedent is not None else modifiers)],
                  **({'antecedent_group': _span(doc, antecedent.subtree),
                      'independent_modifiers': [_span(doc, t.subtree) for t in modifiers]}
                     if antecedent is not None else {}),
                  'attachment': ('restrictive_subset' if antecedent is not None and antecedent_modifiers else
                                 'independent' if antecedent is not None else
                                 'ambiguous' if uncertain_modifiers else
                                 'restrictive_subset' if scope is not None and modifiers else
                                 'ambiguous' if modifiers else 'independent')}
        comparisons.append(record)
    return {
        'comparisons': comparisons,
        'noun_phrases': [{'text': c.text, 'start': c.start_char, 'end': c.end_char,
                         'head': c.root.lemma_} for c in chunks],
        'dependencies': [{'text': t.text, 'start': t.idx, 'end': t.idx + len(t),
                          'pos': t.pos_, 'dependency': t.dep_, 'head': t.head.i}
                         for t in doc],
        'policy': 'dependency attachments are hints, not object IDs or verified predicates',
    }


def analyze_query(query):
    settings = syntax_settings()
    from eviground.grounding.v4_policy import v5_enabled
    return {**settings, **(_analyze(query, v5_enabled('scope')) if settings['backend'] == 'spacy' else {
        'comparisons': [], 'noun_phrases': [], 'dependencies': [],
        'policy': 'syntax disabled for singleton-only ablation'})}


def comparison_hint(parsed, word, category):
    from eviground.grounding.evidence import retrieval_category
    matches = [c for c in parsed.get('query_syntax', {}).get('comparisons', [])
               if c['word'] == word and retrieval_category(c['head_category']) == retrieval_category(category)]
    return matches[0] if len(matches) == 1 else None
