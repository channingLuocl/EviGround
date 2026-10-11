"""English dependency hints; language structure never certifies scene evidence."""
from functools import lru_cache
from importlib.metadata import version
import os

from size_semantics import SIZE_WORDS


MODEL = 'en_core_web_sm'


def syntax_settings():
    backend = os.getenv('SEEGROUND_SYNTAX_BACKEND', 'spacy').strip()
    if backend not in ('spacy', 'none'):
        raise ValueError('SEEGROUND_SYNTAX_BACKEND must be spacy or none')
    settings = {'backend': backend, 'implementation_version': 'relative-size-v3'}
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
def _analyze(query):
    doc = _pipeline()(query)
    chunks = list(doc.noun_chunks)
    comparisons = []
    for token in doc:
        if token.lower_ not in SIZE_WORDS:
            continue
        head = token.head if token.dep_ == 'amod' else token
        prepositions = [t for t in head.children if t.lower_ in ('of', 'than', 'among')]
        scope = prepositions[0] if len(prepositions) == 1 else None
        noun = head if head.pos_ in ('NOUN', 'PROPN') and head.lower_ not in SIZE_WORDS else None
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
        modifiers += uncertain_modifiers
        counts = [t for t in (noun.children if noun is not None else []) if t.dep_ == 'nummod']
        if scope is not None:
            counts += [t for t in scope.children if t.dep_ == 'pobj' and t.pos_ == 'NUM']
        numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                   'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
        count_values = [int(t.text) if t.text.isdigit() else numbers.get(t.lower_)
                        for t in counts]
        count_values = [n for n in count_values if n is not None]
        phrase = _span(doc, token.head.subtree if reference is not None else head.subtree)
        record = {'word': token.lower_, 'source_phrase': phrase['text'],
                  'source_span': [phrase['start'], phrase['end']],
                  'word_span': [token.idx, token.idx + len(token)],
                  'head_category': (' '.join([t.lemma_ for t in noun.children if t.dep_ == 'compound']
                                             + [noun.lemma_]) if noun is not None else ''),
                  'noun_phrase': noun_phrase.text if noun_phrase is not None else '',
                  'reference_category': reference.lemma_ if reference is not None else None,
                  'scope_phrase': _span(doc, members)['text'],
                  'scope_type': ('reference_comparison' if scope is not None and scope.lower_ == 'than'
                                 else 'explicit_set' if scope is not None else 'category'),
                  'required_member_count': (count_values[0] if scope is not None
                      and scope.lower_ != 'than' and len(count_values) == 1 else None),
                  'member_binding_status': 'pending_scene_binding',
                  'modifiers': [_span(doc, t.subtree) for t in modifiers],
                  'attachment': ('ambiguous' if uncertain_modifiers else
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
    return {**settings, **(_analyze(query) if settings['backend'] == 'spacy' else {
        'comparisons': [], 'noun_phrases': [], 'dependencies': [],
        'policy': 'syntax disabled for singleton-only ablation'})}


def comparison_hint(parsed, word, category):
    from evidence import retrieval_category
    matches = [c for c in parsed.get('query_syntax', {}).get('comparisons', [])
               if c['word'] == word and retrieval_category(c['head_category']) == retrieval_category(category)]
    return matches[0] if len(matches) == 1 else None
