"""Per-attribute visual evidence, independent of grounding and render cameras."""

import re

SIZE_WORDS = {
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
GEOMETRIC_SIZE_WORDS = {
    "smallest",
    "largest",
    "biggest",
    "tallest",
    "highest",
    "shortest",
    "lowest",
}
_MODIFIERS = SIZE_WORDS | {
    "wide",
    "thick",
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
    "wooden",
    "metal",
    "metallic",
    "plastic",
    "glass",
    "folded",
    "vertical",
    "horizontal",
    "round",
    "rectangular",
}
_TEXTURES = {
    "stripe",
    "stripes",
    "striped",
    "dotted",
    "checkered",
    "speckled",
    "patterned",
    "smooth",
    "rough",
    "textured",
}
_PART_PHRASES = re.compile(
    "\\bon top\\b|\\b(?:top|bottom)\\s+(?:surface|shelf)\\b|\\b(?:front|back|side)\\s+panel\\b|\\bwood\\s+grain\\b",
    re.I,
)
_REFERENT_RELATION = re.compile(
    "\\b(?:next to|beside|between|against|near|behind|under|underneath|beneath|above|below|left of|right of|in front of|on top of|attached to|mounted on|hanging on|resting on|facing)\\b|\\b(?:on|in|at|with)\\s+(?:the|a|an)\\s+(?!(?:top|bottom)\\s+(?:surface|shelf)\\b)",
    re.I,
)


def attribute_units(phrase):
    """Expose literal properties while retaining qualified phrases as conditions."""
    if not isinstance(phrase, str) or not phrase.strip():
        return []
    tokens = re.findall("\\b\\w+\\b", phrase)
    modifiers = [word for word in tokens if word.casefold() in _MODIFIERS]
    if modifiers and all(
        (word.casefold() in _MODIFIERS or word.casefold() == "and" for word in tokens)
    ):
        return list(dict.fromkeys(modifiers))
    if re.search(
        "\\b(?:not|nothing|no|none|neither|without|except|excluding|rather than|other than|instead of)\\b",
        phrase,
        re.I,
    ) or _REFERENT_RELATION.search(phrase):
        return [phrase]
    properties = [
        word
        for word in tokens
        if word.casefold() in _MODIFIERS - GEOMETRIC_SIZE_WORDS | _TEXTURES
    ]
    parts = [match.group() for match in _PART_PHRASES.finditer(phrase)]
    return list(dict.fromkeys([phrase, *properties, *parts]))
