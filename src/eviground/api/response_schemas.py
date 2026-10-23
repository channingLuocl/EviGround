"""Strict structured output contracts for parsing and ReAct decisions."""


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def array(items):
    return {"type": "array", "items": items}


def enum(*values):
    return {"type": "string", "enum": list(values)}


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


TEXT = {"type": "string"}
REASON = {"type": "string", "minLength": 1, "pattern": "\\S"}
ROLE = enum("target", "reference")
ATTRIBUTE = obj({"category": TEXT, "role": ROLE, "phrase": TEXT})
RELATION = obj(
    {
        "target_category": TEXT,
        "anchor_category": TEXT,
        "anchor_categories": array(TEXT),
        "relation": enum(
            "nearest",
            "farthest",
            "near",
            "left",
            "right",
            "front",
            "back",
            "above",
            "below",
            "underneath",
            "between",
            "other",
        ),
        "scope": enum("", "side", "relative_anchor", "relative_anchors"),
        "phrase": TEXT,
    }
)
OBSERVER = obj(
    {
        "origin": obj(
            {
                "kind": enum("room_center", "category", "unspecified"),
                "category": TEXT,
                "phrase": TEXT,
            }
        ),
        "facing": obj({"category": TEXT, "group": {"type": "boolean"}, "phrase": TEXT}),
    }
)


def parser_schema():
    return obj(
        {
            "Target": array(TEXT),
            "Anchor": array(TEXT),
            "Relations": array(RELATION),
            "Viewpoint": TEXT,
            "Observer": OBSERVER,
            "Attributes": array(ATTRIBUTE),
            "Unhandled": array(TEXT),
        }
    )


ATTRIBUTES = obj({"Attributes": array(ATTRIBUTE)})


def schema_kwargs(name, schema):
    from eviground.api.client import api_provider

    if api_provider() == "openai":
        return {"response_schema": {"name": name, "strict": True, "schema": schema}}
    return {}
