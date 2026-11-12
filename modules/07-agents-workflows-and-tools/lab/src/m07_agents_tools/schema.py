"""A small JSON Schema subset validator (standard library only).

It understands the keywords that tool definitions use most: ``type``,
``properties``, ``required``, ``additionalProperties`` (``false`` only),
``enum``, ``minLength``, ``maxLength``, ``pattern``, ``minimum``, ``maximum``
and ``items``. It is a teaching tool, not a full implementation of the JSON
Schema specification: unknown keywords are ignored.
"""

from __future__ import annotations

import re
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected in ("integer", "number") and isinstance(value, bool):
        return False  # in Python, True is an int; in JSON it is not a number
    return isinstance(value, _TYPES[expected])


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable errors; an empty list means the value is valid."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match pattern {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, dict):
        props: dict[str, Any] = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property '{name}'")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append(f"{path}: unexpected property '{name}'")
        for name, sub in props.items():
            if name in value:
                errors.extend(validate(value[name], sub, f"{path}.{name}"))
    return errors
