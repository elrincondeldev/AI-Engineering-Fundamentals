"""The JSON Schema subset validator."""

from __future__ import annotations

from m07_agents_tools.schema import validate
from m07_agents_tools.tools import build_toolbox

RETURN_SCHEMA = build_toolbox()["create_return"].parameters


def test_valid_arguments_have_no_errors() -> None:
    args = {"order_id": "HW-310042", "sku": "HW-EMB-T3", "quantity": 1, "reason": "changed_mind"}
    assert validate(args, RETURN_SCHEMA) == []


def test_missing_required_and_unexpected_properties() -> None:
    errors = validate({"order_id": "HW-310042", "colour": "red"}, RETURN_SCHEMA)
    assert "$: missing required property 'sku'" in errors
    assert "$: unexpected property 'colour'" in errors


def test_pattern_enum_and_range() -> None:
    args = {"order_id": "310042", "sku": "HW-XXX", "quantity": 0, "reason": "changed_mind"}
    errors = validate(args, RETURN_SCHEMA)
    assert any("does not match pattern" in e for e in errors)
    assert any("is not one of" in e for e in errors)
    assert any("below the minimum 1" in e for e in errors)


def test_booleans_are_not_integers() -> None:
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got bool"]
    assert validate(3, {"type": "number"}) == []


def test_strings_lengths_and_arrays() -> None:
    schema = {"type": "array", "items": {"type": "string", "minLength": 3, "maxLength": 5}}
    assert validate(["abcd"], schema) == []
    assert validate(["ab", "abcdef"], schema) == [
        "$[0]: shorter than 3 characters",
        "$[1]: longer than 5 characters",
    ]
    assert validate("not a list", schema) == ["$: expected array, got str"]
