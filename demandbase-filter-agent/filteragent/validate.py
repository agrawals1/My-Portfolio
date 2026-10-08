"""Strict validation of untrusted filter JSON against the schema. Anything not explicitly allowed is rejected.
Raises ValidationError with a message safe to show to the end user."""
from __future__ import annotations

import difflib
import math

from filteragent.filter_ast import Condition, Group, Node
from filteragent.schema import FieldSpec, Schema

MAX_DEPTH = 3
MAX_CONDITIONS = 20
MAX_VALUES = 50
MAX_STRING = 100


class ValidationError(ValueError):
    def __init__(self, message: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.message, self.code = message, code


def _show(value: object) -> str:
    text = "".join(ch for ch in str(value) if ch.isprintable())[:40]
    return f"'{text}'"


def lookup_field(schema: Schema, name: object) -> FieldSpec:
    if not isinstance(name, str) or name not in schema.fields:
        close = difflib.get_close_matches(str(name), list(schema.fields), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValidationError(f"Unknown field {_show(name)}.{hint}", "unknown_field")
    return schema.fields[name]


def _number(spec: FieldSpec, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{spec.label} needs a number, got {_show(value)}.", "bad_value")
    if spec.min is not None and value < spec.min or spec.max is not None and value > spec.max:
        raise ValidationError(f"{spec.label} must be between {spec.min} and {spec.max}.", "bad_value")
    return value


def _enum(spec: FieldSpec, value: object) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value  # a bare string for "in" is a harmless LLM slip
    if not isinstance(items, list) or not items or len(items) > MAX_VALUES:
        raise ValidationError(f"{spec.label} needs a non-empty list of values.", "bad_value")
    canonical = {v.casefold(): v for v in spec.values}
    out: list[str] = []
    noun = spec.name.split(".")[-1].replace("_", " ")
    for item in items:
        hit = canonical.get(item.strip().casefold()) if isinstance(item, str) else None
        if hit is None:
            raise ValidationError(f"Unknown {noun} {_show(item)}; supported: {', '.join(spec.values)}", "bad_value")
        if hit not in out:
            out.append(hit)
    return tuple(out)


def _string(spec: FieldSpec, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_STRING or "\x00" in value:
        raise ValidationError(f"{spec.label} needs text of 1-{MAX_STRING} characters.", "bad_value")
    return value.strip()


def parse_condition(raw: object, schema: Schema) -> Condition:
    if not isinstance(raw, dict) or set(raw) != {"field", "op", "value"}:
        raise ValidationError("Each condition must have exactly: field, op, value.", "bad_shape")
    spec = lookup_field(schema, raw["field"])
    op = raw["op"]
    if not isinstance(op, str) or op not in spec.ops:
        raise ValidationError(f"Operator {_show(op)} is not supported for {spec.label}; use: {', '.join(spec.ops)}",
                              "bad_op")
    parse = {"number": _number, "enum": _enum, "string": _string}[spec.type]
    return Condition(spec.name, op, parse(spec, raw["value"]))


def _group(raw: object, schema: Schema, depth: int, count: list[int], root: bool) -> Group:
    if not isinstance(raw, dict) or set(raw) != {"op", "conditions"}:
        raise ValidationError("A filter must have exactly: op, conditions.", "bad_shape")
    if raw["op"] not in ("and", "or"):
        raise ValidationError(f"Group operator must be 'and' or 'or', got {_show(raw['op'])}.", "bad_shape")
    items = raw["conditions"]
    if not isinstance(items, list) or (not items and not root):
        raise ValidationError("A group needs a list of conditions.", "bad_shape")
    nodes: list[Node] = []
    for item in items:
        if isinstance(item, dict) and "conditions" in item:
            if depth + 1 > MAX_DEPTH:
                raise ValidationError(f"Filters can be nested at most {MAX_DEPTH} levels deep.", "too_complex")
            nodes.append(_group(item, schema, depth + 1, count, False))
        else:
            count[0] += 1
            if count[0] > MAX_CONDITIONS:
                raise ValidationError(f"Filters can have at most {MAX_CONDITIONS} conditions.", "too_complex")
            nodes.append(parse_condition(item, schema))
    return Group(raw["op"], tuple(nodes))


def parse_filter(raw: object, schema: Schema) -> Group:
    """Untrusted JSON -> immutable AST, or ValidationError. The root group may be empty (= all accounts)."""
    return _group(raw, schema, 1, [0], True)
