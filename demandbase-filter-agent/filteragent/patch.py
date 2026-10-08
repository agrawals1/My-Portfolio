"""Follow-up turns: the model proposes a small PATCH, code applies it. The model never rewrites the whole filter,
so it cannot silently drop or alter conditions the user did not mention."""
from __future__ import annotations

from dataclasses import dataclass

from filteragent.filter_ast import Condition, Group, Node
from filteragent.validate import ValidationError, lookup_field, parse_condition, parse_filter

MAX_ACTIONS = 10


@dataclass(frozen=True)
class Diff:
    added: tuple[Node, ...]
    removed: tuple[Node, ...]

    def to_dict(self) -> dict:
        return {"added": [n.to_dict() for n in self.added], "removed": [n.to_dict() for n in self.removed]}


def apply_patch(previous: Group, raw: object, schema) -> Group:
    if not isinstance(raw, dict) or set(raw) != {"actions"} or not isinstance(raw["actions"], list):
        raise ValidationError("A follow-up must be a list of actions.", "bad_shape")
    if not 1 <= len(raw["actions"]) <= MAX_ACTIONS:
        raise ValidationError(f"A follow-up can have 1-{MAX_ACTIONS} actions.", "bad_shape")
    nodes = list(previous.conditions)
    for act in raw["actions"]:
        kind = act.get("action") if isinstance(act, dict) else None
        if kind == "remove" and set(act) == {"action", "field"}:
            spec = lookup_field(schema, act["field"])
            kept = [n for n in nodes if not (isinstance(n, Condition) and n.field == spec.name)]
            if len(kept) == len(nodes):
                raise ValidationError(f"There is no {spec.label} condition to remove.", "nothing_to_remove")
            nodes = kept
        elif kind in ("add", "set") and set(act) == {"action", "condition"}:
            cond = parse_condition(act["condition"], schema)
            if kind == "set":
                nodes = [n for n in nodes if not (isinstance(n, Condition) and n.field == cond.field)]
            nodes.append(cond)
        else:
            raise ValidationError("Each follow-up action must be remove (field), add (condition) or set (condition).",
                                  "bad_shape")
    return parse_filter(Group(previous.op, tuple(nodes)).to_dict(), schema)  # re-validate size/depth limits


def diff_filters(old: Group, new: Group) -> Diff:
    return Diff(added=tuple(n for n in new.conditions if n not in old.conditions),
                removed=tuple(n for n in old.conditions if n not in new.conditions))
