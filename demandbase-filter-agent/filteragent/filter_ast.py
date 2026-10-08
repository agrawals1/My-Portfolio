"""Immutable filter AST: the ONLY thing the compiler accepts. Built exclusively by validate.parse_filter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

Value = Union[int, float, str, tuple]


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Value

    def to_dict(self) -> dict:
        v = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"field": self.field, "op": self.op, "value": v}


@dataclass(frozen=True)
class Group:
    op: str  # "and" | "or"
    conditions: tuple["Node", ...] = ()

    def to_dict(self) -> dict:
        return {"op": self.op, "conditions": [c.to_dict() for c in self.conditions]}


Node = Union[Condition, Group]
