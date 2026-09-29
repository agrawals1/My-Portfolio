"""All tunable knobs live here so tests and experiments never touch scoring code."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Weights:
    name: float = 0.65
    country: float = 0.25
    employees: float = 0.10

    def __post_init__(self) -> None:
        total = self.name + self.country + self.employees
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class MatchConfig:
    threshold: float = 0.85  # precision-first: below this we prefer "no match"
    weights: Weights = field(default_factory=Weights)
    unknown_signal: float = 0.5  # neutral value for a missing country / employee count
    fuzzy_cap: float = 0.99  # a fuzzy match may never outrank an exact-domain match (1.0)
    domain_mismatch_penalty: float = 0.08  # both domains known but different
    employee_ratio_cap: float = 10.0  # a 10x size difference => employee signal 0
    prefix_len: int = 4  # name blocking key = first N chars of first name token
    on_tie: Literal["reject", "lowest_id"] = "reject"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.prefix_len < 1:
            raise ValueError("prefix_len must be >= 1")
        if self.employee_ratio_cap <= 1.0:
            raise ValueError("employee_ratio_cap must be > 1")
