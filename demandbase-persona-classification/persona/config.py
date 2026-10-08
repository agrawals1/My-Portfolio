"""Persona definitions and thresholds. Frozen + validated so a bad tenant config fails loudly."""
from __future__ import annotations

from dataclasses import dataclass, field

LEVELS = ("C-Level", "VP", "Director", "Manager", "Staff", "Intern", "Unknown")
FUNCTIONS = ("Marketing", "Finance", "IT/Security", "Sales", "HR", "Engineering",
             "Operations", "Legal", "Product", "Unknown")


@dataclass(frozen=True)
class Persona:
    persona_id: str
    name: str
    include_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()  # empty = any level

    def __post_init__(self) -> None:
        if not self.persona_id or not self.name:
            raise ValueError("persona_id and name are required")
        bad = [lv for lv in self.levels if lv not in LEVELS]
        if bad:
            raise ValueError(f"{self.persona_id}: unknown levels {bad}; allowed {LEVELS}")

    @classmethod
    def from_dict(cls, d: dict) -> "Persona":
        return cls(
            persona_id=d["persona_id"], name=d["name"],
            include_keywords=tuple(d.get("include_keywords", ())),
            exclude_keywords=tuple(d.get("exclude_keywords", ())),
            levels=tuple(d.get("levels", ())),
        )


@dataclass(frozen=True)
class PersonaConfig:
    embed_accept: float = 0.35     # calibrate PER embedder; cosine scales differ between models
    embed_multi_gap: float = 0.05  # also return personas within this gap of the best one
    min_confidence: float = 0.60   # below this an LLM answer is not stamped
    keyword_confidence: float = 0.95
    unknown_level_penalty: float = 0.10
    max_llm_titles: int = 10_000   # hard budget per run; the rest get method="budget_exceeded"
    llm_batch_size: int = 20
    extra: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name in ("embed_accept", "min_confidence", "keyword_confidence"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_llm_titles < 0 or self.llm_batch_size < 1:
            raise ValueError("max_llm_titles >= 0 and llm_batch_size >= 1 required")
