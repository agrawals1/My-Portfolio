"""LLM fallback boundary: a Protocol, strict validation of whatever comes back, and a scripted fake.
Never trust model output: unknown persona ids, out-of-vocabulary levels and non-numeric confidences are rejected."""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from persona.config import FUNCTIONS, LEVELS


class LLMClient(Protocol):
    def translate(self, titles: Sequence[str]) -> dict[str, str]:
        """title -> English title (missing key = could not translate)."""
        ...

    def classify(self, titles: Sequence[str], personas: Sequence[dict]) -> dict[str, dict]:
        """title -> {"personas": [ids], "level": str, "function": str, "confidence": float}."""
        ...


def validate_llm_result(raw: object, valid_ids: set[str]) -> dict | None:
    """Return a clean result dict or None if the payload is unusable (caller records method='llm_invalid')."""
    if not isinstance(raw, dict):
        return None
    conf = raw.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf):
        return None
    ids = raw.get("personas")
    if not isinstance(ids, list):
        return None
    level, function = raw.get("level"), raw.get("function")
    return {
        "personas": sorted({i for i in ids if isinstance(i, str) and i in valid_ids}),  # drop hallucinated ids
        "level": level if level in LEVELS else "Unknown",
        "function": function if function in FUNCTIONS else "Unknown",
        "confidence": min(1.0, max(0.0, float(conf))),
    }


class ScriptedLLM:
    """Deterministic stand-in for tests and the offline demo; records calls so tests can assert on cost."""

    def __init__(self, translations: dict[str, str] | None = None, classifications: dict[str, dict] | None = None):
        self.translations = translations or {}
        self.classifications = classifications or {}
        self.translate_calls: list[list[str]] = []
        self.classify_calls: list[list[str]] = []

    def translate(self, titles: Sequence[str]) -> dict[str, str]:
        self.translate_calls.append(list(titles))
        return {t: self.translations[t] for t in titles if t in self.translations}

    def classify(self, titles: Sequence[str], personas: Sequence[dict]) -> dict[str, dict]:
        self.classify_calls.append(list(titles))
        return {t: self.classifications[t] for t in titles if t in self.classifications}
