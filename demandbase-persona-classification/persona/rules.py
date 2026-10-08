"""Deterministic level / function detection and keyword matching. Pure functions over normalised text."""
from __future__ import annotations

import re
from functools import lru_cache

from persona.normalize import normalize_title

_C_TOKENS = {"ceo", "cfo", "cmo", "cto", "ciso", "cio", "coo", "cro", "chro", "cpo", "clo"}
_STAFF_WORDS = {"analyst", "engineer", "developer", "specialist", "associate", "coordinator", "executive",
                "representative", "consultant", "assistant", "administrator", "accountant", "recruiter"}

FUNCTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Marketing": ("marketing", "demand", "brand", "content", "seo", "growth", "campaign", "cmo", "communications"),
    "Finance": ("finance", "financial", "controller", "accounting", "accountant", "treasury", "tax", "cfo", "audit"),
    "IT/Security": ("security", "infosec", "cyber", "ciso", "it", "information technology", "infrastructure", "cio"),
    "Sales": ("sales", "revenue", "account executive", "bdr", "sdr", "cro", "business development"),
    "HR": ("hr", "human resources", "people", "talent", "recruiting", "recruiter", "chro"),
    "Engineering": ("engineering", "engineer", "developer", "software", "cto"),
    "Operations": ("operations", "supply chain", "logistics", "coo", "procurement"),
    "Legal": ("legal", "counsel", "compliance", "clo"),
    "Product": ("product", "cpo"),
}


@lru_cache(maxsize=4096)
def _kw_pattern(keyword: str) -> re.Pattern[str] | None:
    kw = normalize_title(keyword)
    if not kw:
        return None
    # <=3 chars must be a whole token ("it", "cmo", "hr"); longer ones match as a word PREFIX so
    # "demand gen" hits "demand generation" and "financ" style stems work, without matching mid-word.
    tail = r"(?!\w)" if len(kw) <= 3 else ""
    return re.compile(rf"(?<!\w){re.escape(kw)}{tail}")


def keyword_hit(norm: str, keyword: str) -> bool:
    pat = _kw_pattern(keyword)
    return bool(pat and pat.search(norm))


def detect_level(norm: str) -> str:
    toks = set(norm.split())
    if not toks:
        return "Unknown"
    if toks & {"intern", "internship", "trainee", "apprentice", "student"}:
        return "Intern"
    if re.search(r"\bassistant to\b", norm):  # "Assistant to the CEO" is staff, not C-level
        return "Staff"
    if "vice" in toks or "vp" in toks:
        return "VP"
    if toks & _C_TOKENS or "chief" in toks or toks & {"founder", "cofounder", "owner"} or "president" in toks:
        return "C-Level"
    if toks & {"director", "head"}:
        return "Director"
    if toks & {"manager", "lead", "supervisor"}:
        return "Manager"
    if toks & _STAFF_WORDS:
        return "Staff"
    return "Unknown"


def detect_function(norm: str) -> str:
    """Function with the most keyword hits; a tie between different functions is Unknown (precision first)."""
    scores = {fn: sum(keyword_hit(norm, kw) for kw in kws) for fn, kws in FUNCTION_KEYWORDS.items()}
    best = max(scores.values(), default=0)
    winners = [fn for fn, s in scores.items() if s == best]
    return winners[0] if best > 0 and len(winners) == 1 else "Unknown"
