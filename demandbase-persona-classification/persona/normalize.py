"""Title normalisation. Accepts junk (None, NaN, numbers, emoji-only) and returns "" instead of raising."""
from __future__ import annotations

import re
import unicodedata

_ABBREV = {
    "sr": "senior", "jr": "junior", "mgr": "manager", "dir": "director", "svp": "vp", "evp": "vp",
    "avp": "vp", "exec": "executive", "asst": "assistant", "assoc": "associate", "eng": "engineer",
    "dev": "developer", "ops": "operations", "hr": "hr", "sec": "security",
}
_NON_WORD = re.compile(r"[\W_]+")
STOPWORDS = frozenset({"of", "and", "the", "for", "to", "in", "at", "de", "du", "des", "la", "le", "und", "der", "y"})

# Words that appear in English job titles. Used ONLY to decide "probably not English -> translate first".
# Production: swap for fastText lid-176 / langdetect; the interface (a bool) stays the same.
ENGLISH_VOCAB = frozenset("""
chief officer senior junior director head vice president vp manager lead associate analyst engineer
developer specialist coordinator executive representative consultant assistant intern trainee
marketing demand generation growth brand content product sales revenue finance financial controller
accounting treasury security infosec cyber risk compliance legal counsel it information technology
infrastructure operations supply chain logistics hr human resources people talent recruiting
engineering software data founder owner partner general global regional account customer success
business development strategy digital communications procurement and of the for to
""".split())


def normalize_title(value: object) -> str:
    """'Sr. Director, Demand Generation 🎉' -> 'senior director demand generation'. '' for non-strings/junk."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # Sécurité -> Securite
    text = "".join(" " if unicodedata.category(ch)[0] in "SC" else ch for ch in text)  # emoji, symbols
    tokens = [t for t in _NON_WORD.split(text.casefold()) if t]
    return " ".join(_ABBREV.get(t, t) for t in tokens)


def tokens_of(norm: str) -> list[str]:
    return norm.split() if norm else []


def looks_english(norm: str) -> bool:
    """Cheap heuristic: at least half of the tokens are known English title words."""
    toks = tokens_of(norm)
    if not toks:
        return True
    known = sum(t in ENGLISH_VOCAB or t in _ABBREV.values() for t in toks)
    return known / len(toks) >= 0.5
