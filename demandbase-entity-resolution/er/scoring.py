"""Pairwise scoring. Pure functions on plain records => trivially unit-testable."""
from __future__ import annotations

import math
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from er.config import MatchConfig


@dataclass(frozen=True)
class Record:
    """A normalised company. Missing values are None (never NaN)."""
    id: str
    name: str
    domain: str | None
    country: str | None
    employees: float | None


@dataclass(frozen=True)
class MatchScore:
    score: float
    method: str
    reason: str
    name_sim: float  # kept separately: used to break ties between equal scores


def name_similarity(a: str, b: str) -> float:
    """Mean of Jaro-Winkler (rewards shared prefixes) and token-sort ratio (tolerates word order)."""
    if not a or not b:
        return 0.0
    return 0.5 * JaroWinkler.normalized_similarity(a, b) + 0.5 * fuzz.token_sort_ratio(a, b) / 100


def employee_compat(a: float | None, b: float | None, ratio_cap: float) -> float | None:
    """1.0 for equal sizes, falling linearly (in log space) to 0.0 at a `ratio_cap`x gap.
    None if either side is unknown. log1p keeps 0 employees safe."""
    if a is None or b is None:
        return None
    gap = abs(math.log1p(a) - math.log1p(b))
    return max(0.0, 1.0 - gap / math.log(ratio_cap))


def score_pair(crm: Record, canon: Record, cfg: MatchConfig) -> MatchScore:
    sim = name_similarity(crm.name, canon.name)

    if crm.domain and crm.domain == canon.domain:
        return MatchScore(1.0, "domain_exact", f"normalized domain {crm.domain}", sim)

    country = None if not (crm.country and canon.country) else float(crm.country == canon.country)
    employees = employee_compat(crm.employees, canon.employees, cfg.employee_ratio_cap)
    w, unknown = cfg.weights, cfg.unknown_signal
    raw = (
        w.name * sim
        + w.country * (unknown if country is None else country)
        + w.employees * (unknown if employees is None else employees)
    )
    score = min(raw, cfg.fuzzy_cap)

    parts = [f"name sim {sim:.2f}"]
    parts.append({None: "country unknown", 1.0: "country match", 0.0: "country mismatch"}[country])
    parts.append("employees unknown" if employees is None else f"employees compat {employees:.2f}")
    if crm.domain and canon.domain:  # both known but different
        score -= cfg.domain_mismatch_penalty
        parts.append(f"domain mismatch ({crm.domain} vs {canon.domain})")

    method = "name_fuzzy+country" if country == 1.0 else "name_fuzzy"
    return MatchScore(max(score, 0.0), method, "; ".join(parts), sim)
