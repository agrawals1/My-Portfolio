"""Orchestration: validate -> normalise -> block -> score -> decide."""
from __future__ import annotations

import logging
import math

import pandas as pd

from er.blocking import block_keys, build_blocks, candidate_positions
from er.config import MatchConfig
from er.normalize import normalize_frame
from er.scoring import MatchScore, Record, score_pair

logger = logging.getLogger(__name__)

CRM_REQUIRED = ("crm_id", "name", "website")
CANON_REQUIRED = ("company_id", "name", "domain")
OPTIONAL = ("country", "employees")
RESULT_COLUMNS = ["crm_id", "company_id", "score", "method", "reason", "ultimate_parent_id"]


def _validate(df: pd.DataFrame, required: tuple[str, ...], id_col: str, label: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    df = df.copy()
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = pd.NA
    if df[id_col].isna().any():
        raise ValueError(f"{label}.{id_col} contains nulls")
    df[id_col] = df[id_col].astype(str)
    if df[id_col].duplicated().any():
        dupes = df.loc[df[id_col].duplicated(), id_col].unique().tolist()[:5]
        raise ValueError(f"{label}.{id_col} must be unique; duplicates e.g. {dupes}")
    return df


def _to_records(df: pd.DataFrame, id_col: str) -> list[Record]:
    def text(x: object) -> str | None:
        return x if isinstance(x, str) and x else None

    def number(x: object) -> float | None:
        return None if pd.isna(x) else float(x)

    return [
        Record(i, n or "", text(d), text(c), number(e))
        for i, n, d, c, e in zip(
            df[id_col], df["name_norm"], df["domain_norm"], df["country_norm"], df["employees_num"]
        )
    ]


def ultimate_parent(company_id: str, parent_of: dict[str, str | None]) -> str:
    """Follow parent links to the root. Cycles resolve to the smallest id in the cycle;
    a dangling parent_id (not in the canonical table) stops the walk."""
    path = [company_id]
    while (parent := parent_of.get(path[-1])) is not None and parent in parent_of:
        if parent in path:
            return min(path[path.index(parent):])
        path.append(parent)
    return path[-1]


def _reject(crm_id: str, reason: str) -> dict:
    return {"crm_id": crm_id, "company_id": None, "score": 0.0, "method": "rejected", "reason": reason}


def _decide(crm: Record, candidates: list[Record], cfg: MatchConfig) -> dict:
    prefix = "" if crm.domain else "no usable domain; "
    if not candidates:
        return _reject(crm.id, prefix + "no candidate companies")

    scored: list[tuple[MatchScore, Record]] = sorted(
        ((score_pair(crm, c, cfg), c) for c in candidates),
        key=lambda sc: (-sc[0].score, -sc[0].name_sim, sc[1].id),  # deterministic order
    )
    best, best_company = scored[0]
    if best.score < cfg.threshold:
        return _reject(
            crm.id, f"{prefix}best candidate {best_company.id} scored {best.score:.2f} < {cfg.threshold}"
        )

    tied = [
        c.id for s, c in scored
        if math.isclose(s.score, best.score) and math.isclose(s.name_sim, best.name_sim)
    ]
    if len(tied) > 1 and cfg.on_tie == "reject":
        return _reject(crm.id, f"tie at {best.score:.2f} between {tied}; refusing to guess")

    return {
        "crm_id": crm.id,
        "company_id": best_company.id,
        "score": round(best.score, 4),
        "method": best.method,
        "reason": best.reason,
    }


def resolve(crm: pd.DataFrame, canon: pd.DataFrame, cfg: MatchConfig | None = None) -> pd.DataFrame:
    """One output row per CRM row, in input order."""
    cfg = cfg or MatchConfig()
    crm = _validate(crm, CRM_REQUIRED, "crm_id", "crm")
    canon = _validate(canon, CANON_REQUIRED, "company_id", "canonical")

    crm_n = normalize_frame(crm, domain_col="website")
    canon_n = normalize_frame(canon, domain_col="domain")
    blocks = build_blocks(canon_n, cfg)
    canon_records = _to_records(canon_n, "company_id")
    crm_records = _to_records(crm_n, "crm_id")

    parent_col = canon["parent_id"] if "parent_id" in canon.columns else pd.Series(pd.NA, index=canon.index)
    parent_of = {
        cid: (str(p) if pd.notna(p) else None) for cid, p in zip(canon["company_id"], parent_col)
    }

    rows, comparisons = [], 0
    for rec in crm_records:
        positions = candidate_positions(block_keys(rec.domain, rec.name, cfg.prefix_len), blocks)
        comparisons += len(positions)
        rows.append(_decide(rec, [canon_records[p] for p in positions], cfg))

    out = pd.DataFrame(rows, columns=RESULT_COLUMNS[:-1])  # explicit columns: works for 0 rows
    out["ultimate_parent_id"] = out["company_id"].map(
        lambda cid: ultimate_parent(cid, parent_of) if isinstance(cid, str) else None
    )
    logger.info(
        "resolved %d CRM rows: %d matched, %d pair comparisons (full cross join would be %d)",
        len(out), out["company_id"].notna().sum(), comparisons, len(crm) * len(canon),
    )
    return out[RESULT_COLUMNS]
