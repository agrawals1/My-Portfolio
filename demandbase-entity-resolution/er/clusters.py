"""Duplicate detection among CRM rows via union-find."""
from __future__ import annotations

import pandas as pd


class UnionFind:
    """Disjoint sets with path compression and union by size."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # compress the path
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]


def find_duplicates(resolved: pd.DataFrame) -> pd.DataFrame:
    """Cluster CRM rows that resolved to the same company_id.

    Returns one row per cluster of size > 1: company_id, crm_ids (sorted), size.
    """
    matched = resolved.dropna(subset=["company_id"])
    uf = UnionFind()
    first_seen: dict[str, str] = {}
    for crm_id, company_id in zip(matched["crm_id"], matched["company_id"]):
        uf.find(crm_id)
        uf.union(crm_id, first_seen.setdefault(company_id, crm_id))
    members: dict[str, list[str]] = {}
    for crm_id in matched["crm_id"]:
        members.setdefault(uf.find(crm_id), []).append(crm_id)
    company_of = dict(zip(matched["crm_id"], matched["company_id"]))
    rows = [
        {"company_id": company_of[ids[0]], "crm_ids": sorted(ids), "size": len(ids)}
        for ids in members.values()
        if len(ids) > 1
    ]
    out = pd.DataFrame(rows, columns=["company_id", "crm_ids", "size"])
    return out.sort_values("company_id").reset_index(drop=True)
