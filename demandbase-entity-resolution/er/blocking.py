"""Candidate generation. Blocking keeps matching ~O(n + m) instead of the O(n*m) cross join.

Two key types (a record can emit both):
  ("d", registrable_domain)      -> exact-domain block
  ("p", first_name_token[:N])    -> name-prefix block (tolerates typos after N chars)
Country is deliberately *not* part of the key: a missing country would otherwise cause
false negatives. It is used as a scoring signal instead.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from er.config import MatchConfig

BlockKey = tuple[str, str]


def block_keys(domain: str | None, name: str, prefix_len: int) -> list[BlockKey]:
    keys: list[BlockKey] = []
    if isinstance(domain, str) and domain:
        keys.append(("d", domain))
    if name:
        keys.append(("p", name.split()[0][:prefix_len]))
    return keys


def build_blocks(df: pd.DataFrame, cfg: MatchConfig | None = None) -> dict[BlockKey, list[int]]:
    """Index a *normalised* frame: block key -> row positions."""
    cfg = cfg or MatchConfig()
    blocks: dict[BlockKey, list[int]] = defaultdict(list)
    for pos, (domain, name) in enumerate(zip(df["domain_norm"], df["name_norm"])):
        for key in block_keys(domain, name, cfg.prefix_len):
            blocks[key].append(pos)
    return dict(blocks)


def candidate_positions(keys: list[BlockKey], blocks: dict[BlockKey, list[int]]) -> list[int]:
    """Union of the matching blocks, sorted so results never depend on set ordering."""
    found: set[int] = set()
    for key in keys:
        found.update(blocks.get(key, ()))
    return sorted(found)
