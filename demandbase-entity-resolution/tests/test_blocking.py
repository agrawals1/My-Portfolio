import pandas as pd

from er.blocking import block_keys, build_blocks, candidate_positions
from er.normalize import normalize_frame


def test_block_keys():
    assert block_keys("acme.com", "acme labs", 4) == [("d", "acme.com"), ("p", "acme")]
    assert block_keys(None, "ab", 4) == [("p", "ab")]
    assert block_keys(None, "", 4) == []


def _synthetic(n: int) -> pd.DataFrame:
    # distinct 4-letter first tokens so blocks stay small
    names = [f"{chr(97 + i // 676 % 26)}{chr(97 + i // 26 % 26)}{chr(97 + i % 26)}x Holdings" for i in range(n)]
    return pd.DataFrame({
        "name": names, "domain": [f"{n.split()[0]}.com" for n in names],
        "country": "US", "employees": 100,
    })


def test_blocking_is_far_smaller_than_cross_join_and_keeps_true_match():
    canon = normalize_frame(_synthetic(300), "domain")
    crm = _synthetic(300).assign(website=lambda d: d["domain"]).drop(columns="domain")
    crm = normalize_frame(crm, "website")
    blocks = build_blocks(canon)

    total = 0
    for pos, (d, n) in enumerate(zip(crm["domain_norm"], crm["name_norm"])):
        cands = candidate_positions(block_keys(d, n, 4), blocks)
        assert pos in cands  # the true match is never blocked out
        total += len(cands)
    assert total < 0.02 * len(crm) * len(canon)


def test_candidate_positions_sorted_and_deduplicated():
    blocks = {("d", "a.com"): [5, 1], ("p", "acme"): [1, 3]}
    assert candidate_positions([("d", "a.com"), ("p", "acme"), ("p", "zzzz")], blocks) == [1, 3, 5]
