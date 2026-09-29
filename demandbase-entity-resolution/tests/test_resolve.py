import pandas as pd
import pytest

from er.clusters import UnionFind, find_duplicates
from er.config import MatchConfig
from er.resolve import resolve, ultimate_parent


def by_id(result: pd.DataFrame) -> dict:
    return result.set_index("crm_id").to_dict("index")


def test_sample_end_to_end(crm, canon):
    r = by_id(resolve(crm, canon))
    assert r["001A"]["company_id"] == "C1" and r["001A"]["method"] == "domain_exact"
    assert r["001B"]["company_id"] == "C1"
    assert r["001C"]["company_id"] == "C2" and r["001C"]["method"] == "name_fuzzy+country"
    assert r["001C"]["score"] == pytest.approx(0.91)
    assert pd.isna(r["001D"]["company_id"]) and r["001D"]["method"] == "rejected"
    assert r["001E"]["company_id"] == "C3" and r["001E"]["ultimate_parent_id"] == "C9"


def test_every_crm_id_appears_exactly_once_in_input_order(crm, canon):
    out = resolve(crm, canon)
    assert out["crm_id"].tolist() == crm["crm_id"].tolist()


def test_deterministic_under_row_shuffling(crm, canon):
    def run(c, k):
        return resolve(c, k).sort_values("crm_id").reset_index(drop=True)

    baseline = run(crm, canon)
    for seed in range(5):
        shuffled = run(crm.sample(frac=1, random_state=seed), canon.sample(frac=1, random_state=seed))
        pd.testing.assert_frame_equal(shuffled, baseline)


def test_free_mail_never_matches_on_domain():
    crm = pd.DataFrame({"crm_id": ["1"], "name": ["Acme"], "website": ["gmail.com"]})
    canon = pd.DataFrame({"company_id": ["G"], "name": ["Gmail Holdings"], "domain": ["gmail.com"]})
    out = resolve(crm, canon)
    assert out.loc[0, "method"] == "rejected"


def test_name_only_match_below_threshold_is_rejected_with_explanation(canon):
    crm = pd.DataFrame({"crm_id": ["1"], "name": ["Acme Corp"], "website": [None], "country": ["DE"]})
    out = resolve(crm, canon)
    assert out.loc[0, "method"] == "rejected" and "C1" in out.loc[0, "reason"]


def test_subsidiary_sharing_parent_domain_resolves_by_name_similarity():
    canon = pd.DataFrame({
        "company_id": ["P", "S"], "name": ["Acme Corporation", "Acme Europe Ltd"],
        "domain": ["acme.com", "acme.com"], "parent_id": [None, "P"],
    })
    crm = pd.DataFrame({"crm_id": ["1", "2"], "name": ["Acme Corp", "Acme Europe"], "website": ["acme.com"] * 2})
    out = by_id(resolve(crm, canon))
    assert out["1"]["company_id"] == "P" and out["2"]["company_id"] == "S"
    assert out["2"]["ultimate_parent_id"] == "P"


def test_exact_ties_are_rejected_by_default_and_deterministic_when_configured():
    canon = pd.DataFrame({"company_id": ["B", "A"], "name": ["Acme", "Acme"], "domain": ["acme.com"] * 2})
    crm = pd.DataFrame({"crm_id": ["1"], "name": ["Acme"], "website": ["acme.com"]})
    assert resolve(crm, canon).loc[0, "method"] == "rejected"
    picked = resolve(crm, canon, MatchConfig(on_tie="lowest_id"))
    assert picked.loc[0, "company_id"] == "A"


def test_different_country_tld_matches_via_name_and_country(crm, canon):
    out = by_id(resolve(crm, canon))["001C"]
    assert "domain mismatch" in out["reason"]


def test_duplicates_report_and_union_find(crm, canon):
    dups = find_duplicates(resolve(crm, canon))
    assert dups.to_dict("records") == [{"company_id": "C1", "crm_ids": ["001A", "001B"], "size": 2}]
    uf = UnionFind()
    uf.union("a", "b"); uf.union("b", "c")
    assert uf.find("a") == uf.find("c") != uf.find("d")


def test_no_duplicates_gives_empty_frame():
    resolved = pd.DataFrame({"crm_id": ["1", "2"], "company_id": ["A", None]})
    assert find_duplicates(resolved).empty


@pytest.mark.parametrize("start, expected", [("C3", "C9"), ("C9", "C9"), ("D", "D")])
def test_ultimate_parent(start, expected):
    parents = {"C3": "C9", "C9": None, "D": "MISSING"}  # D has a dangling parent
    assert ultimate_parent(start, parents) == expected


def test_ultimate_parent_cycle_is_stable():
    parents = {"a": "b", "b": "c", "c": "b", "t": "a", "self": "self"}
    assert {ultimate_parent(x, parents) for x in ("a", "b", "c", "t")} == {"b"}
    assert ultimate_parent("self", parents) == "self"


def test_empty_inputs_and_missing_optional_columns():
    crm = pd.DataFrame({"crm_id": [], "name": [], "website": []})
    canon = pd.DataFrame({"company_id": ["A"], "name": ["Acme"], "domain": ["acme.com"]})
    out = resolve(crm, canon)
    assert out.empty and list(out.columns)[:2] == ["crm_id", "company_id"]
    crm2 = pd.DataFrame({"crm_id": ["1"], "name": ["Acme"], "website": ["acme.com"]})
    assert resolve(crm2, canon).loc[0, "company_id"] == "A"


def test_bad_input_raises_clear_errors(crm, canon):
    with pytest.raises(ValueError, match="missing required columns"):
        resolve(crm.drop(columns="website"), canon)
    with pytest.raises(ValueError, match="must be unique"):
        resolve(pd.concat([crm, crm]), canon)
    with pytest.raises(ValueError, match="nulls"):
        resolve(crm.assign(crm_id=[None] * len(crm)), canon)
