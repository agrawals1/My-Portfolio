import math

import pytest

from er.config import MatchConfig, Weights
from er.scoring import Record, employee_compat, name_similarity, score_pair

CFG = MatchConfig()


def rec(name="acme", domain=None, country=None, employees=None, id="x"):
    return Record(id, name, domain, country, employees)


def test_domain_exact_scores_one_even_if_names_differ():
    s = score_pair(rec("foo", "acme.com"), rec("bar", "acme.com"), CFG)
    assert (s.score, s.method) == (1.0, "domain_exact")


def test_fuzzy_never_reaches_domain_exact_score():
    s = score_pair(rec("acme", None, "US", 100), rec("acme", None, "US", 100), CFG)
    assert s.score == CFG.fuzzy_cap < 1.0


def test_domain_mismatch_is_penalised_and_explained():
    same = score_pair(rec("societe generale", None, "FR", 5), rec("societe generale", None, "FR", 5), CFG)
    diff = score_pair(rec("societe generale", "sg.fr", "FR", 5), rec("societe generale", "sg.com", "FR", 5), CFG)
    assert diff.score == pytest.approx(same.score - CFG.domain_mismatch_penalty)
    assert "domain mismatch" in diff.reason


def test_country_mismatch_pushes_identical_names_below_threshold():
    s = score_pair(rec("acme", None, "US", 100), rec("acme", None, "DE", 100), CFG)
    assert s.score < CFG.threshold and "country mismatch" in s.reason


def test_missing_signals_are_neutral_not_fatal():
    s = score_pair(rec("acme"), rec("acme"), CFG)
    w = CFG.weights
    assert s.score == pytest.approx(w.name + 0.5 * w.country + 0.5 * w.employees)
    assert s.score < CFG.threshold  # a bare name match is not enough evidence
    assert "country unknown" in s.reason and "employees unknown" in s.reason


def test_empty_names_score_zero_similarity():
    assert name_similarity("", "acme") == 0.0
    assert score_pair(rec(""), rec(""), CFG).score < CFG.threshold


def test_name_similarity_tolerates_word_order():
    reordered = name_similarity("generale societe", "societe generale")
    assert reordered > 0.75 > name_similarity("acme widgets", "zenith rockets")


@pytest.mark.parametrize("a, b, expected", [
    (100, 100, 1.0), (100, 1e6, 0.0), (1e6, 100, 0.0), (0, 0, 1.0), (None, 5, None), (5, None, None),
])
def test_employee_compat(a, b, expected):
    assert employee_compat(a, b, 10.0) == expected


def test_employee_compat_is_symmetric_and_decreasing():
    assert employee_compat(100, 300, 10.0) == employee_compat(300, 100, 10.0)
    assert employee_compat(100, 150, 10.0) > employee_compat(100, 500, 10.0) > 0.0


def test_employee_compat_zero_vs_positive_is_finite_and_bounded():
    v = employee_compat(0, 500, 10.0)
    assert v is not None and 0.0 <= v <= 1.0 and not math.isnan(v)


def test_config_validation():
    with pytest.raises(ValueError):
        Weights(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        MatchConfig(threshold=1.5)
