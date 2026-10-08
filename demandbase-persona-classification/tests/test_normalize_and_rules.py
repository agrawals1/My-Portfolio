import math

import pytest
from hypothesis import given, strategies as st

from persona.normalize import looks_english, normalize_title
from persona.rules import detect_function, detect_level, keyword_hit


@pytest.mark.parametrize("raw,expected", [
    ("Sr. Director, Demand Generation", "senior director demand generation"),
    ("Chief Happiness Officer 🎉", "chief happiness officer"),
    ("Head of InfoSec & Risk", "head of infosec and risk"),   # '&' is a symbol -> dropped, 'and' stays absent
    ("Directeur  Financier", "directeur financier"),
    ("Responsable Sécurité", "responsable securite"),
    ("マーケティング部長", "マーケティング部長"),              # dakuten is not an accent: must survive
    ("Ｍａｒｋｅｔｉｎｇ", "marketing"),                       # full-width -> ASCII via NFKC
    ("  ", ""), ("🎉🎉", ""), (None, ""), (float("nan"), ""), (42, ""),
])
def test_normalize_title(raw, expected):
    if raw == "Head of InfoSec & Risk":
        expected = "head of infosec risk"
    assert normalize_title(raw) == expected


@given(st.one_of(st.text(), st.none(), st.floats(), st.integers()))
def test_normalize_never_raises_and_is_idempotent(value):
    once = normalize_title(value)
    assert normalize_title(once) == once


@pytest.mark.parametrize("norm,level", [
    ("senior director demand generation", "Director"),
    ("vice president of sales", "VP"),
    ("vp sales", "VP"),
    ("vice president", "VP"),                  # must not be read as C-level via 'president'
    ("president", "C-Level"),
    ("chief financial officer", "C-Level"),
    ("cmo", "C-Level"),
    ("assistant to the ceo", "Staff"),         # trap: contains 'ceo'
    ("marketing intern", "Intern"),
    ("head of infosec", "Director"),
    ("engineering manager", "Manager"),
    ("data analyst", "Staff"),
    ("chief happiness officer", "C-Level"),
    ("", "Unknown"), ("wizard", "Unknown"),
])
def test_detect_level(norm, level):
    assert detect_level(norm) == level


@pytest.mark.parametrize("norm,function", [
    ("senior director demand generation", "Marketing"),
    ("chief financial officer", "Finance"),
    ("head of infosec risk", "IT/Security"),
    ("chief happiness officer", "Unknown"),
    ("finance marketing director", "Unknown"),  # tie between functions -> refuse to guess
    ("hr business partner", "HR"),
])
def test_detect_function(norm, function):
    assert detect_function(norm) == function


@pytest.mark.parametrize("norm,kw,hit", [
    ("senior director demand generation", "demand gen", True),   # prefix match
    ("head of it", "it", True),
    ("white collar crime", "it", False),                         # short keyword = whole token only
    ("chief mission officer", "cmo", False),
    ("cmo emea", "cmo", True),
    ("telemarketing agent", "marketing", False),                 # no mid-word match
    ("marketing manager", "Marketing", True),                    # keyword normalised too
    ("anything", "  ", False),
])
def test_keyword_hit(norm, kw, hit):
    assert keyword_hit(norm, kw) is hit


def test_looks_english():
    assert looks_english("senior marketing manager")
    assert looks_english("chief happiness officer")    # 2/3 known words
    assert not looks_english("directeur financier")
    assert looks_english("")
