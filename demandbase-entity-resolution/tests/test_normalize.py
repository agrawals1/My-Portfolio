import numpy as np
import pandas as pd
import pytest

from er.normalize import normalize_country, normalize_domain, normalize_frame, normalize_name


@pytest.mark.parametrize("raw, expected", [
    ("https://www.acme.com/about", "acme.com"),
    ("WWW.ACME.COM:443", "acme.com"),
    ("acme.com/path?q=1#frag", "acme.com"),
    ("http://user:pw@shop.acme.com:8080/x", "acme.com"),
    ("initech.co.in", "initech.co.in"),      # NOT "co.in"
    ("www.bbc.co.uk", "bbc.co.uk"),
    ("sub.deep.initech.co.in", "initech.co.in"),
    ("münchen.de", "xn--mnchen-3ya.de"),      # IDN -> punycode
    ("xn--mnchen-3ya.de", "xn--mnchen-3ya.de"),
    ("jane@acme.com", "acme.com"),            # email-ish input
    ("gmail.com", None),                       # free mail is not evidence
    ("someone@Yahoo.com", None),
    ("", None), ("   ", None), (None, None), (np.nan, None), (pd.NA, None), (42, None),
    ("localhost", None), ("10.0.0.1", None), ("co.uk", None), ("not a domain", None),
    ("http://[bad", None),
])
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Société Générale S.A.", "societe generale"),
    ("Acme Corp.", "acme"),
    ("ACME Corporation", "acme"),
    ("Acme Corp Ltd", "acme"),                     # stacked suffixes
    ("Corporate Dynamics", "corporate dynamics"),  # substring replace would give "orate dynamics"
    ("Incredible Ltd", "incredible"),              # "inc" is only a suffix as a whole token
    ("Inc", "inc"),                                # never strip the last token
    ("Initech India Pvt Ltd", "initech india"),
    ("The Home Depot, Inc.", "home depot"),
    ("Johnson & Johnson", "johnson and johnson"),
    ("Levi's", "levis"),
    ("  ACME   —  Labs  ", "acme labs"),
    ("🚀 Rocket Inc", "rocket"),
    ("株式会社 トヨタ", "株式会社 トヨタ"),         # CJK survives (no crash, no data loss)
    ("", ""), (None, ""), (np.nan, ""), (pd.NA, ""), (123, ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("United States", "US"), ("USA", "US"), ("U.S.A.", "US"), ("us", "US"), (" US ", "US"),
    ("UK", "GB"), ("United Kingdom", "GB"), ("India", "IN"), ("fr", "FR"),
    ("Narnia", None), ("", None), (None, None), (np.nan, None),
])
def test_normalize_country(raw, expected):
    assert normalize_country(raw) == expected


def test_normalize_frame_handles_nan_negative_and_text_employees():
    df = pd.DataFrame({
        "name": ["A Inc", None], "website": ["a.com", None], "country": ["USA", None],
        "employees": ["10", "-5"],
    }, index=[7, 9])
    out = normalize_frame(df, "website")
    assert list(out.index) == [0, 1]  # fresh index so positions line up with blocks
    assert out["name_norm"].tolist() == ["a", ""]
    assert out["employees_num"].iloc[0] == 10 and np.isnan(out["employees_num"].iloc[1])


def test_normalizers_are_idempotent_and_never_raise():
    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(st.text())
    def check(text):
        assert normalize_name(normalize_name(text)) == normalize_name(text)
        once = normalize_domain(text)
        assert normalize_domain(once) == once
        normalize_country(text)

    check()
