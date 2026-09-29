"""Pure, side-effect-free normalisation functions. Every function accepts junk
(None, NaN, numbers, empty strings) and returns None / "" instead of raising."""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

import pandas as pd

FREE_MAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "yahoo.co.uk", "hotmail.com",
    "outlook.com", "live.com", "msn.com", "aol.com", "icloud.com", "me.com", "protonmail.com",
    "proton.me", "gmx.com", "mail.com", "rediffmail.com", "qq.com", "163.com", "126.com",
    "comcast.net", "verizon.net", "att.net", "sbcglobal.net", "btinternet.com",
})

# Minimal public-suffix list. In production use `tldextract` / the Public Suffix List.
MULTI_PART_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "net.in", "org.in", "ac.in", "com.au",
    "net.au", "org.au", "co.jp", "co.nz", "co.za", "com.br", "com.cn", "com.mx", "com.sg",
    "com.hk", "com.tr", "co.kr",
})

LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc",
    "llp", "lp", "plc", "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv", "kk", "pvt", "private",
    "pte", "srl", "spa", "holdings", "holding",
})

_COUNTRY_ALIASES = {
    "US": ("united states", "united states of america", "usa", "america"),
    "GB": ("uk", "united kingdom", "great britain", "england", "gbr"),
    "AE": ("uae", "united arab emirates"),
    "IN": ("india", "ind"),
    "FR": ("france", "fra"),
    "DE": ("germany", "deutschland", "deu"),
    "CA": ("canada", "can"),
    "AU": ("australia", "aus"),
    "SG": ("singapore",),
    "JP": ("japan",),
    "CN": ("china",),
    "NL": ("netherlands", "the netherlands"),
    "IE": ("ireland",),
    "BR": ("brazil",),
    "ES": ("spain",),
    "IT": ("italy",),
    "MX": ("mexico",),
    "CH": ("switzerland",),
    "KR": ("south korea", "korea"),
}
COUNTRY_LOOKUP = {alias: iso for iso, aliases in _COUNTRY_ALIASES.items() for alias in aliases}

_DOMAIN_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")
_STRIP_CHARS_RE = re.compile(r"[.'’`]")  # "S.A." -> "SA", "Levi's" -> "Levis"
_NON_WORD_RE = re.compile(r"[\W_]+")


def normalize_domain(value: object) -> str | None:
    """URL / host / email-ish string -> registrable domain (punycode), else None.

    'https://WWW.Acme.com:443/about' -> 'acme.com'; 'x.initech.co.in' -> 'initech.co.in';
    'gmail.com', 'localhost', '10.0.0.1', '' -> None (free-mail and non-domains are not evidence).
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        host = urlsplit(text if "://" in text else "//" + text).hostname
    except ValueError:  # e.g. unbalanced "[" in the netloc
        return None
    if not host:
        return None
    host = host.rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    try:
        host = host.encode("idna").decode("ascii")  # münchen.de -> xn--mnchen-3ya.de
    except UnicodeError:
        return None
    if not _DOMAIN_RE.match(host):
        return None
    labels = host.split(".")
    if labels[-1].isdigit():  # IPv4 address
        return None
    if ".".join(labels[-2:]) in MULTI_PART_SUFFIXES:
        if len(labels) < 3:  # bare public suffix such as "co.uk"
            return None
        registrable = ".".join(labels[-3:])
    else:
        registrable = ".".join(labels[-2:])
    return None if registrable in FREE_MAIL_DOMAINS else registrable


def normalize_name(value: object) -> str:
    """'Société Générale S.A.' -> 'societe generale'. Returns '' when nothing usable.

    Legal suffixes are removed as whole trailing *tokens* (never substrings), so
    'Corporate Dynamics' is untouched, and the last remaining token is never removed.
    """
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = _STRIP_CHARS_RE.sub("", text.replace("&", " and "))
    tokens = _NON_WORD_RE.sub(" ", text).split()  # \w keeps CJK; emoji/punctuation dropped
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_country(value: object) -> str | None:
    """'United States' / 'USA' / 'us' -> 'US'. Unknown long names -> None."""
    if not isinstance(value, str):
        return None
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", value.casefold())).strip()
    if key in COUNTRY_LOOKUP:
        return COUNTRY_LOOKUP[key]
    return key.upper() if len(key) == 2 else None  # assume ISO alpha-2


def normalize_frame(df: pd.DataFrame, domain_col: str) -> pd.DataFrame:
    """Return a copy with domain_norm, name_norm, country_norm, employees_num columns.

    Row order and a fresh RangeIndex are kept so positions line up with `build_blocks`.
    """
    out = df.reset_index(drop=True).copy()
    out["domain_norm"] = out[domain_col].map(normalize_domain)
    out["name_norm"] = out["name"].map(normalize_name)
    out["country_norm"] = out["country"].map(normalize_country)
    employees = pd.to_numeric(out["employees"], errors="coerce")
    out["employees_num"] = employees.where(employees >= 0)  # negatives -> NaN
    return out
