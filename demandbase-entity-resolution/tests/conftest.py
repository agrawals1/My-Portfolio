import pandas as pd
import pytest


@pytest.fixture
def crm() -> pd.DataFrame:
    return pd.DataFrame({
        "crm_id": ["001A", "001B", "001C", "001D", "001E"],
        "name": ["Acme Corp.", "ACME Corporation", "Société Générale", "Globex", "Initech India Pvt Ltd"],
        "website": ["https://www.acme.com/about", "acme.com", "societegenerale.fr", "gmail.com", "initech.co.in"],
        "country": ["US", "United States", "FR", "US", None],
        "employees": [1200, None, 117000, 40, 300],
    }).assign(country=lambda d: d["country"].where(d["crm_id"] != "001E", "IN"))


@pytest.fixture
def canon() -> pd.DataFrame:
    return pd.DataFrame({
        "company_id": ["C1", "C2", "C3", "C9"],
        "name": ["Acme Corporation", "Societe Generale SA", "Initech Software Services", "Initech Inc"],
        "domain": ["acme.com", "societegenerale.com", "initech.co.in", "initech.com"],
        "country": ["US", "FR", "IN", "US"],
        "parent_id": [None, None, "C9", None],
        "employees": [1150, 117000, 280, 5000],
    })
