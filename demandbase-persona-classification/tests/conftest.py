import json
from pathlib import Path

import pandas as pd
import pytest

from persona.config import Persona
from persona.llm import ScriptedLLM

DATA = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def personas() -> list[Persona]:
    return [Persona.from_dict(d) for d in json.loads((DATA / "personas.json").read_text(encoding="utf-8"))]


@pytest.fixture
def contacts() -> pd.DataFrame:
    return pd.read_csv(DATA / "contacts.csv", dtype={"title": "string"}, keep_default_na=False)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(
        translations={"Directeur Financier": "Chief Financial Officer"},
        classifications={"chief happiness officer": {"personas": [], "level": "C-Level", "function": "HR",
                                                      "confidence": 0.40}},
    )


def frame(*titles) -> pd.DataFrame:
    return pd.DataFrame({"contact_id": range(len(titles)), "title": list(titles)})
