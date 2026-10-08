import json
from pathlib import Path

import pytest

from filteragent.schema import load_schema

SPEC = json.loads((Path(__file__).resolve().parent.parent / "sample_data" / "schema.json").read_text("utf-8"))


@pytest.fixture
def schema():
    return load_schema(SPEC)


def cond(field, op, value):
    return {"field": field, "op": op, "value": value}


def group(*conds, op="and"):
    return {"op": op, "conditions": list(conds)}
