"""python -m persona.cli --contacts sample_data/contacts.csv --personas sample_data/personas.json"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from persona.classify import classify_titles
from persona.config import Persona
from persona.demo_llm import DEMO_LLM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--personas", required=True)
    ap.add_argument("--no-llm", action="store_true", help="rules + embeddings only")
    args = ap.parse_args()
    with open(args.personas, encoding="utf-8") as fh:
        personas = [Persona.from_dict(d) for d in json.load(fh)]
    contacts = pd.read_csv(args.contacts, dtype={"title": "string"}, keep_default_na=False)
    out = classify_titles(contacts, personas, llm=None if args.no_llm else DEMO_LLM)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(out.drop(columns="reason").to_string(index=False))


if __name__ == "__main__":
    main()
