"""python -m filteragent.cli   (offline demo with scripted LLM answers)"""
from __future__ import annotations

import json
from pathlib import Path

from filteragent.agent import run_query
from filteragent.demo_llm import DEMO_LLM, FOLLOWUP, Q1, Q2, Q3, Q4
from filteragent.schema import load_schema


def main() -> None:
    spec = json.loads((Path(__file__).resolve().parent.parent / "sample_data" / "schema.json").read_text("utf-8"))
    schema = load_schema(spec)
    prev = None
    for q in (Q1, Q2, Q3, Q4):
        r = run_query(q, schema, DEMO_LLM, "tenant_acme")
        print(f"\n> {q}")
        if not r.ok:
            print("  ERROR:", r.error)
            continue
        print("  " + "\n  ".join(r.explanation or ("(no conditions)",)))
        print("  SQL:", r.sql, "\n  params:", r.params)
        if q == Q2:
            prev = r.filter
    r = run_query(FOLLOWUP, schema, DEMO_LLM, "tenant_acme", previous=prev)
    print(f"\n> follow-up: {FOLLOWUP}\n  diff:", json.dumps(r.diff.to_dict()), "\n  SQL:", r.sql)


if __name__ == "__main__":
    main()
