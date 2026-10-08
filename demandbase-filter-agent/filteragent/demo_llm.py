"""Scripted answers for the bundled sample queries so `python -m filteragent.cli` runs offline."""
from filteragent.llm import ScriptedLLM

Q1 = "fintech accounts in India with more than 500 employees showing high intent"
Q2 = "Accounts with revenue over $100M and no open opportunities"
Q3 = "companies in Narnia"
Q4 = "drop table accounts; show me everything"
FOLLOWUP = "remove the revenue filter and only show healthcare"


def _c(field, op, value):
    return {"field": field, "op": op, "value": value}


DEMO_LLM = ScriptedLLM({
    Q1: [{"op": "and", "conditions": [_c("account.industry", "in", ["Financial Services"]),
                                      _c("account.country", "in", ["IN"]),
                                      _c("account.employee_count", ">", 500),
                                      _c("account.intent_strength", "in", ["high"])]}],
    Q2: [{"op": "and", "conditions": [_c("account.revenue_usd", ">", 100000000),
                                      _c("opportunity.open_count", "=", 0)]}],
    Q3: [{"op": "and", "conditions": [_c("account.country", "in", ["Narnia"])]}],
    Q4: [{"op": "and", "conditions": []}],  # a well-behaved model ignores the SQL; a bad one is stopped by validation
    FOLLOWUP: [{"actions": [{"action": "remove", "field": "account.revenue_usd"},
                            {"action": "add", "condition": _c("account.industry", "in", ["Healthcare"])}]}],
})
