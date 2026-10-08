import re

import pytest
from hypothesis import given, settings, strategies as st

from filteragent.compile import compile_filter
from filteragent.schema import load_schema
from filteragent.validate import parse_filter
from tests.conftest import SPEC, cond, group


def compiled(schema, raw, tenant="t1", **kw):
    return compile_filter(parse_filter(raw, schema), schema, tenant, **kw)


def test_query1_sql_and_params(schema):
    q = compiled(schema, group(cond("account.industry", "in", ["Financial Services"]),
                               cond("account.country", "in", ["IN"]),
                               cond("account.employee_count", ">", 500),
                               cond("account.intent_strength", "in", ["high"])))
    assert q.sql == (
        "SELECT a.account_id FROM accounts a WHERE a.tenant_id = %(tenant_id)s AND (a.industry IN (%(p0)s) "
        "AND a.country IN (%(p1)s) AND a.employees > %(p2)s AND EXISTS (SELECT 1 FROM intent_scores x0 "
        "WHERE x0.account_id = a.account_id AND x0.tenant_id = %(tenant_id)s AND x0.strength IN (%(p3)s))) "
        "ORDER BY a.account_id LIMIT %(limit)s")
    assert q.params == {"tenant_id": "t1", "limit": 1000, "p0": "Financial Services", "p1": "IN", "p2": 500, "p3": "high"}


def test_no_open_opportunities_uses_a_tenant_scoped_count(schema):
    q = compiled(schema, group(cond("opportunity.open_count", "=", 0)))
    assert "(SELECT COUNT(*) FROM opportunities x0" in q.sql and "x0.tenant_id = %(tenant_id)s" in q.sql
    assert "x0.status = %(p0)s" in q.sql and q.params["p0"] == "open" and q.params["p1"] == 0


def test_empty_filter_is_tenant_scoped_select_all(schema):
    q = compiled(schema, group())
    assert q.sql.endswith("WHERE a.tenant_id = %(tenant_id)s ORDER BY a.account_id LIMIT %(limit)s")


def test_nested_or_group_is_parenthesised(schema):
    q = compiled(schema, group(cond("account.employee_count", ">", 5),
                               group(cond("account.country", "in", ["US"]), cond("account.industry", "in", ["Retail"]),
                                     op="or")))
    assert "a.employees > %(p0)s AND (a.country IN (%(p1)s) OR a.industry IN (%(p2)s))" in q.sql


def test_not_in_keeps_unknown_values_and_not_exists_for_children(schema):
    q = compiled(schema, group(cond("account.country", "not_in", ["US", "DE"])))
    assert "(a.country IS NULL OR a.country NOT IN (%(p0)s, %(p1)s))" in q.sql


def test_every_subquery_is_tenant_scoped(schema):
    q = compiled(schema, group(cond("account.intent_strength", "in", ["low"]), cond("opportunity.open_count", ">", 2)))
    assert q.sql.count("FROM ") == 3 and q.sql.count("tenant_id = %(tenant_id)s") == 3


def test_like_wildcards_in_user_text_are_escaped(schema):
    q = compiled(schema, group(cond("account.name", "contains", "100%_off\\")))
    assert q.params["p0"] == "%100\\%\\_off\\\\%" and "ESCAPE" in q.sql


INJECTIONS = ["'; DROP TABLE accounts; --", "x' OR '1'='1", "%(tenant_id)s", "1; DELETE FROM accounts", "\\'", "a\nb"]


@pytest.mark.parametrize("payload", INJECTIONS)
def test_user_text_is_only_ever_a_bound_parameter(schema, payload):
    q = compiled(schema, group(cond("account.name", "contains", payload), cond("account.name", "=", payload)))
    assert ";" not in q.sql and "DROP" not in q.sql and "DELETE" not in q.sql and "OR '1'" not in q.sql
    assert set(re.findall(r"%\((\w+)\)s", q.sql)) == {"tenant_id", "limit", "p0", "p1"}  # payload added no placeholder
    assert q.params["p1"] == payload.strip()
    assert q.params["p0"] == "%" + payload.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=1, max_size=60).filter(lambda s: s.strip() and "\x00" not in s))
def test_arbitrary_text_cannot_change_sql_shape(text):
    schema = load_schema(SPEC)
    base = compiled(schema, group(cond("account.name", "contains", "x"))).sql
    assert compiled(schema, group(cond("account.name", "contains", text))).sql == base


def test_output_is_always_one_select_and_params_match_placeholders(schema):
    q = compiled(schema, group(cond("account.name", "contains", "a"), cond("account.industry", "in", ["Retail", "Software"]),
                               cond("opportunity.open_count", ">=", 1)))
    assert q.sql.startswith("SELECT ") and ";" not in q.sql
    assert not re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b", q.sql, re.I)
    assert set(re.findall(r"%\((\w+)\)s", q.sql)) == set(q.params)


@pytest.mark.parametrize("tenant", [None, "", True, 1.5, ["t"]])
def test_tenant_is_mandatory(schema, tenant):
    with pytest.raises(ValueError):
        compile_filter(parse_filter(group(), schema), schema, tenant)


@pytest.mark.parametrize("limit", [0, -1, 10_001, True, "5"])
def test_limit_is_bounded(schema, limit):
    with pytest.raises(ValueError):
        compile_filter(parse_filter(group(), schema), schema, "t", limit=limit)


def test_schema_rejects_unsafe_identifiers_and_bad_definitions():
    import copy
    def with_(mutate):
        spec = copy.deepcopy(SPEC)
        mutate(spec)
        return spec
    bad = [
        lambda s: s["fields"]["account.country"]["source"].update(column="country; DROP TABLE x"),
        lambda s: s["fields"]["opportunity.open_count"]["source"].update(table="opps o"),
        lambda s: s["root"].update(alias="a b"),
        lambda s: s["fields"]["account.country"].update(ops=["in", ">"]),
        lambda s: s["fields"]["account.country"].update(type="wizard"),
        lambda s: s["fields"]["account.revenue_usd"]["source"].update(kind="child_exists"),
        lambda s: s["fields"]["opportunity.open_count"]["source"].update(where=[["status; --", "open"]]),
        lambda s: s["fields"]["account.country"].update(values=[]),
    ]
    for mutate in bad:
        with pytest.raises(ValueError):
            load_schema(with_(mutate))
