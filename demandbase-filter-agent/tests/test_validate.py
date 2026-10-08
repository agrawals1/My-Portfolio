import pytest
from hypothesis import given, settings, strategies as st

from filteragent.filter_ast import Condition, Group
from filteragent.validate import MAX_CONDITIONS, ValidationError, parse_filter
from tests.conftest import cond, group


def err(schema, raw):
    with pytest.raises(ValidationError) as e:
        parse_filter(raw, schema)
    return e.value


def test_query1_parses_and_canonicalises_case(schema):
    f = parse_filter(group(cond("account.industry", "in", ["financial services"]),
                           cond("account.country", "in", " in "[1:3].upper().split()),
                           cond("account.employee_count", ">", 500),
                           cond("account.intent_strength", "in", ["HIGH"])), schema)
    assert f == Group("and", (Condition("account.industry", "in", ("Financial Services",)),
                              Condition("account.country", "in", ("IN",)),
                              Condition("account.employee_count", ">", 500),
                              Condition("account.intent_strength", "in", ("high",))))


def test_unknown_country_message_matches_the_brief(schema):
    e = err(schema, group(cond("account.country", "in", ["Narnia"])))
    assert e.message == "Unknown country 'Narnia'; supported: US, IN, GB, DE"


def test_unknown_field_suggests_close_match_and_never_echoes_sql(schema):
    assert "Did you mean 'account.revenue_usd'" in err(schema, group(cond("account.revenue_usdd", ">", 1))).message
    msg = err(schema, group(cond("a.revenue; DROP TABLE accounts", ">", 1))).message
    assert msg.startswith("Unknown field")


@pytest.mark.parametrize("raw", [None, [], "select 1", 5, {"op": "and"}, {"op": "and", "conditions": [], "sql": "x"},
                                 {"op": "xor", "conditions": []}, {"op": "and", "conditions": "all"},
                                 group({"field": "account.country", "op": "in", "value": ["US"], "extra": 1}),
                                 group(cond("account.country", "in", ["US"]) | {"sql": "1=1"}),
                                 group("account.country = 'US'")])
def test_bad_shapes_rejected(schema, raw):
    assert err(schema, raw).code == "bad_shape"


def test_operator_must_be_allowed_for_the_field(schema):
    e = err(schema, group(cond("account.industry", ">", ["Software"])))
    assert e.code == "bad_op" and "use: in, not_in" in e.message
    assert err(schema, group(cond("account.journey_stage", "not_in", ["Aware"]))).code == "bad_op"
    assert err(schema, group(cond("account.revenue_usd", "; DROP TABLE x", 1))).code == "bad_op"
    assert err(schema, group(cond("account.revenue_usd", ["="], 1))).code == "bad_op"


@pytest.mark.parametrize("value", ["100M", "1; DROP TABLE accounts", True, None, float("nan"), float("inf"), [5], -1])
def test_number_values_are_strict(schema, value):
    assert err(schema, group(cond("account.revenue_usd", ">", value))).code == "bad_value"


def test_range_message_is_readable(schema):
    assert err(schema, group(cond("account.revenue_usd", ">", -5))).message == "Revenue (USD) must be at least 0."


@pytest.mark.parametrize("value", [[], [1], [None], ["US", "' OR 1=1 --"], {"a": 1}, None, 5])
def test_enum_values_are_strict(schema, value):
    assert err(schema, group(cond("account.country", "in", value))).code == "bad_value"


def test_enum_scalar_is_wrapped_and_duplicates_collapsed(schema):
    f = parse_filter(group(cond("account.country", "in", "us"), cond("account.industry", "in", ["Retail", "retail"])), schema)
    assert f.conditions[0].value == ("US",) and f.conditions[1].value == ("Retail",)


def test_string_field_bounds(schema):
    assert parse_filter(group(cond("account.name", "contains", "  Acme ")), schema).conditions[0].value == "Acme"
    for bad in ["", "   ", "x" * 101, 5, "a\x00b"]:
        assert err(schema, group(cond("account.name", "contains", bad))).code == "bad_value"


def test_complexity_limits(schema):
    many = group(*[cond("account.revenue_usd", ">", i) for i in range(MAX_CONDITIONS + 1)])
    assert err(schema, many).code == "too_complex"
    deep = group(cond("account.country", "in", ["US"]))
    for _ in range(3):
        deep = group(deep, op="or")
    assert err(schema, deep).code == "too_complex"
    assert err(schema, group(group())).code == "bad_shape"           # empty nested group
    assert parse_filter(group(), schema).conditions == ()              # empty root = all accounts


def test_tenant_scoping_hides_fields_without_leaking_existence(schema):
    raw = group(cond("account.beta_score", ">", 1))
    assert parse_filter(raw, schema.for_tenant("tenant_beta")).conditions
    e = err(schema.for_tenant("tenant_acme"), raw)
    assert e.code == "unknown_field"


json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True), st.text(max_size=20))
json_val = st.recursive(json_leaf, lambda c: st.one_of(st.lists(c, max_size=4),
                                                       st.dictionaries(st.text(max_size=12), c, max_size=4)), max_leaves=15)


@settings(max_examples=300, deadline=None)
@given(json_val)
def test_fuzz_only_validation_errors_ever_escape(raw):
    from tests.conftest import SPEC
    from filteragent.schema import load_schema
    try:
        parse_filter(raw, load_schema(SPEC))
    except ValidationError:
        pass
