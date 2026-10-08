"""LLM -> structured filter -> validation -> deterministic SQL, with one bounded repair round-trip."""
from __future__ import annotations

from dataclasses import dataclass, field

from filteragent.compile import CompiledQuery, compile_filter
from filteragent.filter_ast import Condition, Group
from filteragent.llm import FilterLLM
from filteragent.patch import Diff, apply_patch, diff_filters
from filteragent.schema import Schema, schema_for_llm
from filteragent.validate import ValidationError, parse_filter

MAX_UTTERANCE = 500
_OP_WORDS = {">": "is greater than", ">=": "is at least", "<": "is less than", "<=": "is at most", "=": "is",
             "in": "is any of", "not_in": "is none of", "contains": "contains"}


@dataclass(frozen=True)
class QueryResult:
    ok: bool
    filter: Group | None = None
    sql: str | None = None
    params: dict | None = None
    diff: Diff | None = None
    explanation: tuple[str, ...] = ()
    error: str | None = None
    repaired: bool = False
    warnings: tuple[str, ...] = field(default=())


def describe(filt: Group, schema: Schema, indent: int = 0) -> list[str]:
    """Human-readable selectors and operators (what the UI shows as the 'visual query')."""
    lines: list[str] = []
    pad = "  " * indent
    for n in filt.conditions:
        if isinstance(n, Condition):
            spec = schema.fields[n.field]
            val = ", ".join(n.value) if isinstance(n.value, tuple) else n.value
            lines.append(f"{pad}{spec.label} {_OP_WORDS[n.op]} {val}")
        else:
            lines.append(f"{pad}{'Any' if n.op == 'or' else 'All'} of:")
            lines += describe(n, schema, indent + 1)
    return lines


def _fail(message: str) -> QueryResult:
    return QueryResult(ok=False, error=message)


def run_query(utterance: object, schema: Schema, llm: FilterLLM, tenant_id: object,
              previous: object = None, max_repairs: int = 1, limit: int = 1000) -> QueryResult:
    """Never raises for bad user/LLM input; returns QueryResult(ok=False, error=<user-facing message>)."""
    if not isinstance(utterance, str) or not utterance.strip():
        return _fail("Please type what you want to filter on.")
    if len(utterance) > MAX_UTTERANCE:
        return _fail(f"Please keep requests under {MAX_UTTERANCE} characters.")
    if not isinstance(tenant_id, (str, int)) or isinstance(tenant_id, bool) or tenant_id == "":
        return _fail("Missing tenant.")
    tschema = schema.for_tenant(tenant_id)
    prev: Group | None = None
    if previous is not None:  # the client sends this back, so it is untrusted too
        try:
            prev = parse_filter(previous.to_dict() if isinstance(previous, Group) else previous, tschema)
        except ValidationError as e:
            return _fail(f"The previous filter is no longer valid: {e.message}")

    feedback: str | None = None
    error = "Sorry, I couldn't turn that into a filter."
    for attempt in range(max_repairs + 1):
        try:
            raw = llm.propose(utterance.strip(), schema_for_llm(tschema), prev.to_dict() if prev else None, feedback)
        except Exception:  # provider/timeouts/etc.: never leak internals to the user
            return _fail("Sorry, I couldn't turn that into a filter right now. Please try again.")
        try:
            filt = apply_patch(prev, raw, tschema) if prev is not None else parse_filter(raw, tschema)
        except ValidationError as e:
            feedback, error = e.message, e.message
            continue
        compiled: CompiledQuery = compile_filter(filt, tschema, tenant_id, limit)
        warnings = ("This filter has no conditions and matches every account.",) if not filt.conditions else ()
        return QueryResult(ok=True, filter=filt, sql=compiled.sql, params=compiled.params,
                           diff=diff_filters(prev, filt) if prev is not None else None,
                           explanation=tuple(describe(filt, tschema)), repaired=attempt > 0, warnings=warnings)
    return _fail(error)
