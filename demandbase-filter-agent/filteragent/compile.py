"""Deterministic AST -> parameterised SELECT. Every identifier comes from the validated schema, every value from a
bound parameter, every operator from a fixed map. Output is always one SELECT; there is no code path to anything else."""
from __future__ import annotations

from dataclasses import dataclass

from filteragent.filter_ast import Condition, Group, Node
from filteragent.schema import Schema

_NUM_OPS = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "=": "="}
MAX_LIMIT = 10_000


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    params: dict


class _Ctx:
    def __init__(self, schema: Schema, tenant_id: object, limit: int) -> None:
        self.schema, self.params, self.aliases = schema, {"tenant_id": tenant_id, "limit": limit}, 0

    def bind(self, value: object) -> str:
        name = f"p{len(self.params) - 2}"  # tenant_id and limit are the first two params
        self.params[name] = value
        return f"%({name})s"

    def alias(self) -> str:
        self.aliases += 1
        return f"x{self.aliases - 1}"


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _in_list(ctx: _Ctx, values: tuple) -> str:
    return ", ".join(ctx.bind(v) for v in values)


def _condition(c: Condition, ctx: _Ctx) -> str:
    s, spec = ctx.schema, ctx.schema.fields[c.field]
    src = spec.source
    root = s.alias
    if src["kind"] == "column":
        col = f"{root}.{src['column']}"
        if spec.type == "number":
            return f"{col} {_NUM_OPS[c.op]} {ctx.bind(c.value)}"
        if spec.type == "enum":
            lst = _in_list(ctx, c.value)
            return f"{col} IN ({lst})" if c.op == "in" else f"({col} IS NULL OR {col} NOT IN ({lst}))"
        if c.op == "contains":
            return f"{col} ILIKE {ctx.bind('%' + _like_escape(c.value) + '%')} ESCAPE '\\'"
        return f"LOWER({col}) = LOWER({ctx.bind(c.value)})"
    x = ctx.alias()
    base = (f"FROM {src['table']} {x} WHERE {x}.{src['fk']} = {root}.{s.id_column} "
            f"AND {x}.{s.tenant_column} = %(tenant_id)s")
    if src["kind"] == "child_count":
        extra = "".join(f" AND {x}.{col} = {ctx.bind(val)}" for col, val in src.get("where", ()))
        return f"(SELECT COUNT(*) {base}{extra}) {_NUM_OPS[c.op]} {ctx.bind(c.value)}"
    inner = f"SELECT 1 {base} AND {x}.{src['column']} IN ({_in_list(ctx, c.value)})"
    return f"EXISTS ({inner})" if c.op == "in" else f"NOT EXISTS ({inner})"


def _node(n: Node, ctx: _Ctx) -> str:
    if isinstance(n, Condition):
        return _condition(n, ctx)
    return "(" + f" {n.op.upper()} ".join(_node(child, ctx) for child in n.conditions) + ")"


def compile_filter(filt: Group, schema: Schema, tenant_id: object, limit: int = 1000) -> CompiledQuery:
    if not isinstance(tenant_id, (str, int)) or isinstance(tenant_id, bool) or tenant_id == "":
        raise ValueError("tenant_id is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
    ctx = _Ctx(schema, tenant_id, limit)
    a = schema.alias
    where = f"{a}.{schema.tenant_column} = %(tenant_id)s"
    if filt.conditions:
        where += f" AND {_node(filt, ctx)}"
    sql = f"SELECT {a}.{schema.id_column} FROM {schema.table} {a} WHERE {where} ORDER BY {a}.{schema.id_column} LIMIT %(limit)s"
    return CompiledQuery(sql, ctx.params)
