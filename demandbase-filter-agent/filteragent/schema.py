"""Semantic-layer schema. Developer-owned and trusted: it maps field names to SQL *structure* (table/column
identifiers, validated at load) — never raw SQL fragments. The LLM only ever sees `schema_for_llm()`."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_OPS_BY_TYPE = {
    "number": {">", ">=", "<", "<=", "="},
    "enum": {"in", "not_in"},
    "string": {"contains", "="},
}
_KINDS_BY_TYPE = {"column": {"number", "enum", "string"}, "child_count": {"number"}, "child_exists": {"enum"}}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    type: str
    ops: tuple[str, ...]
    source: dict
    values: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    tenants: tuple[str, ...] = ()  # empty = every tenant


@dataclass(frozen=True)
class Schema:
    table: str
    alias: str
    id_column: str
    tenant_column: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    def for_tenant(self, tenant_id: str) -> "Schema":
        """Tenant-aware access: fields a tenant may not use simply do not exist for it."""
        visible = {n: f for n, f in self.fields.items() if not f.tenants or tenant_id in f.tenants}
        return Schema(self.table, self.alias, self.id_column, self.tenant_column, visible)


def _ident(value: object, what: str) -> str:
    if not isinstance(value, str) or not _IDENT.match(value):
        raise ValueError(f"schema: {what} {value!r} is not a safe identifier")
    return value


def load_schema(spec: dict) -> Schema:
    root = spec["root"]
    fields: dict[str, FieldSpec] = {}
    for name, f in spec["fields"].items():
        ftype = f["type"]
        if ftype not in _OPS_BY_TYPE:
            raise ValueError(f"schema: {name}: unknown type {ftype!r}")
        bad = set(f["ops"]) - _OPS_BY_TYPE[ftype]
        if bad or not f["ops"]:
            raise ValueError(f"schema: {name}: ops {sorted(bad) or 'empty'} invalid for type {ftype}")
        src = dict(f["source"])
        kind = src.get("kind")
        if kind not in _KINDS_BY_TYPE or ftype not in _KINDS_BY_TYPE[kind]:
            raise ValueError(f"schema: {name}: source kind {kind!r} cannot back a {ftype} field")
        for key in ("column", "table", "fk"):
            if key in src:
                _ident(src[key], f"{name}.{key}")
        for pair in src.get("where", ()):
            _ident(pair[0], f"{name}.where column")
            if not isinstance(pair[1], (str, int, bool)):
                raise ValueError(f"schema: {name}: where values must be str/int/bool")
        if ftype == "enum" and not f.get("values"):
            raise ValueError(f"schema: {name}: enum needs values")
        fields[name] = FieldSpec(name, f.get("label", name), ftype, tuple(f["ops"]), src,
                                 tuple(f.get("values", ())), f.get("min"), f.get("max"), tuple(f.get("tenants", ())))
    return Schema(_ident(root["table"], "root.table"), _ident(root["alias"], "root.alias"),
                  _ident(root["id_column"], "root.id_column"), _ident(root["tenant_column"], "root.tenant_column"),
                  fields)


def schema_for_llm(schema: Schema) -> list[dict]:
    """What the model sees: names, labels, types, operators, allowed values. No tables, columns or SQL."""
    out = []
    for f in schema.fields.values():
        item: dict = {"field": f.name, "label": f.label, "type": f.type, "ops": list(f.ops)}
        if f.values:
            item["values"] = list(f.values)
        if f.min is not None:
            item["min"] = f.min
        out.append(item)
    return out
