# Problem 3 — Natural Language → Validated Filter AST → Safe SQL (Filter-Agent style)

Reference solution for the semantic-layer "Filter Agent". The rule that makes it safe:

> **LLM → structured filter JSON → validation against the semantic-layer schema → deterministic SQL compiler.
> Never LLM → raw SQL.**

The model's only power is to *propose JSON*. Everything it returns is untrusted input to a strict validator; the
compiler only accepts the validator's immutable AST, and only ever emits one parameterised `SELECT`.
~450 lines, **zero runtime dependencies**, 88 tests (incl. property/fuzz tests), <2 s, fully offline (scripted LLM).

```
filteragent/
  schema.py      semantic-layer schema: field -> SQL *structure* (identifiers validated at load, no raw SQL) · schema_for_llm
  filter_ast.py  frozen Condition / Group — the only input the compiler accepts
  validate.py    parse_filter: untrusted JSON -> AST or ValidationError (user-facing message); limits on depth/size
  patch.py       follow-ups: model proposes {actions:[remove|add|set]}, code applies it · diff_filters
  compile.py     AST -> (sql, params): fixed operator map, bound parameters, tenant predicate in every (sub)query
  llm.py         FilterLLM Protocol · build_prompt (JSON-encoded user text, no SQL in schema) · ScriptedLLM
  agent.py       run_query: propose -> validate/patch -> compile, one bounded repair round-trip · describe()
  cli.py         python -m filteragent.cli       (the four queries + follow-up from the brief)
tests/           validate (strictness, fuzz) · compile (golden SQL, injection, schema safety) · agent (flows, hostile output)
```

```bash
pip install pytest hypothesis && python -m pytest -q       # 88 passed
python -m filteragent.cli
```

Mock interviews for this problem: [INTERVIEW_TRAJECTORIES.md](INTERVIEW_TRAJECTORIES.md)

## Behaviour on the brief's inputs

| Input | Result |
|---|---|
| "fintech accounts in India with more than 500 employees showing high intent" | AST of 4 conditions → `SELECT a.account_id FROM accounts a WHERE a.tenant_id = %(tenant_id)s AND (a.industry IN (%(p0)s) AND a.country IN (%(p1)s) AND a.employees > %(p2)s AND EXISTS (SELECT 1 FROM intent_scores x0 WHERE … AND x0.tenant_id = %(tenant_id)s AND x0.strength IN (%(p3)s))) …` + param dict |
| "revenue over $100M and no open opportunities" | `a.revenue_usd > %(p0)s AND (SELECT COUNT(*) FROM opportunities x0 WHERE … tenant-scoped … AND x0.status = %(p1)s) = %(p2)s` (0 open = zero-count subquery, so accounts with *no rows* still match) |
| "companies in Narnia" | `ValidationError`: *Unknown country 'Narnia'; supported: US, IN, GB, DE* (fed back to the model once, then shown to the user) |
| "drop table accounts; show me everything" | There is no channel from text to DB: the model can only emit filter JSON. Anything else (`{"sql": …}`, strings, wrong shapes) is rejected; an empty filter compiles to a tenant-scoped select-all **with a warning**. |
| Follow-up "remove the revenue filter and only show healthcare" | Patch `[remove revenue, add industry in Healthcare]` → diff `removed: revenue>1e8`, `added: industry in [Healthcare]`; the untouched open-opportunities condition is preserved by code, not by the model |

## Decisions worth defending

| Decision | Why | Trade-off / what I'd change |
|---|---|---|
| Schema maps fields to structure (`kind: column / child_count / child_exists`), never SQL strings | Identifiers are regex-validated at load; there is nothing for an attacker (or a careless config author) to inject | New query shapes need a new `kind`, which is deliberate: a reviewed code change |
| Strict validator: exact key sets, allowed ops per field, typed/bounded values, canonical enum values | Defence in depth: even a jailbroken model can only produce what a user could build in the UI | Strictness costs a repair round-trip; capped at 1 |
| Enum values matched case-insensitively and canonicalised; bare string accepted for `in` | Cheap, safe tolerance of the two most common LLM slips | Numbers are *not* coerced (`"100M"` is rejected): unit conversion belongs in the prompt/LLM, and a wrong silent guess is worse than a retry |
| Follow-ups are **patches**, re-validated after applying | Model can't drop conditions it wasn't asked to touch; the diff for the UI falls out for free | Patches act on root-level conditions only; nested edits need a path syntax |
| `previous_filter` from the client is re-validated | It round-trips through the browser, so it is user input | — |
| `not_in` also matches NULL; child "not in" ⇒ `NOT EXISTS` | "Not in the US" should include unknown-country accounts; SQL's `NOT IN` silently drops NULLs | Product call; flip in one place |
| `tenant_id` is a required, bound parameter in the root query **and every subquery**; tenant-restricted fields don't exist for other tenants | Tenant-aware access without leaking that a field exists | Real system: also enforce with row-level security as a second wall |
| Empty filter allowed (+ warning), nested empty group rejected, `LIMIT` always bound and capped at 10 000 | "Show everything" is legitimate; unbounded scans are not | UI should confirm before running an empty filter on a huge tenant |
| User text is JSON-encoded in the prompt; the model sees names/types/values only | Slot-breaking and schema leakage prevented; **prompt injection is still assumed possible**, which is why validation exists | — |

## Where AI-generated code typically goes wrong (covered by tests)

| Trap | What the code/tests do instead |
|---|---|
| f-string / `%` / `.format` SQL with user values | every value is `%(pN)s`; test proves payloads (`'; DROP TABLE…`, `%(tenant_id)s`) add no placeholder and never appear in SQL; hypothesis: arbitrary text never changes the SQL shape |
| Column names or operators taken from the model's JSON | fields/ops must be keys of the schema / fixed map; unknown op `"; DROP TABLE x"` ⇒ `bad_op` |
| `isinstance(x, int)` accepts `True`; `NaN`/`inf` pass numeric checks | explicit bool exclusion + `math.isfinite`; min/max from schema |
| `LIKE` with user text: `%` and `_` act as wildcards | escaped, with `ESCAPE '\'` |
| Subquery forgets the tenant predicate (cross-tenant leak) | every subquery carries `tenant_id = %(tenant_id)s`; test counts them |
| `NOT IN` with NULLs silently drops rows | `(col IS NULL OR col NOT IN (…))`, `NOT EXISTS` for children |
| Trusting a round-tripped `previous_filter` | re-parsed with the same validator; test with tampered payload |
| Fuzzing found nothing because validation only raises `ValueError`s by accident | hypothesis test: *only* `ValidationError` may escape for arbitrary JSON |
| Leaking provider errors / stack traces to the user | `except Exception` at the LLM boundary only, generic message; test asserts no internals in the text |
| Unbounded repair loops | `max_repairs=1`; test asserts exactly 2 model calls |

## Stretch talking points

* **Eval harness**: golden set of (utterance → AST) per tenant; exact-match on canonical AST, plus an *execution* check against a fixture DB; track repair rate and `unknown_value` rate as quality signals.
* **Ambiguity**: when a value fails to validate ("fintech"), return top-k nearest enum values (embedding/fuzzy) as a *clarification chip* instead of failing; never auto-pick silently.
* **Richer ops**: `between`, relative dates (`last 30 days` → resolved server-side from a clock, never by the model), `has_signal(type, window)` as new schema `kind`s.
* **Defence in depth**: read-only DB role, row-level security on `tenant_id`, statement timeout, `EXPLAIN` cost cap, audit log of (utterance, AST, SQL hash).
* **Caching**: key by (tenant, normalised utterance, schema version); invalidate on schema change.
