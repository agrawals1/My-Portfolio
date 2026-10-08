# Mock interview trajectories: Problem 3 (NL → Validated Filter AST → Safe SQL)

Four realistic ways a 60–90 minute round on this problem can go. **I** = interviewer, **C** = candidate (you).
`[AI]` lines are what you type into the coding assistant, and `[screen]` lines are what happens on screen.
SQL, messages and numbers quoted here match the code in this folder.

| # | Situation | What it trains |
|---|---|---|
| 1 | Standard, cooperative interviewer | Clarify, state the safety rule, schema → validator → compiler → agent, demo, wrap up |
| 2 | Security / red-team interviewer | Walking through attacks one by one and pointing at the line that stops each |
| 3 | The AI gets it wrong, and the design gets challenged | Catching unsafe SQL building, a wrong test, "why not let the LLM write SQL and validate it?" |
| 4 | Multi-turn added late, then a production deep dive | Patch vs rewrite under time pressure; deployment, evaluation, schema versioning |

At the end there's a short **answer bank**.

---

## Trajectory 1: the standard run

**00:00**

**I:** This is the filter agent. A user types something like "Likely accounts with more than 100M revenue and no open opportunities", and we produce a filter they can see and edit, plus the SQL that runs. Read it and ask questions.

**C:** *(reads for ~2 minutes)* Some questions.
1. Is the SQL executed by us, shown to the user, or both?
2. Which database and driver? That decides the parameter style.
3. Is the semantic-layer schema given to me as data? And can fields live on other tables, like opportunities?
4. Multi-tenancy: shared tables with a `tenant_id` column, or a schema per tenant?
5. Follow-ups: does "remove the revenue filter" modify the previous filter or start fresh?
6. If the user asks for something the filter language can't express, should I fail, ask, or approximate?

**I:** Both: the UI shows the filter, and the SQL runs on the warehouse. Assume Postgres-style with psycopg named parameters. The schema is given; the sample has fields on intent and opportunity tables. Shared tables with `tenant_id`. Follow-ups modify. If it can't be expressed, tell the user clearly.

**C:** Then the core rule is: **the LLM produces filter JSON, never SQL**. The JSON is validated against the schema, and a deterministic compiler produces the SQL. The reason in one sentence: I can list every valid filter, but I can't list every valid SQL statement.

**00:05: interfaces**

```
[screen] filteragent/filter_ast.py  Condition(field, op, value), Group(op, conditions)   frozen dataclasses
         filteragent/schema.py      load_schema(spec) -> Schema ; schema_for_llm(schema) -> list[dict]
         filteragent/validate.py    parse_filter(raw, schema) -> Group   raises ValidationError(message, code)
         filteragent/compile.py     compile_filter(filt, schema, tenant_id, limit) -> CompiledQuery(sql, params)
         filteragent/agent.py       run_query(utterance, schema, llm, tenant_id, previous=None) -> QueryResult
```

**I:** Why a separate frozen AST? You could compile the dict directly.

**C:** If the compiler accepts dicts, sooner or later someone passes raw LLM output straight into it. If it accepts only `Group` objects, and only the validator builds them, the type itself says "this has been checked". Frozen means nothing can change it after validation.

**00:09: schema design**

**C:** The schema maps each field to a *structure*, not to SQL text. There are three kinds:
- `column`: a column on the accounts table
- `child_count`: a count subquery on a child table, for example open opportunities
- `child_exists`: an `EXISTS` subquery, for example intent strength

Every table and column name is checked against `^[a-z_][a-z0-9_]*$` when the schema loads.

**I:** Why not let the schema hold a SQL snippet, like `EXISTS (SELECT 1 FROM intent_scores ...)`?

**C:** Two reasons. First, the schema would become a place where SQL can be injected, even if only by a careless config change. Second, I couldn't guarantee that every snippet includes the tenant predicate. With structured kinds, the compiler adds `tenant_id = %(tenant_id)s` to every subquery itself, every time.

**00:13: validator, tests first**

```
[screen] tests/test_validate.py
  Narnia -> "Unknown country 'Narnia'; supported: US, IN, GB, DE"
  {"field": "account.industry", "op": ">", ...}            -> bad_op
  revenue "100M", True, NaN, inf, -1                       -> bad_value
  extra keys like "sql": "..."                             -> bad_shape
  21 conditions, 4 levels of nesting                       -> too_complex
```

```
[AI] Implement parse_filter(raw, schema) in filteragent/validate.py. Untrusted JSON. A group has exactly
     keys {op, conditions}; a condition has exactly {field, op, value}. Field must exist in schema; op must be in
     the field's ops. number: int/float, not bool, finite, within min/max. enum: list of strings, match
     case-insensitively and return the canonical spelling, error lists supported values. string: 1-100 chars.
     Max depth 3, max 20 conditions. Raise ValidationError(message, code) with a user-facing message.
```

**C:** *(reading)* It excludes `bool` explicitly, which matters because `isinstance(True, int)` is true in Python. Enum matching returns the schema's spelling, so "high", "HIGH" and " High " all become `high`.

**I:** Why are numbers not coerced? "100M" seems like an easy win.

**C:** Unit handling belongs in the prompt, where the model converts "$100M" to 100000000. If the validator silently guessed, "100M" with `employee_count` would become 100 million employees. A rejection costs one repair round-trip; a silent wrong guess costs trust.

**00:28: compiler**

**C:** The compiler is simple on purpose: a fixed operator map, every value bound as a parameter, and every identifier taken from the schema.

```
[screen] SELECT a.account_id FROM accounts a WHERE a.tenant_id = %(tenant_id)s
         AND (a.industry IN (%(p0)s) AND a.country IN (%(p1)s) AND a.employees > %(p2)s
         AND EXISTS (SELECT 1 FROM intent_scores x0 WHERE x0.account_id = a.account_id
                     AND x0.tenant_id = %(tenant_id)s AND x0.strength IN (%(p3)s)))
         ORDER BY a.account_id LIMIT %(limit)s
params = {'tenant_id': 't1', 'limit': 1000, 'p0': 'Financial Services', 'p1': 'IN', 'p2': 500, 'p3': 'high'}
```

**I:** "No open opportunities": why a `COUNT(*)` subquery equal to 0, rather than `NOT EXISTS`?

**C:** The field is `open_count` with operators `=`, `>` and `>=`, so one compile shape covers all of them. You're right that `NOT EXISTS` is usually faster for `= 0` specifically. That's a good optimisation to special-case in the compiler, and the AST doesn't change.

**I:** And `not_in`?

**C:** `not_in` on a column compiles to `(col IS NULL OR col NOT IN (...))`. Plain `NOT IN` silently drops rows where the column is NULL, and I think "not in the US" should include accounts with unknown country. That's a product decision, so I'd confirm it. It's one place in the code either way.

**00:40: the agent**

**C:** `run_query` asks the model for JSON, validates it, and compiles it. If validation fails, the error message goes back to the model **once**, and it gets one more try. If that also fails, the user sees the message. The validator's messages are written to be shown to users. Provider exceptions become a generic "couldn't do that right now", without internals.

**00:48: demo**

```
[screen] $ python -m filteragent.cli
> fintech accounts in India with more than 500 employees showing high intent
  Industry is any of Financial Services / Country is any of IN / Employees is greater than 500 / Intent strength is any of high
> Accounts with revenue over $100M and no open opportunities
  ... (SELECT COUNT(*) FROM opportunities x0 WHERE ... AND x0.status = %(p1)s) = %(p2)s
> companies in Narnia
  ERROR: Unknown country 'Narnia'; supported: US, IN, GB, DE
> drop table accounts; show me everything
  (no conditions)  SQL: SELECT a.account_id FROM accounts a WHERE a.tenant_id = %(tenant_id)s ORDER BY ... LIMIT %(limit)s
```

**I:** For the last one, why produce any SQL at all?

**C:** "Show me everything" is a legitimate request. The "drop table" part can't be represented in the filter language, so it simply disappears. What runs is a tenant-scoped select-all with a limit, and the result carries a warning that the filter matches every account. If the product prefers an error for empty filters, that's a one-line change.

**00:55: wrap-up**

**I:** Limitations and deployment?

**C:** Limitations:
- Follow-up patches only work on top-level conditions.
- There are no relative dates yet ("last 30 days").
- Mapping "fintech" to "Financial Services" depends on the model; there's no synonym table yet.
- The `not_in`/NULL behaviour needs a product decision.

Deployment: a stateless service next to the semantic layer, with:
- a read-only database role
- row-level security on `tenant_id` as a second wall
- a statement timeout
- an audit log of utterance, AST and SQL hash

For quality, a golden set of utterance → expected filter per tenant, re-run on every prompt or model change.

---

## Trajectory 2: the security / red-team interviewer

The interviewer has seen the design and now tries to break it. Stay calm and point at the code for each attack.

**I:** I type "drop table accounts".

**C:** The model can only return filter JSON. There's no field or operator that means "drop", so the worst case is an empty or irrelevant filter, and the compiler only produces one `SELECT`.

**I:** Suppose the model is jailbroken and returns `{"sql": "DROP TABLE accounts"}`.

**C:** The validator requires the keys to be exactly `op` and `conditions`, so that's `bad_shape`. Nothing is compiled. There's a parametrised test with strings, `None`, lists and that exact payload, and all of them come back as `ok=False` with no SQL.

**I:** It puts the injection in the field name: `"account.revenue_usd; DROP TABLE x"`.

**C:** Field names must be keys of the tenant's schema, and that isn't one, so the error is "Unknown field". Column names in the SQL come from the schema, never from the request.

**I:** In the operator, then.

**C:** Operators must be in that field's allowed list, and the compiler maps them through a fixed dictionary. `"; DROP TABLE x"` gives `bad_op`.

**I:** The account name field accepts free text. I search for `x' OR '1'='1`.

**C:** It's bound as a parameter, so the database sees it as a string, not as code. The test checks that the payload adds no placeholders: the set of `%(name)s` placeholders in the SQL is exactly `{tenant_id, limit, p0, p1}`. A hypothesis test also checks that arbitrary text never changes the SQL string at all.

**I:** I search for `%`.

**C:** Good one. Bound or not, `%` and `_` are wildcards inside `LIKE`, so searching for "%" would match everything. The compiler escapes them and adds `ESCAPE '\'`. There's a test with `100%_off\`.

**I:** I type `%(tenant_id)s` as the value.

**C:** It's still just a value. It goes into `params` as a plain string, and the driver never interprets the contents of a parameter as a placeholder. The placeholder-set test covers exactly that payload.

**I:** Cross-tenant leaks?

**C:** Three layers:
1. `tenant_id` comes from the authenticated session, not the request body, and it's a bound parameter in the root query **and every subquery**. There's a test that counts the tenant predicates: three `FROM`s, three predicates.
2. Tenant-restricted fields don't exist for other tenants. Using one gives the same "Unknown field" error as a made-up field, so it doesn't reveal that the field exists.
3. In production, row-level security in the database as well.

**I:** The client sends `previous_filter` back on follow-ups. I tamper with it.

**C:** It goes through the same validator before anything else. A tampered value like `"0; DROP TABLE accounts"` in a number field gives "The previous filter is no longer valid: ..." and nothing runs.

**I:** Denial of service? 10,000 conditions, deep nesting, a huge IN list.

**C:** There are limits:
- at most 20 conditions
- at most 3 levels of nesting
- at most 50 values per list
- utterances up to 500 characters
- `LIMIT` always bound and capped at 10,000

At the database: a statement timeout and an `EXPLAIN` cost cap.

**I:** Can error messages leak anything?

**C:** They echo at most 40 printable characters of the bad value, and they never include SQL, table names or stack traces.

**I:** The model sees the schema. Can a prompt injection make it reveal other tenants' fields?

**C:** It only ever sees `schema_for_llm` for the current tenant: names, labels, types, operators and allowed values. No table names, no columns, no other tenants. There's nothing else in its context to leak. The user's text is JSON-encoded in the prompt, so it can't break out of its slot.

**I:** So what risk is left?

**C:** The main one is semantic, not security: the model picks a valid but wrong filter, like "Software" for "fintech". It can't do damage, but it can give wrong results. That's why the UI shows the filter in plain words (`describe()`), why the user can edit it, and why we need an evaluation set.

---

## Trajectory 3: the AI gets it wrong, and the design gets challenged

**00:25: unsafe IN list**

```
[AI] Write the compiler for enum conditions.
[screen] return f"{col} IN ({', '.join(repr(v) for v in cond.value)})"
```

**C:** `repr` is Python quoting, not SQL escaping. A value with a quote in it breaks out. Even though the validator only allows known enum values today, the compiler shouldn't depend on that. I want every value bound.

```
[AI] Rewrite: every value must go through ctx.bind(value), which returns a %(pN)s placeholder and stores
     the value in params. No value may ever be formatted into the SQL string.
```

**00:31: subquery without the tenant predicate**

```
[screen] FAILED test_every_subquery_is_tenant_scoped   assert 2 == 3
```

**C:** The AI's `child_count` subquery filters on `account_id` only, not on `tenant_id`. Account IDs are probably unique across tenants, so this might "work". But if they ever aren't, this leaks another tenant's opportunity counts. The test that counts predicates per `FROM` caught it. Fixed by building the subquery's base clause in one place, so both subquery kinds share it.

**00:35: True is a number**

```
[screen] FAILED test_number_values_are_strict[True]   DID NOT RAISE
```

**C:** `isinstance(True, int)` is true in Python, so `"value": true` became revenue > 1. I'll add an explicit `bool` check before the numeric check.

**00:42: my own test was wrong**

```
[screen] FAILED test_values_never_reach_the_sql_text[%(tenant_id)s]
         assert '%(tenant_id)s' not in 'SELECT ... WHERE a.tenant_id = %(tenant_id)s ...'
```

**C:** My test said "the payload must not appear in the SQL". But `%(tenant_id)s` is legitimately in the SQL template, and so is `\'` in the `ESCAPE` clause. The code is right and the assertion is wrong. A better assertion: the set of placeholders is unchanged, `p1` equals the payload exactly, and `p0` equals the escaped LIKE pattern. That checks what I actually mean: user text only ever exists as a bound value.

**00:50: a bug no test caught**

**C:** Before the demo I always read the user-facing messages, because tests usually check codes, not wording.

```
[screen] revenue_usd = -5  →  "Revenue (USD) must be between 0 and None."
```

**C:** That's a bad message. Revenue has a minimum but no maximum. I'll split it into "must be at least 0" and "must be at most N", and pin the wording with a test.

**00:55: the design pushback**

**I:** This is a lot of machinery. Why not let the LLM write SQL and validate *that* with a SQL parser like sqlglot? It's far more expressive.

**C:** It's a fair option, and I'd consider it in some contexts. For this product I prefer the AST:
1. **Allow-listing**: with SQL I'd have to allow-list every node type (joins, CTEs, window functions, `UNION`) and reason about each one. With the AST, the allow-list *is* the schema.
2. **Tenant isolation**: injecting `tenant_id` into arbitrary SQL with CTEs, subqueries and unions is hard to get right. With my compiler it's structural: every `FROM` it emits gets the predicate.
3. **The UI needs the filter anyway**: "show the underlying selectors and operators" is the AST. Going from SQL back to an editable filter is the hard direction.
4. **Expressiveness** is added deliberately: a new schema `kind` is a reviewed code change.

Where I'd use LLM-written SQL: an internal analyst tool, on a read-only replica, with a sandboxed role, where flexibility matters more than strict guarantees.

**I:** "Fintech" isn't in your industry list. What if the model maps it wrongly?

**C:** Today the model maps it and the validator only checks the result is a real industry. Two improvements:
1. A synonym table in the schema ("fintech" → "Financial Services"), applied deterministically before validation. That's cheap, testable, and owned by the semantic layer.
2. When a value doesn't validate, return the closest few allowed values as clarification chips instead of failing, and never auto-pick silently.

---

## Trajectory 4: multi-turn added late, then production

**00:40**

**I:** We need follow-ups now. "Remove the revenue filter and only show healthcare." You have about 20 minutes.

**C:** There are two options.
- **(a) The model rewrites the full filter** from the previous one plus the utterance. It's simple, but the model can silently drop or change conditions the user didn't mention.
- **(b) The model returns a small patch**: a list of `remove`, `add` and `set` actions. Code applies it, and the result is validated again.

I'll do (b). The user said two things, so there should be two changes, and code guarantees nothing else moves. The diff for the UI also comes for free.

```
[AI] Add filteragent/patch.py: apply_patch(previous: Group, raw, schema) -> Group. raw must be exactly
     {"actions": [...]}, 1-10 actions. remove {field}: error if no root condition on that field.
     add {condition}: append. set {condition}: replace all root conditions on that field.
     Validate conditions with parse_condition; re-validate the final result with parse_filter.
     Also diff_filters(old, new) -> Diff(added, removed) on root-level conditions.
```

```
[screen] > follow-up: remove the revenue filter and only show healthcare
  diff: {"added":   [{"field": "account.industry", "op": "in", "value": ["Healthcare"]}],
         "removed": [{"field": "account.revenue_usd", "op": ">", "value": 100000000}]}
  remaining: opportunity.open_count = 0, account.industry in [Healthcare]
```

**I:** What if the model returns a full filter instead of a patch on a follow-up?

**C:** It's rejected, because a follow-up must be exactly `{"actions": [...]}`, and the error goes back for the one repair. I'd rather fail loudly than accept a rewrite that might drop conditions.

**I:** "Remove Germany" when the filter says country in [US, DE]?

**C:** The model sends `set` with country in [US]. Value-level operations would be nicer, but `set` covers it. The limitation: patches act on top-level conditions only. If there's a nested OR group, I'd need a path to address it. I'd add that when the UI supports nested groups.

**I:** What if the user says "remove the revenue filter" and there isn't one?

**C:** The user gets "There is no Revenue (USD) condition to remove." That's better than silently doing nothing.

**01:00: production**

**I:** Say we ship this. Walk me through it.

**C:**
- **Service**: stateless, next to the semantic layer, so it reads the same schema version the warehouse uses.
- **Latency**: the model call dominates, so validation and compilation don't matter. I'd use a small, fast model with structured output (JSON schema or tool-use), and escalate to a bigger one only after a failed repair. Cache by (tenant, normalised utterance, schema version).
- **Evaluation**: a golden set per tenant of utterance → expected filter, compared as canonical ASTs, plus an execution check on a fixture database. I'd track the repair rate and the rate of `unknown_value` errors.
- **Monitoring**: the distribution of validation error codes. A spike in `unknown_value` for `industry` usually means users have a word we don't support yet. That's a signal to extend the synonym table, not the prompt.
- **Schema versioning**: users save filters. When the schema changes (a value renamed, a field removed), saved filters must be re-validated, and the ones that broke must be migrated or flagged. They can't fail at query time.
- **Rollout**: shadow mode first. Generate filters for internal users without executing them, and compare them with what those users build by hand in the UI.

**I:** What would you not do?

**C:** Fine-tune a model before having the golden set. Until we can measure it, we can't tell if fine-tuning helped.

---

## Answer bank

| Question | Short answer |
|---|---|
| Why never LLM → SQL? | Valid filters can be enumerated; valid SQL can't. The model's only power is proposing JSON that must pass the validator. |
| Why a frozen AST? | The compiler accepts only validated objects; frozen means nothing changes them after validation. |
| Why no SQL in the schema? | Nothing to inject, and the compiler can guarantee the tenant predicate in every subquery. |
| Why reject "100M" instead of parsing it? | A wrong silent guess (e.g. on `employee_count`) is worse than one repair round-trip. |
| Why one repair, not three? | It bounds cost and latency; if two attempts fail, the user's message is more useful than a third guess. |
| How do you stop cross-tenant reads? | Session-supplied, bound `tenant_id` in every `FROM`; tenant-restricted fields don't exist for others; row-level security underneath. |
| Why are follow-ups patches? | The model can't drop conditions it wasn't asked to touch, and the diff falls out for free. |
| Why does `not_in` include NULLs? | SQL `NOT IN` silently drops NULL rows; "not in US" should include unknown country. It's a product call, in one place. |
| Why is an empty filter allowed? | "Show me everything" is legitimate; it's tenant-scoped, limited, and returns a warning. |
| Biggest remaining risk? | A valid but wrong filter (semantic error). Mitigated by showing the filter in words, letting users edit it, and evaluating on a golden set. |
