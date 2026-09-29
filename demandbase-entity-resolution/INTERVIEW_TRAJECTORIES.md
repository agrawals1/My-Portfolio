# Mock interview trajectories: Problem 1 (Company Entity Resolution)

Five realistic ways the 60–90 minutes can go. **I** = interviewer, **C** = candidate (you).
`[AI]` lines are what you type into the assistant, and `[screen]` lines are what happens on screen.
The numbers match the code in this repo, so you can say them with confidence.

| # | Situation | What it trains |
|---|---|---|
| 1 | Standard, cooperative interviewer | Clarify, then tests first, then build module by module, with many "why this?" questions |
| 2 | Requirements change mid-way and time runs short | Re-scoping, changing config not code, deciding what to cut |
| 3 | The AI produces wrong code and a test fails | Catching AI mistakes, debugging with a hypothesis, "how do you know it's right?" |
| 4 | The interviewer drills into scale, PySpark and the semantic layer | Walking through `spark_resolve.py`, skew, UDFs, embeddings/LLM, evaluation |
| 5 | A quiet interviewer who says "your call" to everything | Making assumptions explicit, driving the session yourself |

After the trajectories there's a **"why this / why that" answer bank**.

---

## Trajectory 1: the standard run

**00:00**

**I:** Here's the problem. CRM accounts from a customer need to be matched to our canonical company table. Take a few minutes to read it and ask anything.

**C:** *(reads for ~2 min)* A few questions before I write anything.
1. Precision or recall? The brief says Demandbase prefers false negatives. So when unsure we return "no match", and a human or another process reviews it later. Is that right?
2. Is it one canonical company per CRM row, or can a row legitimately map to several?
3. Rough data sizes: are we talking thousands of CRM rows per tenant, or millions?
4. If two canonical companies tie exactly, should I pick one or reject?

**I:** Precision, yes. One company per row. A tenant can have a few million rows, but for today assume it fits in memory. On ties, what would you do?

**C:** I'd reject by default and make that a config switch. A wrong match flows into intent and scoring downstream, while a missing match just shows up in a review queue.

**I:** Fine.

**00:05: interfaces first**

**C:** I'll write the skeleton myself before involving the AI. That way the AI fills in functions whose contracts I've already decided.

```
[screen] er/normalize.py  -> normalize_domain(value) -> str | None, normalize_name(value) -> str, normalize_country
         er/blocking.py   -> build_blocks(df) -> dict[key, list[pos]]
         er/scoring.py    -> score_pair(crm, canon, cfg) -> MatchScore(score, method, reason)
         er/resolve.py    -> resolve(crm_df, canon_df, cfg) -> DataFrame
         er/config.py     -> MatchConfig(threshold=0.85, weights=...)
```

**I:** Why a config dataclass instead of just constants at the top?

**C:** Three reasons. Tests can pass a different config without monkeypatching. Every tunable is in one place when someone asks "why did this match?". And it's frozen and validated in `__post_init__`, so bad values like weights not summing to 1 or a threshold of 1.5 fail at startup instead of producing silent garbage.

**00:08: tests first for normalisation**

**C:** Normalisation is where most bugs hide, so I'll write the test table first and give it to the AI as the target.

```
[screen] tests/test_normalize.py
  ("https://www.acme.com/about", "acme.com"),
  ("WWW.ACME.COM:443", "acme.com"),
  ("initech.co.in", "initech.co.in"),     # not "co.in"
  ("gmail.com", None),                     # free-mail isn't company evidence
  ("", None), (None, None), (np.nan, None),
  ("münchen.de", "xn--mnchen-3ya.de"),
```

```
[AI] Implement normalize_domain(value) -> str | None in er/normalize.py only.
     Contract: never raises; None/NaN/non-str -> None. Use urllib.parse, no new deps.
     Handle multi-part suffixes like co.in / co.uk with a small set; free-mail -> None.
     Make tests/test_normalize.py pass; run pytest and show output.
     List any input you didn't handle.
```

**[screen]** The AI returns ~30 lines using `urlsplit(...).hostname`, then strips `www.`, encodes with IDNA, and applies the suffix set. 22/23 tests pass; `"http://[bad"` raises `ValueError`.

**C:** `urlsplit` raises on an unbalanced bracket. The contract says "never raises", so I'll wrap that one call in `try/except ValueError`. I'm deliberately not wrapping the whole function, because a broad `except` would hide real bugs.

**I:** Why `urlsplit` and not a regex or `split("/")`?

**C:** The split version keeps `www.` and ports, and breaks on `user@host`. A regex for URLs is hard to read and still misses cases. `urlsplit` is standard library and already handles scheme, port, credentials and path. The one trick is prefixing `//` when there's no scheme, or `hostname` comes back empty.

**I:** And why not `tldextract`?

**C:** In production I would use it, with an offline snapshot of the Public Suffix List. For a 75-minute exercise, a small hand-rolled set with a comment saying so is easier to explain and has no network dependency at import time.

**00:18: names**

**C:** For names, the trap is removing "Corp" with `replace`. That turns "Corporate Dynamics" into "orate Dynamics". I'll ask for token-level removal of *trailing* legal suffixes, and never remove the last token.

**I:** Why never the last token?

**C:** A company literally called "Holdings" or "Inc" would otherwise normalise to an empty string. Empty names then either match everything or crash similarity functions.

**00:25: blocking**

**I:** Why not just compare every CRM row with every company? It's only a few thousand rows.

**C:** For the sample, sure. But a tenant with 1M rows against a 5M universe is 5×10¹² pairs. So I block: a pair is only compared if it shares an exact registrable domain or the first four characters of the first name token. I'll add a test asserting that the candidate count stays under 2% of n×m on synthetic data, and that the true match is never blocked out.

**I:** Why isn't country part of the blocking key? It would make blocks smaller.

**C:** Country is missing on a lot of CRM rows. If it were in the key, a row with no country could never meet its true match, which is a silent false negative. I use it as a scoring signal instead. If blocks get too big, I'd add country to the key only when both sides have it.

**00:30: scoring**

**C:** An exact domain match scores 1.0. Otherwise it's a weighted sum: 0.65 × name similarity + 0.25 × country agreement + 0.10 × employee-size compatibility. That's capped at 0.99 so a fuzzy match never outranks an exact domain. There's a 0.08 penalty when both domains are known but differ.

**I:** Why those weights?

**C:** Honestly, they're hand-set priors, and I'd fit them on labelled pairs in production. But I chose them against specific cases. A bare name match with nothing else known scores 0.65 + 0.125 + 0.05 = 0.825, which is under 0.85, so it's rejected. An identical name in a different country scores 0.75, also rejected. Both of those are tests, so if someone changes the weights, the tests tell them which behaviour they broke.

**I:** Why is a missing signal 0.5 and not 0?

**C:** Zero would punish missing data as if it were disagreement. Skipping the term and renormalising would inflate the score when we know less. 0.5 means "no evidence either way".

**I:** Why the mean of Jaro-Winkler and token-sort ratio?

**C:** Jaro-Winkler rewards shared prefixes, which suits company names and typos at the end. Token-sort ratio handles word order, like "Generale Societe" vs "Societe Generale". Averaging them is simple and I can explain it. I'd only go fancier with labelled data to show it helps.

**00:45: resolve and determinism**

```
[AI] Implement resolve() in er/resolve.py using the existing modules. For each CRM row:
     candidates from blocks -> score_pair -> pick best. Sort by (-score, -name_sim, company_id)
     so the result never depends on row order. Reject below cfg.threshold and on exact ties
     when cfg.on_tie == "reject". Output one row per CRM row, input order, with reason.
     Don't use DataFrame.apply(axis=1) or a cross join.
```

**C:** *(reading the output)* It uses `idxmax` in one place. I'm replacing that with the explicit sort, because `idxmax` returns the first max in whatever order rows arrive, and that isn't deterministic across shuffles. I'll add a test that shuffles both inputs five times and uses `assert_frame_equal`.

**01:00: wrap-up**

**C:** 92 tests pass. On the sample, 001A and 001B go to C1 by domain. 001C goes to C2 with 0.91: name sim 1.0, country match, capped at 0.99, minus 0.08 for .fr vs .com. 001D is rejected because gmail isn't evidence and "Globex" has no candidates. 001E goes to C3 with ultimate parent C9. The duplicates report shows 001A and 001B as one company.

**I:** The brief's expected output says name similarity 0.93 for 001C. You got 1.00. Why?

**C:** I strip "SA" as a legal suffix, so both sides normalise to "societe generale". The final score is still 0.91 because the fuzzy cap and the domain penalty dominate. If they wanted 0.93, the reference probably kept "sa", and I think stripping it is more correct.

---

## Trajectory 2: requirements change and time runs short

**00:20**, after normalisation and blocking are done:

**I:** Change of plan. Product says this feeds the *intent* pipeline, where missing an account is expensive. They'd rather have more matches and review the doubtful ones.

**C:** So recall matters more now, with a review queue. I don't want to touch scoring logic for that. Three options, cheapest first:
1. Lower `threshold` in `MatchConfig`, say to 0.80.
2. Add a middle band: 0.75–0.85 becomes `method="needs_review"` instead of `rejected`, so downstream can choose.
3. Change tie handling to `on_tie="lowest_id"`, or better, prefer the parent company in a tie.

I'd do option 2. It keeps high-confidence matches clean and doesn't hide uncertainty. Is a review queue realistic for them?

**I:** Let's say yes. But you have 25 minutes left, and `resolve` and the duplicates part aren't written.

**C:** Then I'll prioritise. (The `needs_review` band isn't in the repo; it's the change you'd make live.)
- **Must have:** `resolve` with threshold, reasons and determinism, plus the golden-sample test.
- **Should have:** the `needs_review` band. It's a few lines in `_decide` plus one test.
- **Cut to talk only:** the hypothesis property test, the ultimate-parent cycle handling and the CLI. I'll describe them instead of building them.

```
[AI] In er/resolve.py _decide(): if cfg.review_threshold <= best.score < cfg.threshold, return
     company_id=best id, method="needs_review", reason prefixed with "review: ".
     Add review_threshold: float = 0.75 to MatchConfig, validated review_threshold <= threshold.
     Add one test: identical name, country unknown (score 0.825) -> needs_review.
     Change nothing else.
```

**I:** Why tell it "change nothing else"?

**C:** Assistants tend to "improve" nearby code. That makes the diff harder to review and can break things I've already tested. A small diff is something I can check in 30 seconds.

**I:** Why put the review threshold in the config and not an argument to `resolve`?

**C:** It belongs with the other thresholds, and validation happens in one place. An argument would let callers pass a review threshold above the match threshold.

**00:50**

**I:** You have 10 minutes. What would you still do?

**C:** Run the golden sample and the shuffle test, which is done. Then I'd write down what I cut, as a TODO list and not a promise: cycle-safe parent walk, property tests, CLI. And one risk: prefix-4 blocking misses typos in the first four characters. I'd rather hand over something smaller that I've tested than something bigger that I haven't.

---

## Trajectory 3: the AI gets it wrong

**00:12**

```
[AI] Write normalize_name that lowercases, strips accents and removes legal suffixes like Corp, Inc, Ltd, SA.
```

**[screen]** The AI returns:

```python
for suffix in ["corp", "inc", "ltd", "sa", "llc"]:
    name = name.replace(suffix, "")
```

**C:** I won't accept this. It's substring replacement, so "Corporate Dynamics" loses "corp" and "Santander" loses "sa". It also doesn't handle `None`. That's my fault for a vague prompt. I'll give it the test table instead.

```
[AI] Rewrite normalize_name to satisfy these cases exactly (paste table). Remove legal suffixes only
     as whole trailing tokens, never the last remaining token. Non-str -> "". No regex over 1 line.
```

**I:** How do you know the new version is right, beyond the tests passing?

**C:** I read every branch and ask which test hits it. There's a `the` prefix removal with no test, so I'm adding "The Home Depot, Inc." → "home depot". Then I try one input the tests don't cover, an emoji, to check it doesn't crash. And there's a hypothesis property test: normalising twice gives the same result as normalising once.

**00:35**

```
[AI] Implement resolve() ... (no constraints given about performance)
```

**[screen]** The AI returns `crm.merge(canon, how="cross")` followed by `.apply(score, axis=1)`.

**C:** That's correct on the sample and unusable at real size: 1M × 5M rows in memory, plus a Python function per row. I'll discard it and point it at the blocking module I already wrote. From now on, my prompts will say "no cross join, no apply(axis=1)" up front.

**00:48: a test fails**

**[screen]** `test_country_mismatch_pushes_identical_names_below_threshold`: `assert 0.85 < 0.85` fails.

**C:** Before asking the AI, here's my hypothesis. The test is identical names, different country, equal employee counts. With weights 0.75/0.15/0.10, that's 0.75 + 0 + 0.10 = 0.85, exactly the threshold, and the comparison is `>=`. So the test is right and the weights are wrong. A country mismatch should never be outweighed by name alone.

**I:** Why not change `>=` to `>`?

**C:** That fixes this exact number and nothing else. With a name sim of 0.99 it would pass again. The real problem is that country weighs too little. I'll move it to 0.65/0.25/0.10, so the same case scores 0.75 and the bare-name case scores 0.825, both below threshold. Then I rerun the whole suite, not just this test.

```
[AI] Test X fails: identical name + country mismatch scores exactly 0.85 == threshold.
     Change Weights defaults to name=0.65, country=0.25, employees=0.10. Don't change score_pair
     or the threshold. Run the full suite and show output.
```

**I:** You told it exactly what to change. What's the AI adding, then?

**C:** Here, speed on the mechanical part: editing, updating dependent assertions, rerunning. I don't delegate the diagnosis. If I paste the error and say "fix it", it's just as likely to special-case the test or change the threshold. Making that judgement is my job.

**00:55**

**[screen]** `SettingWithCopyWarning` from AI-written code: `df[df.country == "US"]["country_norm"] = "US"`.

**C:** That's chained assignment. It writes to a temporary copy and does nothing. I'll replace it with `df.loc[mask, "country_norm"] = ...`. Better still, the normaliser already returns a new column via `map`, so the line isn't needed at all. I'm deleting it.

---

## Trajectory 4: scale, PySpark and the semantic layer

**00:55**, with the pandas solution done:

**I:** Say it's 50M CRM rows across tenants against 5M canonical companies. What changes?

**C:** The decisions stay the same. The execution changes, and I've sketched it in `er/spark_resolve.py`. It reuses the same pure normalisers and `score_pair`, so the two versions can't drift. There's a parity test that runs both on 200 random rows and asserts identical output.

**I:** Walk me through it.

**C:**
1. **Normalise**: `pandas_udf` wrappers around `normalize_domain` and friends. Arrow sends a batch of values to Python at once, instead of one row at a time like a plain UDF.
2. **Block**: build an array of keys, `d:<domain>` and `p:<prefix>`, then `explode` so each (record, key) is one row. An equi-join on `block_key` gives candidate pairs, and `dropDuplicates` handles pairs that share both keys.
3. **Score**: join the normalised attributes back on, then one `pandas_udf` returning a struct that calls the same `score_pair`.
4. **Decide**: a `Window` partitioned by `crm_id`, ordered by score desc, name_sim desc, company_id asc. `row_number() == 1` picks the best. `lead()` gives the runner-up for tie detection.
5. **Keep unmatched rows**: left-join from all CRM ids, and `coalesce` fills in "rejected".

**I:** Why `pandas_udf` for scoring and not native Spark functions? Spark has `levenshtein`.

**C:** Native functions are faster, and for a plain Levenshtein I'd use them. But the score here is Jaro-Winkler plus token-sort, with neutral handling of unknowns and a readable reason string. Rewriting that as Column expressions would duplicate logic I've already tested. A `pandas_udf` keeps one source of truth, and Arrow batching makes the Python cost acceptable. The trade-off is serialisation overhead per batch. If profiling showed scoring dominated, I'd first move name similarity into native code.

**I:** What breaks first at 50M?

**C:** Skew. Prefix blocks like `p:inte` or `p:glob` can hold tens of thousands of companies, and one task ends up doing all the work. I handle the worst case by pruning prefix blocks above `max_block_size`: I count per key, broadcast the hot keys and do a `left_anti` join. I never prune domain blocks. The more complete fix is salting: add a random salt 0..N to the CRM side, replicate the canonical side N times, and join on `(key, salt)`. On Spark 3+ I'd also check that AQE skew-join handling is on.

**I:** Why `broadcast` there?

**C:** The hot-key table is tiny, maybe hundreds of rows. Broadcasting sends it to every executor, so the big canonical side doesn't need a shuffle for that join.

**I:** Anything that surprised you writing the Spark version?

**C:** Yes, a real bug. With a single candidate, `lead()` returns NULL, and `NULL == x` is NULL, not False. So `tied` was NULL, `~rejected` was NULL, and `when(NULL, company_id)` silently produced NULL. Every single-candidate match disappeared, with no error. The parity test caught it. The fix is `coalesce(tied, False)`. It's the classic SQL three-valued-logic trap, and exactly the kind of thing that passes a quick look at AI-generated code.

**I:** Ultimate parent at this scale?

**C:** I collect `company_id → parent_id` to the driver and walk it in Python. At 5M rows that's a few hundred MB, which is fine. If it grew or ran often, I'd do pointer jumping: repeatedly self-join `parent = parent.parent` until nothing changes. That's O(log depth) joins. Or use GraphFrames connected components.

**I:** The team works on the semantic layer: embeddings, ranking, RAG. Where would that fit?

**C:** Two places, both behind the current rules and not replacing them.
- **Recall for no-domain rows**: embed "name + city + industry" with a bi-encoder, and use ANN top-k (FAISS or a vector DB) as a third blocking key. Only for rows with no usable domain, since domain is already near-perfect.
- **Adjudication for the grey zone**: scores of 0.75–0.90 go to an LLM with both records and the evidence. It returns structured JSON `{same_entity, confidence, reason}`, cached by pair hash, with a budget cap, and I'd log where it disagrees with the rules.

The rules stay first because they're cheap, deterministic and explainable. That matters when a customer asks why their account matched.

**I:** How would you know any of this is better?

**C:** A labelled set of pairs, stratified by method and score band. I'd report precision and recall per `method`, pick the threshold from the PR curve at our target precision, and measure cluster-level quality (B-cubed) for the duplicates. Then compare before and after for each change, like embeddings or the LLM step, on the same set.

---

## Trajectory 5: the quiet interviewer

**00:00**

**I:** Here's the problem. Use whatever tools you like. I'll mostly watch.

**C:** Okay. I'll say my assumptions out loud, and please stop me if any is wrong.
1. Precision over recall, because the brief says so.
2. One match per CRM row.
3. It fits in memory.
4. Ties are rejected.

**I:** Your call.

**C:** Then I'll write those four into `MatchConfig` and the README, so they're visible, not hidden in code.

*(Throughout, keep narrating briefly: what you're about to do, why, and what you're checking.)*

**00:20**

**C:** I'm checking blocking with a count, because a wrong block key fails silently with fewer matches and no error. The test asserts that every true match is in its candidate set.

**00:40**

**I:** *(first real question)* How do you use AI in your normal work?

**C:** I use it most for the mechanical middle: implementing against a spec, writing test tables, refactoring. I spend my own time on the edges. Up front, that's clarifying, interfaces and deciding what "correct" means as tests. At the end, it's reading the diff, trying inputs it didn't think of, and a separate review pass in a fresh context. The key habit is giving the model something it can run, like tests, so I'm reviewing evidence rather than being the only check.

**I:** What would you not use it for?

**C:** Decisions that are really product choices: thresholds, tie policy, what counts as a match. Also anything I can't verify quickly. If I can't explain a generated line, I don't keep it.

**01:10**

**I:** We're at time. Anything to add?

**C:** Three known gaps. Blocking misses typos in the first four characters. Weights aren't fitted to data. And subsidiaries that share a parent's domain are decided by name similarity alone. Each has a test or a note in the README.

---

## "Why this / why that" answer bank

Short answers you can give straight away. All numbers match the code.

| Question | Answer |
|---|---|
| Why pure functions for normalisation? | They're trivially testable, reusable in Spark `pandas_udf`s unchanged, with no hidden state. |
| Why `isinstance(value, str)` first in every normaliser? | Real columns contain `None`, `NaN`, `pd.NA` and numbers. One guard makes them all return `None`/`""` instead of raising. |
| Why NFKD + remove combining marks? | It splits "é" into "e" plus an accent mark, and then drops the mark. `casefold` rather than `lower` handles cases like German ß. |
| Why free-mail domains → None? | gmail.com says nothing about the company. Treating it as a domain would merge unrelated accounts. |
| Why punycode (IDNA)? | Canonical data usually stores ASCII domains, so the same domain in Unicode or ASCII must compare equal. |
| Why exact domain = 1.0 even if the names differ? | A registrable domain is the strongest identifier we have. The one exception is subsidiaries sharing a domain; then there are several candidates and name similarity breaks the tie (tested). |
| Why cap fuzzy at 0.99? | So a fuzzy match can never outrank or tie an exact domain match. |
| Why the domain-mismatch penalty? | Both sides having different known domains is weak evidence against a match (.fr vs .com is still allowed through). |
| Why log-space for employees? | 100 vs 200 is a big difference, 100k vs 100.1k isn't. Ratios matter, not differences. `log1p` makes 0 safe. |
| Why `name_sim` as a secondary sort key? | Between two equal scores, prefer the closer name before falling back to id. |
| Why reject ties instead of picking one? | A wrong match costs more than a missing one here, and `on_tie="lowest_id"` is available if the business disagrees. |
| Why union-find for duplicates? | It's near-linear and extends naturally if later we also link CRM rows to each other directly (for example, same domain but unmatched). |
| Why check `crm_id` is unique and non-null? | Output is one row per id. Duplicates would silently double-count downstream. A clear `ValueError` is better. |
| Why `dtype=str` when reading CSVs? | So ids like "001A" or "007" don't become numbers or lose leading zeros. |
| Why `logging` and not `print` in the library? | The caller controls levels and handlers. Only the CLI prints. |
| Why no classes except small dataclasses and `UnionFind`? | Functions compose and test more easily. Classes are used where there is real state. |
| Why pytest parametrize tables? | The edge cases become readable documentation, and they're exactly what I paste into the AI as the spec. |
| Why a hypothesis test? | It finds inputs I didn't think of, and checks properties like idempotence and "never raises". |
| Why pandas and not Polars? | The brief and team use pandas, and it's not the bottleneck here. Polars would be a reasonable swap for normalisation speed. |
| Why not an ML model for matching? | No labelled data today. The rules are explainable. The features I compute are exactly what a model would use later. |
| How long does it take? | 20k × 20k synthetic takes 0.7 s, with ~21k comparisons instead of 400M. The names there are unique, so treat it as a sanity check, not a benchmark. |
