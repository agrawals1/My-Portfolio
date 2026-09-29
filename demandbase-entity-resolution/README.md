# Problem 1 — Company Entity Resolution (CRM → canonical company universe)

Reference solution for the Demandbase "AI-assisted coding" round. It is deliberately **small enough to read,
re-type and explain in the time box** (~450 lines of core code, 92 tests, runs in <1 s). It is *not* a
production system — it is what "production-shaped" looks like in 75 minutes.

```
er/
  config.py     MatchConfig + Weights (frozen dataclasses, validated in __post_init__)
  normalize.py  normalize_domain / normalize_name / normalize_country / normalize_frame   (pure functions)
  blocking.py   block_keys / build_blocks / candidate_positions                            (no cross join)
  scoring.py    Record, MatchScore, name_similarity, employee_compat, score_pair            (pure, no pandas)
  resolve.py    validate -> normalise -> block -> score -> decide  (+ ultimate parent)
  clusters.py   UnionFind + find_duplicates
  spark_resolve.py  same pipeline in PySpark (explode+join blocking, pandas_udf scoring, Window pick)
  cli.py        python -m er.cli --crm sample_data/crm_accounts.csv --canonical sample_data/canonical_companies.csv
tests/          normalize · blocking · scoring · resolve (golden sample, shuffling, ties, cycles, bad input)
```

```bash
pip install -r requirements.txt
python -m pytest -q                       # 92 passed (+3 Spark parity tests if pyspark is installed)
pip install -e '.[spark]'                 # optional: pyspark 4.x + Java 17/21
python -m er.cli --crm sample_data/crm_accounts.csv --canonical sample_data/canonical_companies.csv
```

## The 60-second design pitch (say this out loud first)

1. **Normalise** (pure functions): domain → registrable domain, name → accent-stripped/casefolded/legal-suffix-free tokens, country → ISO2.
2. **Block**: two keys per company — exact registrable domain, and first-4-chars of the first name token. Candidates = union of blocks. Work is ~O(n + m), not O(n·m). (20k × 20k CRM/canonical: 0.7 s, 20,868 comparisons instead of 400M.)
3. **Score** a pair: exact domain ⇒ 1.0. Otherwise `0.65·name_sim + 0.25·country + 0.10·employee_compat`, capped at 0.99 (a fuzzy match can never outrank an exact-domain one), minus 0.08 if both domains are known but differ. Missing signals are *neutral* (0.5), not zero and not skipped.
4. **Decide** (precision-first): best candidate must reach `threshold=0.85`; exact ties are **rejected** (configurable `on_tie="lowest_id"`). Every row gets `method` + human-readable `reason`.
5. **Hierarchy + dupes**: `ultimate_parent_id` by walking `parent_id` (cycle/dangling safe); CRM rows sharing a `company_id` are clustered with union-find.

## Decisions worth defending (interviewers poke at these)

| Decision | Why | Trade-off / how I'd change it |
|---|---|---|
| Fuzzy weights 0.65/0.25/0.10, unknown = 0.5 | A bare name match (all else unknown) scores 0.825 < 0.85 → **rejected**. Identical name + *different* country = 0.75 → rejected. | Real weights should be fit on a labelled pair set (logistic regression / GBM on the same features). |
| Country not in blocking key | A missing country would otherwise block out the true match (false negative). Country is a scoring signal instead. | With huge blocks, add country to the key when both known, or salt hot keys in Spark. |
| Prefix-4 name block | Cheap typo tolerance after char 4. | Typos in the first 4 chars are missed → add a 2nd key (e.g. last token prefix / phonetic / MinHash on shingles). |
| Ties rejected | False positives corrupt scoring; false negatives just get reviewed. | Business may prefer "parent wins" → add `parent_id is None` to the sort key. |
| Rejected rows have `score=0.0` and the best-candidate score lives in `reason` | Downstream can filter on `company_id` **or** `score` safely. | — |
| Hand-rolled public-suffix list | Zero dependencies, explainable. | Production: `tldextract` (offline snapshot) — say so. |
| `Series.map(fn)` per column, Python loop per CRM row | Simple; 20k×20k in <1 s. | 50M rows: PySpark UDF/pandas-UDF, blocking keys as join keys, broadcast small side. |

## Where AI-generated code typically goes wrong (all covered by tests here)

| Trap | What the code/tests do instead |
|---|---|
| `url.split("//")[1].split("/")[0]` keeps `www.` and `:443` | `urlsplit(...).hostname`, then strip `www.` (`test_normalize_domain`) |
| `domain.split(".")[-2:]` turns `initech.co.in` into `co.in` | multi-part suffix set (`initech.co.in`, `bbc.co.uk`, bare `co.uk` → None) |
| `name.replace("Corp", "")` → "Corporate Dynamics" ⇒ "orate Dynamics" | strip legal suffixes as whole trailing *tokens*, never the last token (`Inc` stays `inc`) |
| Free-mail domain (gmail.com) treated as company evidence | free-mail ⇒ `None` domain; test proves no domain match |
| `merge(how="cross")` / `apply(axis=1)` over all pairs | blocking + tiny candidate lists; test asserts candidates < 2 % of n×m and true match never blocked out |
| `idxmax` with no tie-break ⇒ nondeterminism | sort key `(-score, -name_sim, id)`; ties rejected; shuffle test uses `assert_frame_equal` |
| `NaN` employees, `log(0)` | `employee_compat` returns `None` for unknown, uses `log1p`; tests for 0, NaN, negative |
| `NaN`/`None`/`pd.NA`/numbers into string functions | every normaliser starts with `isinstance(value, str)`; hypothesis test: idempotent & never raises for any text |
| Chained assignment (`df[df.x=="US"]["y"] = …`) | we never mutate inputs; `normalize_frame` works on a copy |
| Sample says 001C = 0.91, `name sim 0.93` | We get 0.91 too (0.99 cap − 0.08 domain-mismatch penalty); our name sim is 1.00 because "SA" is stripped — say this if asked |

**A bug my own tests caught while building this** (great interview anecdote): the first weights (0.75/0.15/0.10) gave
*identical name + different country* a score of exactly 0.85 = threshold → accepted. Fixed by raising the country weight; the test
`test_country_mismatch_pushes_identical_names_below_threshold` now pins the behaviour. Lesson: boundary values need tests, and
"looks right on the sample" is not verification.

## 75-minute plan (from the brief)

| min | do | note |
|---|---|---|
| 0–8 | Clarify + interfaces | Ask: precision vs recall? tie policy? is CRM→canonical 1:1? size of data? Write the 5 function signatures + `MatchConfig` before any implementation. |
| 8–18 | `normalize.py` **tests first** | Paste the parametrised table (edge cases from the brief), let AI implement, run, read every branch. |
| 18–28 | `blocking.py` + test | Show the candidate-count assertion. |
| 28–43 | `scoring.py` + `config.py` | Explain weights + neutral-unknown + cap. |
| 43–58 | `resolve.py`, `clusters.py` | Determinism + `reason`. |
| 58–65 | Run sample, fix, `simplify` pass | |
| 65–75 | Edge cases + discussion | Use the tables above; mention Spark / embeddings / eval. |

See also: [INTERVIEW_TRAJECTORIES.md](INTERVIEW_TRAJECTORIES.md) (mock interviews) · [PYSPARK_CRASH_COURSE.md](PYSPARK_CRASH_COURSE.md) · [AI_CODING_PLAYBOOK.md](AI_CODING_PLAYBOOK.md)

## Stretch talking points (they mentioned LLMs, semantic ranking, RAG)

* **Scale (50 M × 5 M)**: PySpark, blocking keys as join keys, salt hot keys (e.g. "consulting"), MinHash-LSH on name shingles, broadcast the small side.
* **Semantic layer**: add a bi-encoder embedding of "name + description + industry" with ANN (FAISS/pgvector) as a *third blocking key* only for rows with no domain; re-rank the top-k with the fuzzy score. Cache embeddings, budget the calls.
* **LLM as adjudicator, not matcher**: only for the grey zone (score 0.75–0.90) — send the two records + evidence, require structured JSON `{same_entity, confidence, reason}`, cache by pair hash, cap the budget, log disagreements.
* **Evaluation**: labelled pair set → precision/recall per `method`, PR curve to pick the threshold (weighted toward precision), pairwise vs cluster-level (B-cubed) metrics, monitor drift.
* **Hierarchy**: roll subsidiaries up to the ultimate parent for ABM targeting (already implemented: `ultimate_parent_id`).
