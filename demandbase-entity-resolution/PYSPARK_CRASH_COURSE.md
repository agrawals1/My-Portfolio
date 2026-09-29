# PySpark crash course (everything used in `er/spark_resolve.py`)

The goal is to understand every Spark line in this repo and answer follow-ups about it. Read sections 1–3 first; after that, each section maps to specific lines of the code.

```bash
pip install "pyspark>=4.0" pyarrow        # needs Java 17 or 21 on PATH
python -m pytest -q tests/test_spark_resolve.py
```

---

## 1. Mental model (the 5 things that explain 90% of Spark)

1. **Driver and executors.** Your Python script is the *driver*. It builds a plan. *Executors* (JVM processes on many machines) run the plan over the data. On a laptop, `master("local[2]")` means one process with two worker threads.
2. **Partitions.** A DataFrame is split into partitions (chunks of rows). One task processes one partition, so parallelism equals the number of partitions.
3. **Lazy evaluation.** `select`, `filter`, `join` and `withColumn` are *transformations*: they only add to a plan and return instantly. Nothing runs until an *action*: `collect()`, `count()`, `toPandas()`, `write...`, `show()`.
4. **Narrow vs wide transformations.**
   - *Narrow* (`select`, `filter`, `withColumn`, `explode`): each output partition depends on one input partition, with no data movement.
   - *Wide* (`join`, `groupBy`, `dropDuplicates`, `Window.partitionBy`, `distinct`): rows with the same key must end up on the same machine. That's a **shuffle**, which means network, disk and sorting. **Shuffles are the main cost.** Performance work is mostly about removing them or making them smaller.
5. **The optimizer (Catalyst) rewrites your plan.** It pushes filters down, prunes columns and picks join strategies. `df.explain()` shows what will actually run.

```python
spark = SparkSession.builder.master("local[2]").getOrCreate()
df = spark.read.csv("x.csv", header=True)   # plan only (header inference may peek at the file)
df2 = df.filter(F.col("country") == "US")    # plan only
df2.count()                                   # ACTION -> runs the job
```

---

## 2. SparkSession and test setup (`tests/test_spark_resolve.py`)

```python
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
session = (SparkSession.builder.master("local[2]").appName("er-tests")
           .config("spark.sql.shuffle.partitions", "4")
           .config("spark.ui.enabled", "false")
           .getOrCreate())
```

| Line | Why |
|---|---|
| `PYSPARK_PYTHON` | Python UDFs run in separate *worker* Python processes. They must use the same interpreter and packages (pandas, rapidfuzz, the `er` package) as the driver. Otherwise you get `ModuleNotFoundError: pyarrow` inside a task. This happened while building this repo. |
| `local[2]` | Two threads, which is enough to exercise real partitioning and shuffles in tests. |
| `spark.sql.shuffle.partitions=4` | Every shuffle creates this many partitions (default **200**). With tiny test data, 200 near-empty tasks are slow. In production you'd size it so each partition is ~100–200 MB, or let AQE coalesce them (§11). |
| `scope="module"` fixture + `session.stop()` | Starting a JVM takes seconds, so do it once per test module. |
| `crm.sparkSession` (in `spark_resolve`) | Every DataFrame knows its session, so there's no need to pass `spark` around. |

---

## 3. Columns and expressions (most of the file)

`from pyspark.sql import functions as F` is the standard import. Spark code builds **Column expressions**, which run in the JVM. They don't execute Python per row.

```python
F.col("name")               # reference a column
F.lit("d:")                 # a constant as a column
F.concat(F.lit("d:"), F.col("domain_norm"))
F.col("x").cast("double")   # bad strings -> NULL (no exception), like pd.to_numeric(errors="coerce")
F.col("x").alias("y")       # rename in a select
F.when(cond, a).when(cond2, b).otherwise(c)   # SQL CASE WHEN; without otherwise() -> NULL
F.coalesce(a, b, c)         # first non-NULL
F.round("s.score", 6)       # "s.score" = field `score` of struct column `s`
F.format_string("scored %.2f", "score_r")      # printf into a string column
```

**`select` vs `withColumn`.** `select` defines the full output column list, as in `normalize_df`. `withColumn` adds or replaces one column. A long chain of `withColumn` calls is fine for a few columns. For dozens, use one `select`, because each call adds a plan node.

### NULL semantics: the bug worth knowing

Spark uses SQL three-valued logic. **`NULL == x` is `NULL`, not `False`.** In `when(cond, ...)`, `filter(cond)` and `~cond`, a NULL condition behaves like "not true".

The bug in this repo (`_pick_best`, now fixed):

```python
tied = (F.col("next_score") == F.col("score_r")) & ...  # next_score is NULL when only 1 candidate
rejected = below | tied        # False | NULL  -> NULL
F.when(~rejected, F.col("company_id"))                   # ~NULL -> NULL -> company_id becomes NULL!
```

Every CRM row with exactly one candidate silently lost its match, with no error. The parity test against pandas caught it. The fix:

```python
tied = F.coalesce(same, F.lit(False))
```

Rules of thumb:
- Wrap nullable booleans in `coalesce(..., lit(False))`.
- Use `isNull()` / `isNotNull()`, never `== None`.
- For null-safe equality use `a.eqNullSafe(b)`, which is SQL `<=>`.

---

## 4. Strings, arrays and `explode`: blocking (`with_block_keys`)

```python
first_token = F.split(F.col("name_norm"), " ").getItem(0)       # array -> first element
F.substring(first_token, 1, prefix_len)                          # NOTE: 1-based position!
domain_key = F.when(F.col("domain_norm").isNotNull(), F.concat(F.lit("d:"), F.col("domain_norm")))
prefix_key = F.when(F.col("name_norm") != "", F.concat(F.lit("p:"), ...))
keys = F.filter(F.array(domain_key, prefix_key), lambda k: k.isNotNull())   # drop NULL keys
df.withColumn("block_key", F.explode(keys))
```

- `F.array(a, b)` builds an array column from two columns.
- `F.filter(array, lambda)` is a **higher-order function**. The lambda does *not* run in Python. It is translated into a JVM expression, so it's fast.
- `F.explode(arr)` produces **one output row per array element**. A row with 2 keys becomes 2 rows, and a row with an empty array disappears. Use `explode_outer` to keep it with NULL instead. This is the Spark equivalent of the pandas `blocks` dict: instead of a dict from key to list, you get a long table of (record, key) rows, and a join replaces the dict lookup.
- `substring` is 1-based, unlike Python slicing. That's a classic off-by-one.

---

## 5. Joins (`candidate_pairs`, `spark_resolve`)

```python
crm_k.join(canon_k, "block_key")                    # inner equi-join on same-named column
crm_n.select(...).join(best, "crm_id", "left")      # keep all CRM rows (rejected ones too)
canon_k.join(F.broadcast(hot), "block_key", "left_anti")   # rows of canon_k whose key is NOT in hot
pairs.join(c, pairs.crm_id == F.col("c.id"))        # expression join using an alias
```

| Join type | Keeps |
|---|---|
| `inner` (default) | rows matching on both sides |
| `left` | all left rows, NULLs where there's no match |
| `left_anti` | left rows **without** a match. It's a filter ("NOT IN") and adds no columns. |
| `left_semi` | left rows **with** a match ("IN"), adding no columns |

**Joining on a string name vs an expression.**
- `join(other, "key")` merges the key into one column.
- `join(other, a.key == b.key)` keeps both columns, which gives ambiguous names. That's why the code uses `crm_n.alias("c")` and `canon_n.alias("k")`, then selects `c.name_norm` and `k.name_norm`. Joining a DataFrame with a relative of itself without aliases gives "ambiguous column" errors.

**How Spark executes a join.**
- **Sort-merge join (default for big/big):** both sides are shuffled by key, then merged. That's two shuffles.
- **Broadcast hash join:** the small side is copied whole to every executor, and the big side isn't shuffled at all. Spark does this automatically below `spark.sql.autoBroadcastJoinThreshold` (10 MB by default). `F.broadcast(df)` forces it.
- In this code, `F.broadcast(hot...)` (the hot keys, tiny) and `F.broadcast(_parent_table(...))` are broadcast. Don't broadcast something that doesn't fit in executor memory, or you'll get an OOM.

**Why join and not cross join?** A cross join of 50M × 5M is 2.5×10¹⁴ rows. The equi-join on `block_key` only produces pairs that share a key. The explode-then-join pattern *is* blocking.

---

## 6. `groupBy`, `count`, `dropDuplicates` (skew handling)

```python
sizes = canon_k.groupBy("block_key").count()                          # rows per key (a shuffle)
hot = sizes.filter((F.col("count") > max_block_size) & F.col("block_key").startswith("p:"))
...
.select("crm_id", "company_id").dropDuplicates()   # a pair can share both a d: and a p: key
```

- `groupBy(...).agg(F.count("*"), F.max("x"), F.collect_list("y"))` is the general form. `.count()` is shorthand.
- **Skew:** one key with huge cardinality (`p:inte` for "International ...") means one task gets most of the rows. The stage waits for that one slow task. On the Spark UI, you see one task taking 100× longer.
- Remedies, in order:
  1. **Prune hot blocks.** That's what this code does, for prefix keys only. Never prune domain keys.
  2. **AQE skew join.** Spark 3+ can split skewed partitions automatically (§11).
  3. **Salting.** Spread a hot key across N sub-keys:

```python
N = 16
crm_s   = crm_k.withColumn("salt", (F.rand(seed=42) * N).cast("int"))
canon_s = canon_k.withColumn("salt", F.explode(F.array([F.lit(i) for i in range(N)])))  # replicate N x
pairs   = crm_s.join(canon_s, ["block_key", "salt"])
```

Every CRM row lands in exactly one salt bucket, and every canonical row appears in all buckets, so no pairs are lost. The work for the hot key is split across N tasks. The cost is that the canonical side becomes N× bigger, so salt only the hot keys.

---

## 7. UDFs: `udf` vs `pandas_udf` vs built-ins

| | Runs | Speed | Use when |
|---|---|---|---|
| Built-in `F.*` | JVM, optimised | fastest | always, if a function exists (`F.levenshtein`, `F.lower`, `F.regexp_replace`, ...) |
| `@F.pandas_udf` | Python, **batch** of rows as `pd.Series` via Apache Arrow | medium | custom logic, or reusing Python libraries like `rapidfuzz` |
| `@F.udf` | Python, **one row at a time** (pickle) | slowest | avoid. Legacy code only. |

The normalisers in the code:

```python
@F.pandas_udf(T.StringType())
def _domain_udf(s: pd.Series) -> pd.Series:
    return s.map(normalize_domain)
```

- The type hints `pd.Series -> pd.Series` tell Spark it's a *scalar* pandas UDF: same length out as in.
- The return type (`T.StringType()`) must be declared. Spark can't infer it.
- `s.map(normalize_domain)` still calls Python per value. The win is that data crosses JVM↔Python in Arrow batches of about 10k rows, not row by row. The bigger win is **reusing the exact tested function**, so pandas and Spark can't drift.

**Returning several values: a struct.** `SCORE_SCHEMA` is a `StructType` with four fields. A pandas UDF with a struct return type must return a **`pd.DataFrame`** whose columns match the struct fields. The result is one struct column (`.alias("s")`), and you read fields with `"s.score"` or `F.col("s")["score"]`.

**Closures and serialisation.** `_make_score_udf(cfg)` defines the UDF inside a function so it can use `cfg`. Spark pickles the function *and everything it references* and ships it to the executors. That's why `cfg` must be picklable; frozen dataclasses are. Never capture a SparkSession, a DataFrame or an open connection inside a UDF.

**NULLs in pandas UDFs.** A NULL string arrives as `None`, and a NULL double arrives as `NaN`. That's why `rec(...)` uses `isinstance(dom, str)` and `pd.isna(emp)`: the same defensive guards as the pandas code.

**Other pandas-UDF styles, if asked:**
- `groupBy(...).applyInPandas(fn, schema)` gives you a whole group as a DataFrame. You could do a whole block per call that way, though it's riskier with skewed blocks.
- `mapInPandas` works on an iterator of DataFrames per partition.

---

## 8. Window functions: picking the best candidate (`_pick_best`)

```python
w = Window.partitionBy("crm_id").orderBy(F.desc("score_r"), F.desc("sim_r"), F.asc("company_id"))
.withColumn("rank", F.row_number().over(w))       # 1, 2, 3 ... within each crm_id
.withColumn("next_score", F.lead("score_r").over(w))   # value from the next row in this order
.filter(F.col("rank") == 1)
```

- A window function computes a value **per row**, using other rows in the same partition, **without collapsing rows**. `groupBy` would collapse them.
- `partitionBy("crm_id")` causes a shuffle so that all candidates for one CRM row sit together. `orderBy` then sorts within that group.
- `row_number()` gives 1, 2, 3 with no ties. `rank()` gives 1, 1, 3 for ties, and `dense_rank()` gives 1, 1, 2.
- `lead(col)` / `lag(col)` read the next or previous row's value. They're NULL at the edge, which caused the bug in §3.
- **Determinism:** without the final `F.asc("company_id")`, rows with equal scores come out in arbitrary order, which differs between runs. That's the Spark version of "`idxmax` without a tie-break".
- `F.round(..., 6)` before comparing means floating-point noise (0.9100000001 vs 0.91) can't break or create a tie.

The same logic without a window, for comparison: `groupBy("crm_id").agg(F.max_by("company_id", "score"))` gets the argmax but can't detect ties. Hence the window.

---

## 9. `cache`, `collect`, `toPandas`, `createDataFrame`

```python
crm_n = normalize_df(crm, "crm_id", "website").cache()
```

- Because of lazy evaluation, a DataFrame used 3 times is **computed 3 times**. That would mean running the normalisation UDFs 3 times. `.cache()` (or `.persist(StorageLevel.MEMORY_AND_DISK)`) stores it the first time an action computes it.
- Cache only what is reused *and* expensive. Call `.unpersist()` when done in long jobs.
- `canon_n` is also reused, so caching it too is a reasonable extra step if asked.

```python
parent_of = {r.company_id: r.parent_id for r in canon.select("company_id", "parent_id").collect()}
spark.createDataFrame(rows, "company_id string, ultimate_parent_id string")
```

- `collect()` / `toPandas()` bring **all rows to the driver**. That's fine for a small or aggregated result. For big data it means a driver OOM. Say so when asked.
- Here, 5M (id, parent) pairs is a few hundred MB, which is acceptable. The scalable alternative is **pointer jumping**: repeatedly join `parent -> parent.parent` until no row changes, which takes O(log depth) iterations. GraphFrames connected components is another option.
- `createDataFrame(data, "a string, b double")` uses a DDL-string schema. Always give a schema: inference is slow and can guess wrong, and it fails on empty lists.

---

## 10. Reading a plan: `df.explain()`

```python
spark_resolve(crm, canon).explain(mode="formatted")
```

What to look for:
- `Exchange` with `Arguments: hashpartitioning(crm_id, 200)`: a **shuffle**. Count them.
- `BroadcastHashJoin` vs `SortMergeJoin`: which strategy was chosen.
- `BroadcastExchange`: the side that was broadcast.
- `ArrowEvalPython`: your `pandas_udf`, the Python boundary.
- `InMemoryTableScan`: the cached `crm_n` being reused.
- `Window`: the ranking step.

On the 5-row sample, the plan shows 10 `BroadcastHashJoin`s and only 2 `SortMergeJoin`s. Every table is under 10 MB, so Spark auto-broadcasts them. At 50M rows most of those become sort-merge joins, so plans from tiny test data don't represent production.

In an interview: "I'd check `explain()` for unexpected `SortMergeJoin`s on small tables and count the Exchanges. Then I'd look at the Spark UI's stage view for one straggler task, which means skew."

---

## 11. Settings worth naming

| Setting | Meaning |
|---|---|
| `spark.sql.shuffle.partitions` | Partitions after a shuffle (default 200). |
| `spark.sql.adaptive.enabled` (default on in Spark 3.2+) | **AQE**: re-plans at runtime using real sizes. It merges tiny partitions, switches to broadcast joins, and splits skewed partitions. |
| `spark.sql.adaptive.skewJoin.enabled` | Lets AQE split skewed join partitions. |
| `spark.sql.autoBroadcastJoinThreshold` | Size under which Spark broadcasts automatically (10 MB). |
| `spark.sql.execution.arrow.maxRecordsPerBatch` | Batch size for pandas UDFs (10k). Lower it if UDF memory is tight. |
| `spark.executor.memory`, `spark.executor.cores` | Resources per executor. |

---

## 12. Walkthrough of `spark_resolve()` in plain English

| Step | Code | Spark operation | Shuffle? |
|---|---|---|---|
| Fill optional cols | `withColumn(col, F.lit(None).cast("string"))` | narrow | no |
| Normalise | `normalize_df` → pandas UDFs, `cast`, `when` | narrow + Python | no |
| Cache CRM | `.cache()` | storage | no |
| Keys | `with_block_keys` → `array`, `filter`, `explode` | narrow | no |
| Block sizes | `groupBy("block_key").count()` | aggregation | **yes** |
| Prune hot keys | `left_anti` + `broadcast` | broadcast join | no (small side) |
| Candidate pairs | join on `block_key`, `dropDuplicates` | sort-merge join + distinct | **yes** |
| Attach attributes | 2 joins to `c` / `k` by id | joins | **yes** |
| Score | `score_udf(...)` struct | Arrow pandas UDF | no |
| Best per CRM | `Window.partitionBy("crm_id")`, `row_number`, `lead` | window | **yes** |
| Decide | `when`/`coalesce`/`format_string` | narrow | no |
| Keep unmatched | `left` join from all CRM ids | join | **yes** |
| Ultimate parent | driver `collect` → `createDataFrame` → broadcast join | driver + broadcast | no |

**Parity guarantee:** `tests/test_spark_resolve.py` runs the pandas `resolve` and `spark_resolve` on the sample, on a tie case, and on 200 random CRM rows against 300 companies with many collisions. It asserts identical `crm_id`, `company_id`, `score`, `method` and `ultimate_parent_id`. The random test uses `max_block_size=10_000` so pruning doesn't change the candidate set. Reason strings are allowed to differ in wording.

---

## 13. Likely follow-up questions

| Q | A |
|---|---|
| Transformation vs action? | Transformations build the plan lazily. Actions (`count`, `collect`, `write`) execute it. |
| What triggers a shuffle here? | `groupBy`, the joins on `block_key` and ids, `dropDuplicates`, `Window.partitionBy`. |
| How do you reduce shuffles? | Broadcast small sides, filter and select columns early, prune hot blocks, reuse the same partitioning, cache reused inputs. |
| Why is a Python UDF slow? | Each row is serialised JVM→Python→JVM, and the optimizer can't see inside it. `pandas_udf` batches through Arrow, and built-ins avoid Python entirely. |
| How would you remove the scoring UDF? | Replace name similarity with built-ins (`F.levenshtein`, or `jaro_winkler` via a Scala UDF), and express the weighted sum with Column arithmetic. It's faster, but it duplicates logic, so I'd keep the parity tests. |
| `repartition` vs `coalesce`? | `repartition(n, col)` does a full shuffle, and can partition by a key. `coalesce(n)` only merges partitions, with no shuffle. Use it before writing fewer files. |
| How do you write the output? | `result.write.mode("overwrite").partitionBy("tenant_id").parquet(path)`. Partition by something with low cardinality that queries filter on. |
| Why is Parquet better than CSV? | It's columnar, compressed, has a schema, and supports predicate and column pushdown. |
| Is the result deterministic? | Yes. The window order ends with `company_id`, scores are rounded before comparison, and `dropDuplicates` is on the full (crm_id, company_id) pair. `F.rand` (in salting) needs a `seed`. |
| How would you test Spark code? | A local session fixture, small shuffle-partition count, and parity against a trusted pandas implementation. Plus unit tests of the pure functions, which is where most of the logic lives. |
| MinHash LSH? | `pyspark.ml.feature.MinHashLSH` on name shingles, with `approxSimilarityJoin(threshold)`. It's another blocking key that tolerates typos anywhere in the name, not just after the first 4 characters. |
| Where do embeddings go? | Compute them in batches (a pandas UDF or a separate GPU job), store them, then ANN top-k per CRM row as an additional candidate source. Score with the same `score_pair` plus a cosine feature. |
