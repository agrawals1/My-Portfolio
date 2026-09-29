"""PySpark version of `resolve` for the scale-out follow-up (50M CRM rows x 5M companies).

Same decisions as the pandas version, and the *same pure functions* (normalize_*, score_pair)
so the two can never drift apart. What changes is only how work is distributed:

  pandas version                      Spark version
  --------------------------------    ---------------------------------------------------
  build_blocks() -> dict              explode() one row per (record, block key)
  candidate_positions() lookup        equi-join on block key  (no cross join)
  Python loop over candidates         pandas_udf scores a whole Arrow batch of pairs
  sorted(... key=...)[0]              Window + row_number() per crm_id
  dict walk for ultimate parent       small parent table built on the driver, broadcast-joined
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from er.config import MatchConfig
from er.normalize import normalize_country, normalize_domain, normalize_name
from er.resolve import ultimate_parent
from er.scoring import Record, score_pair

SCORE_SCHEMA = T.StructType([
    T.StructField("score", T.DoubleType()),
    T.StructField("method", T.StringType()),
    T.StructField("reason", T.StringType()),
    T.StructField("name_sim", T.DoubleType()),
])


# --- 1. normalise: reuse the tested pure functions, vectorised via Arrow -------------------
@F.pandas_udf(T.StringType())
def _domain_udf(s: pd.Series) -> pd.Series:
    return s.map(normalize_domain)


@F.pandas_udf(T.StringType())
def _name_udf(s: pd.Series) -> pd.Series:
    return s.map(normalize_name)


@F.pandas_udf(T.StringType())
def _country_udf(s: pd.Series) -> pd.Series:
    return s.map(normalize_country)


def normalize_df(df: DataFrame, id_col: str, domain_col: str) -> DataFrame:
    employees = F.col("employees").cast("double")  # bad strings -> null, like pd.to_numeric
    return df.select(
        F.col(id_col).cast("string").alias("id"),
        _name_udf(F.col("name")).alias("name_norm"),
        _domain_udf(F.col(domain_col)).alias("domain_norm"),
        _country_udf(F.col("country")).alias("country_norm"),
        F.when(employees >= 0, employees).alias("employees_num"),  # negatives -> null
    )


# --- 2. block: one row per (record, key), then an equi-join --------------------------------
def with_block_keys(df: DataFrame, prefix_len: int) -> DataFrame:
    """Same keys as blocking.block_keys: 'd:<domain>' and 'p:<first token prefix>'."""
    first_token = F.split(F.col("name_norm"), " ").getItem(0)
    domain_key = F.when(F.col("domain_norm").isNotNull(), F.concat(F.lit("d:"), F.col("domain_norm")))
    prefix_key = F.when(F.col("name_norm") != "", F.concat(F.lit("p:"), F.substring(first_token, 1, prefix_len)))
    keys = F.filter(F.array(domain_key, prefix_key), lambda k: k.isNotNull())
    return df.withColumn("block_key", F.explode(keys))  # explode drops rows with no keys


def candidate_pairs(crm_k: DataFrame, canon_k: DataFrame, max_block_size: int) -> DataFrame:
    """(crm_id, company_id) pairs sharing at least one block key.

    Hot name-prefix blocks ("p:inte" for thousands of "International ..." companies) are dropped:
    they cost O(block^2) and rarely decide a match on their own. Domain blocks are never dropped.
    """
    sizes = canon_k.groupBy("block_key").count()
    hot = sizes.filter((F.col("count") > max_block_size) & F.col("block_key").startswith("p:"))
    canon_k = canon_k.join(F.broadcast(hot.select("block_key")), "block_key", "left_anti")
    return (
        crm_k.select("block_key", F.col("id").alias("crm_id"))
        .join(canon_k.select("block_key", F.col("id").alias("company_id")), "block_key")
        .select("crm_id", "company_id")
        .dropDuplicates()  # a pair can share both a domain and a prefix key
    )


# --- 3. score: batch of pairs -> pandas -> the same score_pair() -----------------------------
def _make_score_udf(cfg: MatchConfig):
    @F.pandas_udf(SCORE_SCHEMA)
    def score_udf(
        c_name: pd.Series, c_dom: pd.Series, c_cty: pd.Series, c_emp: pd.Series,
        k_name: pd.Series, k_dom: pd.Series, k_cty: pd.Series, k_emp: pd.Series,
    ) -> pd.DataFrame:
        def rec(name, dom, cty, emp) -> Record:
            return Record("", name or "", dom if isinstance(dom, str) else None,
                          cty if isinstance(cty, str) else None, None if pd.isna(emp) else float(emp))

        rows = [
            score_pair(rec(*crm), rec(*canon), cfg)
            for crm, canon in zip(zip(c_name, c_dom, c_cty, c_emp), zip(k_name, k_dom, k_cty, k_emp))
        ]
        return pd.DataFrame({
            "score": [r.score for r in rows], "method": [r.method for r in rows],
            "reason": [r.reason for r in rows], "name_sim": [r.name_sim for r in rows],
        })

    return score_udf


# --- 4. decide: best candidate per crm_id, reject below threshold or on ties ----------------
def _pick_best(scored: DataFrame, cfg: MatchConfig) -> DataFrame:
    w = Window.partitionBy("crm_id").orderBy(F.desc("score_r"), F.desc("sim_r"), F.asc("company_id"))
    ranked = (
        scored.withColumn("score_r", F.round("s.score", 6))  # round so float noise can't break ties
        .withColumn("sim_r", F.round("s.name_sim", 6))
        .withColumn("rank", F.row_number().over(w))
        .withColumn("next_score", F.lead("score_r").over(w))
        .withColumn("next_sim", F.lead("sim_r").over(w))
        .filter(F.col("rank") == 1)
    )
    # lead() is NULL when there is only one candidate, and NULL == x is NULL (not False).
    # Without coalesce, ~rejected would be NULL and silently null out company_id.
    same = (F.col("next_score") == F.col("score_r")) & (F.col("next_sim") == F.col("sim_r"))
    tied = F.coalesce(same, F.lit(False))
    below = F.col("score_r") < cfg.threshold
    rejected = below | (tied & F.lit(cfg.on_tie == "reject"))
    reason = (
        F.when(below, F.format_string(f"best candidate %s scored %.2f < {cfg.threshold}", "company_id", "score_r"))
        .when(rejected, F.lit("tie between top candidates; refusing to guess"))
        .otherwise(F.col("s.reason"))
    )
    return ranked.select(
        "crm_id",
        F.when(~rejected, F.col("company_id")).alias("company_id"),
        F.when(rejected, F.lit(0.0)).otherwise(F.round("score_r", 4)).alias("score"),
        F.when(rejected, F.lit("rejected")).otherwise(F.col("s.method")).alias("method"),
        reason.alias("reason"),
    )


def _parent_table(spark, canon: DataFrame) -> DataFrame:
    """company_id -> ultimate_parent_id. Collected to the driver: fine for a few million
    rows; beyond that, iterate a self-join until no row changes (pointer jumping)."""
    if "parent_id" not in canon.columns:
        return spark.createDataFrame([], "company_id string, ultimate_parent_id string")
    parent_of = {r.company_id: r.parent_id for r in canon.select("company_id", "parent_id").collect()}
    rows = [(cid, ultimate_parent(cid, parent_of)) for cid in parent_of]
    return spark.createDataFrame(rows, "company_id string, ultimate_parent_id string")


def spark_resolve(
    crm: DataFrame, canon: DataFrame, cfg: MatchConfig | None = None, max_block_size: int = 1000
) -> DataFrame:
    """One row per CRM row: crm_id, company_id, score, method, reason, ultimate_parent_id."""
    cfg = cfg or MatchConfig()
    spark = crm.sparkSession
    for col in ("country", "employees"):  # optional columns, as in the pandas version
        crm = crm if col in crm.columns else crm.withColumn(col, F.lit(None).cast("string"))
        canon = canon if col in canon.columns else canon.withColumn(col, F.lit(None).cast("string"))

    crm_n = normalize_df(crm, "crm_id", "website").cache()  # reused 3x below: compute once
    canon_n = normalize_df(canon, "company_id", "domain")

    pairs = candidate_pairs(
        with_block_keys(crm_n, cfg.prefix_len), with_block_keys(canon_n, cfg.prefix_len), max_block_size
    )
    c, k = crm_n.alias("c"), canon_n.alias("k")
    enriched = pairs.join(c, pairs.crm_id == F.col("c.id")).join(k, pairs.company_id == F.col("k.id"))
    score = _make_score_udf(cfg)
    scored = enriched.select(
        "crm_id", "company_id",
        score(*[F.col(f"c.{x}") for x in ("name_norm", "domain_norm", "country_norm", "employees_num")],
              *[F.col(f"k.{x}") for x in ("name_norm", "domain_norm", "country_norm", "employees_num")]).alias("s"),
    )

    best = _pick_best(scored, cfg)
    no_candidates = F.lit("no candidate companies")
    result = (
        crm_n.select(F.col("id").alias("crm_id"))
        .join(best, "crm_id", "left")  # keep CRM rows that had zero candidates
        .join(F.broadcast(_parent_table(spark, canon)), "company_id", "left")
        .select(
            "crm_id", "company_id",
            F.coalesce("score", F.lit(0.0)).alias("score"),
            F.coalesce("method", F.lit("rejected")).alias("method"),
            F.coalesce("reason", no_candidates).alias("reason"),
            F.coalesce("ultimate_parent_id", "company_id").alias("ultimate_parent_id"),  # no parent => itself
        )
    )
    return result
