"""Parity tests: the Spark pipeline must agree with the pandas reference on the same inputs.
Skipped automatically when pyspark (and Java) are not installed."""
import os
import random
import string
import sys

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

from er.resolve import resolve  # noqa: E402
from er.spark_resolve import spark_resolve  # noqa: E402

COMPARE = ["crm_id", "company_id", "score", "method", "ultimate_parent_id"]


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # workers use the same venv as the driver
    session = (
        SparkSession.builder.master("local[2]").appName("er-tests")
        .config("spark.sql.shuffle.partitions", "4")  # default 200 is slow for tiny test data
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _to_spark(spark, df: pd.DataFrame):
    return spark.createDataFrame(df.astype(object).where(df.notna(), None))


def _both(spark, crm: pd.DataFrame, canon: pd.DataFrame, **kwargs):
    expected = resolve(crm, canon)[COMPARE].sort_values("crm_id").reset_index(drop=True)
    got = (
        spark_resolve(_to_spark(spark, crm), _to_spark(spark, canon), **kwargs)
        .toPandas()[COMPARE].sort_values("crm_id").reset_index(drop=True)
    )
    return expected.astype(object).where(expected.notna(), None), got.astype(object).where(got.notna(), None)


def test_sample_matches_pandas(spark, crm, canon):
    expected, got = _both(spark, crm, canon)
    pd.testing.assert_frame_equal(got, expected)
    assert got.set_index("crm_id").loc["001E", "ultimate_parent_id"] == "C9"


def test_ties_and_no_candidates_match_pandas(spark):
    canon = pd.DataFrame({"company_id": ["B", "A"], "name": ["Acme", "Acme"], "domain": ["acme.com"] * 2})
    crm = pd.DataFrame({"crm_id": ["1", "2"], "name": ["Acme", "Zzz"], "website": ["acme.com", None]})
    expected, got = _both(spark, crm, canon)
    pd.testing.assert_frame_equal(got, expected)
    assert got["method"].tolist() == ["rejected", "rejected"]


def test_random_synthetic_matches_pandas(spark):
    rnd = random.Random(7)

    def word():
        return "".join(rnd.choices(string.ascii_lowercase[:6], k=rnd.randint(3, 6)))  # small alphabet => collisions

    names = [f"{word()} {word()}" for _ in range(300)]
    canon = pd.DataFrame({
        "company_id": [f"C{i}" for i in range(300)], "name": names,
        "domain": [n.replace(" ", "") + ".com" for n in names],
        "country": [rnd.choice(["US", "DE", None]) for _ in names],
        "employees": [rnd.choice([10, 100, 5000, None]) for _ in names],
    })
    crm = pd.DataFrame({
        "crm_id": [f"R{i}" for i in range(200)],
        "name": [rnd.choice(names).upper() + rnd.choice(["", " Inc", " GmbH"]) for _ in range(200)],
        "website": [rnd.choice([None, "gmail.com", rnd.choice(names).replace(" ", "") + ".com"]) for _ in range(200)],
        "country": [rnd.choice(["US", "Germany", None]) for _ in range(200)],
        "employees": [rnd.choice([12, 90, 4000, None]) for _ in range(200)],
    })
    expected, got = _both(spark, crm, canon, max_block_size=10_000)  # no pruning => same candidates
    pd.testing.assert_frame_equal(got, expected)
