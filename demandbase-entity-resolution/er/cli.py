"""python -m er.cli --crm sample_data/crm_accounts.csv --canonical sample_data/canonical_companies.csv"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from er.clusters import find_duplicates
from er.config import MatchConfig
from er.resolve import resolve


def _read(path: str) -> pd.DataFrame:
    if Path(path).suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str)  # keep ids like "001A" / "007" intact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crm", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--out", help="write results to this CSV")
    parser.add_argument("--threshold", type=float, default=MatchConfig().threshold)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = resolve(_read(args.crm), _read(args.canonical), MatchConfig(threshold=args.threshold))
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(result.to_string(index=False))
    print("\nduplicates:")
    print(find_duplicates(result).to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
