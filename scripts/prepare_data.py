#!/usr/bin/env python3
"""Validate canonical prices and construct causal, unscaled research features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caprl.data.features import build_market_features
from caprl.data.loader import load_price_csv
from caprl.utils.hashing import hash_file, hash_frame


def prepare_price_data(input_path: Path) -> pd.DataFrame:
    market = load_price_csv(input_path)
    prices = market["adjusted_close"]
    features = build_market_features(prices)
    prepared = market.reindex(features.index).join(features)
    prepared["simple_return_1d"] = prices.pct_change().reindex(features.index)
    if not np.isfinite(prepared.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("prepared data contains non-finite numeric values")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    prepared = prepare_price_data(arguments.input)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.suffix == ".parquet":
        prepared.to_parquet(arguments.output, index=True)
    else:
        prepared.to_csv(arguments.output, index=True, date_format="%Y-%m-%d")
    metadata = {
        "input": str(arguments.input),
        "input_sha256": hash_file(arguments.input),
        "prepared_hash": hash_frame(prepared),
        "first_date": str(prepared.index.min().date()),
        "last_date": str(prepared.index.max().date()),
        "observations": len(prepared),
        "scaled": False,
        "scaling_note": "Fit robust scalers separately on each walk-forward training fold.",
    }
    arguments.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(prepared)} rows to {arguments.output}")


if __name__ == "__main__":
    main()

