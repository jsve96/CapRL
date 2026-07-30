#!/usr/bin/env python3
"""Download adjusted daily price history from Yahoo Finance's chart endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

import pandas as pd

from caprl.data.validation import validate_market_data


def download_yahoo_history(
    symbol: str,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    start_timestamp = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_date = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    query = urllib.parse.urlencode(
        {
            "period1": start_timestamp,
            "period2": int(end_date.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CapRL/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo Finance returned an error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo Finance returned no chart result")
    result = results[0]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    data = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adjusted_close": adjusted,
            "volume": quote["volume"],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    ).dropna(subset=["adjusted_close"])
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()
    return validate_market_data(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date; defaults to today")
    parser.add_argument("--output", type=Path, default=Path("data/raw/spy.csv"))
    arguments = parser.parse_args()
    data = download_yahoo_history(arguments.symbol, start=arguments.start, end=arguments.end)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(arguments.output, index=True, date_format="%Y-%m-%d")
    metadata = {
        "symbol": arguments.symbol,
        "source": "Yahoo Finance chart API",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": arguments.start,
        "requested_end_exclusive": arguments.end,
        "first_date": str(data.index.min().date()),
        "last_date": str(data.index.max().date()),
        "observations": len(data),
    }
    metadata_path = arguments.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(data)} rows to {arguments.output}")


if __name__ == "__main__":
    main()

