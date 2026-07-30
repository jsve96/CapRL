import pandas as pd
import pytest

from caprl.data.loader import load_price_csv


def test_loader_preserves_observed_dates_without_forward_filling(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,adj_close\n2020-01-02,100\n2020-01-06,102\n", encoding="utf-8")
    data = load_price_csv(path, adjusted_close_column="adj_close")
    assert data.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert data["adjusted_close"].tolist() == [100, 102]


def test_loader_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,adjusted_close\n2020-01-02,100\n2020-01-02,101\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_price_csv(path)


def test_loader_rejects_missing_prices(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,adjusted_close\n2020-01-02,100\n2020-01-03,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_price_csv(path)

