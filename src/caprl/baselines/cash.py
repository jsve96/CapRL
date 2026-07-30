"""Cash baseline."""

import pandas as pd


def cash_exposure(prices: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=prices.index, name="exposure")

