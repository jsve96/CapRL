import numpy as np
import pandas as pd
import pandas.testing as pdt

from caprl.data.features import build_market_features


def test_future_price_changes_do_not_change_past_features():
    dates = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.Series(100 * np.exp(np.arange(260) * 0.001), index=dates)
    changed = prices.copy()
    changed.iloc[230:] *= 1.5

    original_features = build_market_features(prices)
    changed_features = build_market_features(changed)
    cutoff = dates[229]
    pdt.assert_frame_equal(
        original_features.loc[:cutoff], changed_features.loc[:cutoff], check_exact=True
    )

