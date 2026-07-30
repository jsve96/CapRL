import pandas as pd

from caprl.baselines.full_cap import full_cap_exposure


def test_full_cap_exposure_equals_cap_exactly():
    caps = pd.Series([0.0, 0.25, 1.0], index=pd.date_range("2020-01-01", periods=3))
    exposures = full_cap_exposure(caps)
    pd.testing.assert_series_equal(exposures, caps.rename("exposure"))

