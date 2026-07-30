import numpy as np
import pandas as pd

from caprl.risk.conditional import build_conditional_risk_frame
from caprl.risk.mixture_cvar import TwoStateCVarGrid


def test_conditional_risk_frame_preserves_dates_and_matches_grid():
    dates = pd.date_range("2020-01-01", periods=3)
    probabilities = pd.DataFrame(
        {"bear_probability": [1.0, 0.5, 0.0], "bull_probability": [0.0, 0.5, 1.0]},
        index=dates,
    )
    means = np.array([-0.002, 0.001])
    variances = np.array([0.02**2, 0.008**2])
    grid = TwoStateCVarGrid.build(means, variances, grid_size=1001)
    result = build_conditional_risk_frame(
        probabilities, means, variances, eta=0.02, cvar_grid=grid
    )
    assert result.index.equals(dates)
    np.testing.assert_allclose(result["conditional_cvar"], grid(np.array([0.0, 0.5, 1.0])))
    assert result["exposure_cap"].between(0.0, 1.0).all()

