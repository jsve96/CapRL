import pandas as pd

from caprl.evaluation.plots import plot_cap_vs_bull_probability


def test_cap_diagnostic_plot_is_written(tmp_path):
    index = pd.date_range("2020-01-01", periods=4)
    bull_probability = pd.Series([0.8, 0.0, 1.0, 0.4], index=index)
    caps = pd.Series([0.7, 0.0, 1.0, 0.3], index=index)
    output = tmp_path / "figures" / "cap.png"
    plot_cap_vs_bull_probability(bull_probability, caps, output)
    assert output.is_file()
    assert output.stat().st_size > 0

