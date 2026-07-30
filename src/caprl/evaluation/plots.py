"""Research diagnostic plots."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def plot_cap_vs_bull_probability(
    bull_probability: pd.Series,
    exposure_cap: pd.Series,
    output_path: str | Path,
) -> None:
    """Save the required cap-versus-bull-probability diagnostic."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional plotting extra
        raise ImportError("install caprl with the 'plots' extra to create figures") from exc
    if not bull_probability.index.equals(exposure_cap.index):
        raise ValueError("bull probabilities and caps must have identical indices")
    ordered = pd.DataFrame(
        {"bull_probability": bull_probability, "exposure_cap": exposure_cap}
    ).sort_values("bull_probability")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(ordered["bull_probability"], ordered["exposure_cap"])
    axis.set(xlabel="Filtered bull probability", ylabel="Exposure cap", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

