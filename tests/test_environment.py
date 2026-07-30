import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from caprl.envs.trading_env import TradingEnv


def _environment(*, caps=None, cost=0.001, episode_length=None):
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.Series([100.0, 110.0, 110.0, 99.0, 103.95, 103.95], index=dates)
    features = pd.DataFrame({"feature": np.linspace(-1, 1, 6)}, index=dates)
    caps = pd.Series(caps if caps is not None else np.ones(6), index=dates)
    regimes = pd.DataFrame(
        {
            "bear_probability": np.linspace(1, 0, 6),
            "bull_probability": np.linspace(0, 1, 6),
            "posterior_entropy": np.zeros(6),
        },
        index=dates,
    )
    risk = pd.DataFrame(
        {"conditional_mean": 0.001, "conditional_cvar": 0.02}, index=dates
    )
    return TradingEnv(
        prices,
        features,
        caps,
        regime_data=regimes,
        risk_data=risk,
        transaction_cost_rate=cost,
        episode_length=episode_length,
    )


def test_environment_passes_gymnasium_checker():
    check_env(_environment(episode_length=3), skip_render_check=True)


def test_action_is_rescaled_by_cap_and_logging_is_complete():
    env = _environment(caps=[0.4] * 6, cost=0.001)
    observation, _ = env.reset(seed=4)
    assert env.observation_space.contains(observation)
    _, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert not terminated and not truncated
    assert info["exposure"] == pytest.approx(0.2)
    assert info["turnover"] == pytest.approx(0.2)
    assert info["gross_return"] == pytest.approx(0.02)
    assert info["transaction_cost"] == pytest.approx(0.0002)
    assert reward == pytest.approx(np.log(1.0198))
    assert info["step_risk_charge"] == pytest.approx(0.004)
    assert info["cap_binds"]


def test_full_action_reproduces_cap_and_unit_cap_reproduces_action():
    constrained = _environment(caps=[0.3] * 6, cost=0.0)
    constrained.reset()
    _, _, _, _, constrained_info = constrained.step(np.array([1.0], dtype=np.float32))
    assert constrained_info["exposure"] == pytest.approx(0.3)
    unconstrained = _environment(cost=0.0)
    unconstrained.reset()
    _, _, _, _, unconstrained_info = unconstrained.step(np.array([0.37], dtype=np.float32))
    assert unconstrained_info["exposure"] == pytest.approx(0.37)


def test_constant_zero_action_has_no_return_or_turnover():
    env = _environment(cost=0.01, episode_length=2)
    env.reset()
    for _ in range(2):
        _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert reward == 0.0
        assert info["turnover"] == 0.0
        assert info["wealth"] == 1.0


def test_episode_stops_at_configured_boundary():
    env = _environment(episode_length=2)
    env.reset(options={"start_index": 2})
    _, _, _, first_truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    _, _, _, second_truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert not first_truncated
    assert second_truncated
    assert len(env.trajectory) == 2

