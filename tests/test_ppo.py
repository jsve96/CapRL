import numpy as np
import pandas as pd

from caprl.agents.checkpointing import load_agent, save_agent
from caprl.agents.evaluation import evaluate_agent
from caprl.agents.ppo import PPOConfig, train_ppo
from caprl.envs.trading_env import TradingEnv


def _factory():
    index = pd.bdate_range("2020-01-01", periods=40)
    returns = 0.001 + 0.005 * np.sin(np.arange(40))
    prices = pd.Series(100 * np.cumprod(1 + returns), index=index)
    features = pd.DataFrame(
        {"return_feature": returns, "trend": np.linspace(-1, 1, 40)}, index=index
    )
    caps = pd.Series(np.linspace(0.2, 1.0, 40), index=index)
    return TradingEnv(
        prices,
        features,
        caps,
        episode_length=16,
        random_start=True,
        transaction_cost_rate=0.001,
    )


def _small_config():
    return PPOConfig(
        hidden_layers=(8,),
        rollout_steps=32,
        training_steps=64,
        batch_size=16,
        epochs_per_update=2,
    )


def test_ppo_training_is_reproducible_and_actions_are_bounded():
    first, first_metrics = train_ppo(_factory, config=_small_config(), seed=12)
    second, second_metrics = train_ppo(_factory, config=_small_config(), seed=12)
    observation, _ = _factory().reset(seed=3)
    first_action = first.predict(observation)
    second_action = second.predict(observation)
    np.testing.assert_allclose(first_action, second_action, atol=0.0)
    assert 0.0 <= first_action[0] <= 1.0
    assert first_metrics == second_metrics


def test_checkpoint_round_trip_reproduces_action(tmp_path):
    agent, _ = train_ppo(_factory, config=_small_config(), seed=21)
    path = tmp_path / "agent.pt"
    save_agent(agent, path)
    restored = load_agent(path)
    observation, _ = _factory().reset(seed=3)
    np.testing.assert_allclose(agent.predict(observation), restored.predict(observation), atol=0.0)


def test_deterministic_evaluation_covers_one_complete_interval():
    agent, _ = train_ppo(_factory, config=_small_config(), seed=21)
    first = evaluate_agent(agent, _factory, seed=4)
    second = evaluate_agent(agent, _factory, seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    assert (first["exposure"] <= first["cap"] + 1e-12).all()
