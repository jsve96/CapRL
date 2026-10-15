# CapRL

![alte text](/outputs/RiskManagement.png)

Reproducible research code for Policy-Independent Risk Control for Reinforcement Learning in Trading: Calibrated Exposure Caps under Tail-Risk Budgets under a single-asset, long-only trend-following strategy.


## Setup

```bash
uv sync --all-extras
uv run pytest
```

Alternatively, with the standard library virtual-environment tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,research,rl,plots]'
pytest
```

## Run an experiment

You can download SPY data from Yahoo with

```bash 
uv run python scripts/download_yahoo.py --symbol SPY --output data/raw/spy.csv
uv run python scripts/prepare_data.py \
  data/raw/spy.csv data/processed/spy_features.parquet
```

Alternative:
The repository does not bundle market data. Replace `data/prices.csv` with the
path to a CSV containing at least `date` and `adjusted_close` columns, for example:

```csv
date,adjusted_close
2000-01-03,100.25
2000-01-04,101.10
```


### Sequential execution

Run every walk-forward fold sequentially with:

```bash
uv run python scripts/run_experiment.py data/prices.csv outputs/primary \
  --config configs/experiments/primary.yaml
```

Add `--no-rl` for a faster deterministic-baseline run. Use `-u` when running
Python on a server if progress output would otherwise be buffered:

```bash
uv run python -u scripts/run_experiment.py data/raw/spy.csv outputs/primary
```

### Parallel fold execution

First initialize the experiment directory. This resolves and freezes the
configuration in `outputs/primary/config.yaml` without running a fold:

```bash
uv run python scripts/run_experiment.py \
  data/raw/spy.csv outputs/primary \
  --config configs/experiments/primary.yaml \
  --initialize-only
```

Configuration overrides must be supplied during initialization. For example:

```bash
uv run python scripts/run_experiment.py \
  data/raw/spy.csv outputs/preliminary_50k \
  --config configs/experiments/primary.yaml \
  --training-steps 50000 \
  --initialize-only
```

Once `config.yaml` exists, subsequent workers load it from the experiment
directory. They do not reload the source configuration, and
`--training-steps` can no longer override it.

Run an individual zero-based fold with:

```bash
uv run python -u scripts/run_experiment.py \
  data/raw/spy.csv outputs/primary \
  --fold 0
```

`--fold` can be repeated when assigning several folds to one worker:

```bash
uv run python -u scripts/run_experiment.py \
  data/raw/spy.csv outputs/primary \
  --fold 0 --fold 1
```

For example, after activating `.venv`, run all 19 folds using four local
workers:

```bash
seq 0 18 | xargs -P 4 -I{} \
  python -u scripts/run_experiment.py \
  data/raw/spy.csv outputs/primary \
  --fold {}
```

Assign each fold to only one worker. Fold artifacts are isolated in
`fold_000`, `fold_001`, and so on. Experiment-level aggregation is protected
by a file lock and is rebuilt from all completed fold directories. After all
workers finish, aggregation can be explicitly finalized with:

```bash
uv run python scripts/run_experiment.py \
  data/raw/spy.csv outputs/primary \
  --aggregate-only
```

The data path remains a required positional argument for `--aggregate-only`,
but the market data are not reprocessed in that mode.


### Hyperparameter tuning

The compact PPO tuning script evaluates 12 learning-rate, update-epoch, and
discount-factor configurations using validation periods only:

```bash
uv run python -u scripts/tune_ppo.py data/raw/spy.csv
```

It defaults to three seeds, five screening folds, the `full` PPO strategy, and
300,000 training steps. Results are written to `outputs/hyperparameter`.
Interrupted tuning runs can be continued with `--resume`.

### Sensitivity analysis

Run CVaR-level, and risk-fraction sensitivity design
with:

```bash
uv run python -u scripts/run_cvar_ablation.py data/raw/spy.csv
```

The script runs the 16 CVaR x risk-fraction combinations. It defaults to all configured seeds
and folds for the `full` PPO strategy. Results are written to
`outputs/cvar_risk_ablation`; use `--resume` to continue an interrupted run.




