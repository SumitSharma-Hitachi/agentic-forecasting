# Copper Price Forecasting

> **Reference implementation 5 of 5.** Recommended order: [getting started](../getting_started/) -> [S&P 500](../sp500_forecasting/) -> [food CPI](../food_price_forecasting/) -> [energy / WTI](../energy_oil_forecasting/) -> [BoC rate decisions](../boc_rate_decisions/) -> **copper**. Each stands on its own.

This implementation has three complementary workflows that keep their targets,
units, and frequencies separate:

- [`00_copper_data_exploration.ipynb`](00_copper_data_exploration.ipynb) explores
	Yahoo Finance `HG=F`, the daily copper futures close in USD per pound. It
	also offers an optional visual comparison that converts FRED's monthly copper
	series from USD per metric ton to USD per pound, plus plain-language
	explanations for every printed table and chart.
- [`01_copper_forecasting.ipynb`](01_copper_forecasting.ipynb) forecasts FRED
	series **`PCOPPUSDM`**, the monthly global copper price in USD per metric ton.
	It moves from transparent baselines through conventional statistical models to
	a cutoff-aware agentic forecast on one shared continuous forecasting task.
- [`02_copper_backtest_analysis.ipynb`](02_copper_backtest_analysis.ipynb)
	compares six-month forecast paths in 2011, 2020, and 2026. Every method gets
	exactly 24 monthly inputs, and the notebook emphasizes visual analysis,
	plain-language metrics, method failures, observed-versus-pending results, and
	optional frozen adaptive-strategy variants.

The daily Yahoo series and monthly FRED series are not interchangeable. They
have different units and update schedules, so they are never combined as one
target price.

## Notebooks

[`00_copper_data_exploration.ipynb`](00_copper_data_exploration.ipynb) covers:

1. cache-first daily Yahoo Finance `HG=F` loading and data-quality checks;
2. price, return, volatility, drawdown, trend, and seasonal summaries;
3. an optional FRED `PCOPPUSDM` monthly comparison converted to USD per pound,
	disabled by default;
4. exploratory correlations with lagged oil, US-dollar, and VIX changes;
5. a historical information-cutoff demonstration; and
6. cleaned-data and summary exports under gitignored `data/artifacts/`.

It runs without a FRED key by default. Yahoo data is cached at
`data/yfinance/`; turn on `LOAD_FRED_COMPARISON` only after setting
`FRED_API_KEY` or warming the `data/fred/PCOPPUSDM.parquet` cache.

[`01_copper_forecasting.ipynb`](01_copper_forecasting.ipynb) covers:

1. cache-first FRED loading and validation;
2. level, return, seasonality, autocorrelation, and stationarity diagnostics;
3. leakage-safe expanding-window evaluation;
4. persistence, historical-mean, drift, seasonal-naive, and moving-average baselines;
5. exponential smoothing, damped Holt trend, AutoReg, and Prophet;
6. history-only and cutoff-aware news agent configurations;
7. CRPS, point-error, directional, and interval-coverage comparisons;
8. inverse-CRPS ensembling, future forecasts, plots, and optional exports.

[`02_copper_backtest_analysis.ipynb`](02_copper_backtest_analysis.ipynb) covers:

1. fixed 24-month inputs at February 2011, March 2020, and March 2026 cutoffs;
2. six-month paths from simple baselines, ETS, Holt, AutoReg, Prophet,
	AutoARIMA, and Kalman forecasting;
3. mean and median combinations of eligible numerical forecasts;
4. a primary comparison of the best single statistical or naive method with
	history-only, numerical-model-results, cutoff-aware news, and optional frozen
	adaptive-strategy agents;
5. MAE, RMSE, sMAPE, MASE, bias, direction accuracy, CRPS, forecast-range
	coverage, and interval-width definitions and scorecards; and
6. one forecast-path plot per period showing the 24-month input context,
	six target months, and every method, plus horizon error heatmaps, period
	rankings, a review of the global signals cited by the news agent, visible
	failure diagnostics, and optional result exports.

The numerical path runs from the local cache without model credentials. The
three stateless agents run in the saved main comparison and require configured
model credentials; their individual switches can disable external calls when
needed. Optional seed and trained adaptive variants load a frozen strategy skill
and the same cutoff-safe numerical panel used by the model-results agent. They
are read-only during evaluation, so the backtest cannot learn from its targets.
The seed skill ships under `adaptive_agent/skills/copper-strategy/`; a trained
skill is not fabricated or committed by this notebook.
The 2011 and 2020 agent rows remain available for comparison but are marked
clearly because a modern language model may have encountered descriptions of
those historical outcomes during training.

The notebook defaults to a clearly labeled synthetic development series when the
FRED cache is absent. It does not contact FRED or an LLM unless the corresponding
run guard is enabled.

## Setup

Run `uv sync` from the repository root. When your FRED key is available, add it
to the root `.env`:

```text
FRED_API_KEY=your_key_here
```

For the daily exploration notebook, run all cells after `uv sync`; it loads or
warms the Yahoo `HG=F` cache automatically. The FRED comparison stays off until
you set `LOAD_FRED_COMPARISON = True`.

For the monthly forecasting notebook, set `RUN_DOWNLOAD = True` once. The resulting
`data/fred/PCOPPUSDM.parquet` cache is reused without a key or network request.
Set `RUN_BASELINE_BACKTESTS = True` to score the naive and Prophet predictors.
Set `RUN_AGENT_BACKTEST = True` only when the repository's Vector proxy
credentials are also configured; this runs multiple model and search calls.

The backtest-analysis notebook requires the same real FRED cache and does not
substitute synthetic data. Its numerical methods and three agent approaches run
in the main comparison by default. Set `RUN_HISTORY_AGENT`,
`RUN_MODEL_PANEL_AGENT`, or `RUN_NEWS_AGENT` to `False` to skip individual
external calls. Set `RUN_ADAPTIVE_SEED_AGENT=True` to add the shipped seed
strategy. The notebook also enables an experimental frozen variant at
`adaptive_agent/skills/copper-strategy-trained/SKILL.md`. It adds a possible
rebound and wider uncertainty after a sudden fall, plus less aggressive trend
flattening when validated trend models agree. Those rules were proposed from
the same three displayed scenarios, so their comparison is in-sample analysis,
not protected validation. Use fresh cutoffs to test whether the apparent lift
generalizes.

## Layout

```text
copper_forecasting/
|-- 00_copper_data_exploration.ipynb  # daily Yahoo `HG=F` exploration
|-- 01_copper_forecasting.ipynb  # end-to-end workflow
|-- 02_copper_backtest_analysis.ipynb  # three-period six-month comparison
|-- adaptive_agent/skills/
|   |-- copper-strategy/SKILL.md          # frozen seed strategy
|   `-- copper-strategy-trained/SKILL.md  # frozen experimental corrections
|-- data.py                      # FRED-backed DataService registration
|-- prophet_baseline.py          # monthly Prophet Predictor
|-- agent.py                     # stateless agents and frozen adaptive strategy wiring
`-- specs/copper_backtest.yaml   # quarterly-origin 2018-2024 backtest
```

The FRED adapter currently approximates `released_at` with each observation's
timestamp. The current cache can therefore contain later corrections that were
not available on the original forecast date. Use a source that preserves each
historical release before treating retrospective results as a strict real-time
evaluation.