# Copper Price Forecasting

> **Reference implementation 5 of 5.** Recommended order: [getting started](../getting_started/) -> [S&P 500](../sp500_forecasting/) -> [food CPI](../food_price_forecasting/) -> [energy / WTI](../energy_oil_forecasting/) -> [BoC rate decisions](../boc_rate_decisions/) -> **copper**. Each stands on its own.

This implementation has two complementary workflows that keep their targets,
units, and frequencies separate:

- [`00_copper_data_exploration.ipynb`](00_copper_data_exploration.ipynb) explores
	Yahoo Finance `HG=F`, the daily copper futures close in USD per pound. It
	also offers an optional visual comparison with FRED's monthly copper series
	and plain-language explanations for every printed table and chart.
- [`01_copper_forecasting.ipynb`](01_copper_forecasting.ipynb) forecasts FRED
	series **`PCOPPUSDM`**, the monthly global copper price in USD per metric ton.
	It moves from transparent baselines through conventional statistical models to
	a cutoff-aware agentic forecast on one shared continuous forecasting task.

The daily Yahoo series and monthly FRED series are not interchangeable. They
have different units and update schedules, so they are never combined as one
target price.

## Notebook

[`00_copper_data_exploration.ipynb`](00_copper_data_exploration.ipynb) covers:

1. cache-first daily Yahoo Finance `HG=F` loading and data-quality checks;
2. price, return, volatility, drawdown, trend, and seasonal summaries;
3. an optional FRED `PCOPPUSDM` monthly comparison, disabled by default;
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

## Layout

```text
copper_forecasting/
|-- 00_copper_data_exploration.ipynb  # daily Yahoo `HG=F` exploration
|-- 01_copper_forecasting.ipynb  # end-to-end workflow
|-- data.py                      # FRED-backed DataService registration
|-- prophet_baseline.py          # monthly Prophet Predictor
|-- agent.py                     # prompt, basic agent, and news agent
`-- specs/copper_backtest.yaml   # quarterly-origin 2018-2024 backtest
```

The FRED adapter currently approximates `released_at` with each observation's
timestamp. For revision-sensitive historical studies, replace this with an
ALFRED vintage-aware adapter before treating retrospective agent results as a
strict real-time evaluation.