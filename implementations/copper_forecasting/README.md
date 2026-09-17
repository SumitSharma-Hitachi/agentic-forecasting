# Copper Price Forecasting

> **Reference implementation 5 of 5.** Recommended order: [getting started](../getting_started/) -> [S&P 500](../sp500_forecasting/) -> [food CPI](../food_price_forecasting/) -> [energy / WTI](../energy_oil_forecasting/) -> [BoC rate decisions](../boc_rate_decisions/) -> **copper**. Each stands on its own.

This implementation forecasts FRED series **`PCOPPUSDM`**, the monthly global
price of copper in USD per metric ton. The single notebook moves from transparent
baselines through conventional statistical models to a cutoff-aware agentic
forecast on one shared continuous forecasting task.

## Notebook

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

Then open the notebook and set `RUN_DOWNLOAD = True` once. The resulting
`data/fred/PCOPPUSDM.parquet` cache is reused without a key or network request.
Set `RUN_BASELINE_BACKTESTS = True` to score the naive and Prophet predictors.
Set `RUN_AGENT_BACKTEST = True` only when the repository's Vector proxy
credentials are also configured; this runs multiple model and search calls.

## Layout

```text
copper_forecasting/
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