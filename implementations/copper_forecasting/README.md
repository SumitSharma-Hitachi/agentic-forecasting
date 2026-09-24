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
	exactly 24 monthly inputs in the primary experiment. A separate 2026 deep dive
	compares 24-, 36-, and 60-month windows and tests a signal-disciplined news
	agent using a 36-month numerical panel. It also parses the uploaded Chinese
	NBS activity panel and tests cutoff-specific contemporaneous and lead/lag
	correlations with copper returns.
- [`run_revised_news_agent.py`](run_revised_news_agent.py) runs the advanced
	`gemini-3.5-flash` revised news agent at a selectable monthly cutoff. It uses
	36 months of copper prices, supplies the six-month-ahead rows from a 36-month
	numerical panel, predicts the horizon-six month only, reports its absolute
	error (the one-row MAE), and writes an interactive trend-and-error chart.
	When at least 12 cutoff-safe pairs exist, it also supplies total Chinese
	fixed-asset growth and sign-aware relationship guidance; otherwise it records
	that NBS context was unavailable and runs the remaining revised workflow.

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

1. fixed 24-month inputs at February 2011, March 2020, and March 2026 cutoffs
	in the reproducible primary experiment;
2. six-month paths from simple baselines, ETS, Holt, AutoReg, Prophet,
	AutoARIMA, and Kalman forecasting;
3. mean and median combinations of eligible numerical forecasts;
4. a primary comparison of the best single statistical or naive method with
	history-only, numerical-model-results, and cutoff-aware news agents;
5. MAE, RMSE, sMAPE, MASE, bias, direction accuracy, CRPS, forecast-range
	coverage, and interval-width definitions and scorecards; and
6. one forecast-path plot per period showing the 24-month input context,
	six target months, and every method, plus horizon error heatmaps, period
	rankings, a review of the global signals cited by the news agent, visible
	failure diagnostics, and optional result exports; and
7. an outcome-informed 2026 diagnostic that compares 24-, 36-, and 60-month
	windows, recommends 36 months for Prophet and future news-agent context,
	reinterprets the saved March 2026 signals, and runs a combined numerical-panel
	plus cutoff-aware news agent with explicit signal scope, timing, magnitude,
	confidence, source, and counter-case requirements. A dedicated plot and score
	table compare the original and revised news paths with Prophet, ETS, and damped
	Holt fitted on 36 months; and
8. Chinese NBS industrial, investment, electricity, gas, and natural-gas data
	parsing, three-window visualization, and Pearson/Spearman lead-lag analysis.
	The saved analysis finds no strong relationship at the predeclared 0.40
	threshold, so these signals remain exploratory rather than model or prompt
	inputs.

The numerical path runs from the local cache without model credentials. All
three agents run in the saved main comparison and require configured model
credentials; their individual switches can disable external calls when needed.
The 2011 and 2020 agent rows remain available for comparison but are marked
clearly because a modern language model may have encountered descriptions of
those historical outcomes during training.

The revised 2026 agent is controlled separately by
`RUN_REVISED_2026_NEWS_AGENT`. It remains outside the primary leaderboard because
its 36-month window and signal rules were chosen after the 2026 outcomes were
observed. The saved diagnostic improved on the original news path, but it is a
hypothesis for the next prospective forecast rather than an unbiased backtest.

The standalone runner defaults to January 1, 2026, accepts another month through
`--cutoff YYYY-MM-01`, and uses `ADVANCED_MODEL` for the analyst, search, and
verifier calls. When available, its prompt includes 36 calendar months of total
NBS fixed-asset accumulated growth and the observed correlation between growth
in month $t$ and copper's cumulative return through $t+6$. The guidance follows
the cutoff-specific correlation sign. At the default cutoff this 36-month context
gives Pearson $r=-0.698$ over 28 usable pairs. The latest six reported observations
fall by 5.4 percentage points, so the default prompt applies this as a high-weight
bullish six-month prior under the negative relationship. This is prompt emphasis,
not a fitted causal coefficient; the runner explicitly forbids linear magnitude
extrapolation. The February 2011 window predates the available NBS observations,
so that historical run omits the NBS section rather than fabricating a signal.

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
external calls. Set `RUN_REVISED_2026_NEWS_AGENT` to `False` to skip the separate
outcome-informed 2026 diagnostic call. The NBS analysis reads the included
`data/chinese_nbs_dataset.json` and requires no network request.

Run the standalone agent from the repository root:

```bash
uv run python -m copper_forecasting.run_revised_news_agent
```

Use `--cutoff 2011-02-01` or `--cutoff 2020-03-01` for the other two
six-month comparison periods.

Use `--dry-run` to fit the numerical panel and write the exact agent prompt
without making model or search calls. Outputs are written under
`data/predictions/copper_revised_news_<mon>_<year>_h6/`: `forecasts.csv`,
`metrics.json`, `agent_prompt.json`, and the self-contained
`forecast_trends_and_error.html` chart. The default single forecast targets July
2026, so its absolute error is also the reported one-row MAE.

## Layout

```text
copper_forecasting/
|-- 00_copper_data_exploration.ipynb  # daily Yahoo `HG=F` exploration
|-- 01_copper_forecasting.ipynb  # end-to-end workflow
|-- 02_copper_backtest_analysis.ipynb  # three-period six-month comparison
|-- data/chinese_nbs_dataset.json  # exploratory monthly China activity panel
|-- data.py                      # FRED-backed DataService registration
|-- prophet_baseline.py          # monthly Prophet Predictor
|-- agent.py                     # history, model-results, and news agents
|-- run_revised_news_agent.py    # selectable-cutoff advanced horizon-six run + chart
`-- specs/copper_backtest.yaml   # quarterly-origin 2018-2024 backtest
```

The FRED adapter currently approximates `released_at` with each observation's
timestamp. The current cache can therefore contain later corrections that were
not available on the original forecast date. Use a source that preserves each
historical release before treating retrospective results as a strict real-time
evaluation.

The uploaded NBS file similarly records observation months but not publication
timestamps or vintages. The notebook and standalone runner use an explicit
one-month publication-lag assumption. The runner exposes only total fixed-asset
growth and labels its negative six-month correlation as descriptive. Acquire
release dates and demonstrate rolling-origin lift over a price-only baseline
before treating the NBS context as validated production signal.