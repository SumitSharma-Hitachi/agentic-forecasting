"""Run the revised six-month copper news-agent forecast for a monthly cutoff."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation import ContinuousForecast, ForecastingTask
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import DartsAutoARIMAPredictor, DartsKalmanForecasterPredictor
from aieng.forecasting.models import ADVANCED_MODEL
from copper_forecasting.agent import (
    CopperModelPanelPromptBuilder,
    CopperNBSModelPanelPromptBuilder,
    build_copper_model_panel_predictor,
    build_copper_nbs_model_panel_predictor,
    build_copper_news_config,
)
from copper_forecasting.data import COPPER_FRED_ID, COPPER_SERIES_ID
from copper_forecasting.prophet_baseline import CopperProphetPredictor
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUTOFF = pd.Timestamp("2026-01-01")
INPUT_MONTHS = 36
HORIZONS = (6,)
VALIDATION_MONTHS = 6
NUMERICAL_SAMPLES = 200
NBS_SIGNAL_WEIGHT = "high"
NBS_INDICATOR = "Investment in Fixed Assets, Accumulated Growth Rate(%)"
COPPER_PATH = ROOT / "data" / "fred" / f"{COPPER_FRED_ID}.parquet"
NBS_PATH = ROOT / "implementations" / "copper_forecasting" / "data" / "chinese_nbs_dataset.json"

POINT_METHODS = {
    "Last value": "last_value",
    "Historical mean": "historical_mean",
    "Drift": "drift",
    "Seasonal naive": "seasonal_naive",
    "12-month moving average": "moving_average",
    "ETS trend": "ets",
    "Damped Holt trend": "holt_damped",
    "AutoReg": "autoreg",
}


class InsufficientNBSContextError(ValueError):
    """Raised when a cutoff has too little NBS history for the relationship."""


def load_copper_prices(cutoff: pd.Timestamp, path: Path = COPPER_PATH) -> pd.DataFrame:
    """Load and validate the cached monthly copper series."""
    if not path.exists():
        raise FileNotFoundError(f"Missing copper cache: {path}. Run scripts/fetch_fred.py first.")

    frame = pd.read_parquet(path)
    required_columns = {"timestamp", "value", "released_at"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Copper cache is missing required columns: {sorted(missing_columns)}")

    frame = frame.loc[:, ["timestamp", "value", "released_at"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["released_at"] = pd.to_datetime(frame["released_at"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"]).drop_duplicates("timestamp", keep="last")
    frame = frame.sort_values("timestamp").set_index("timestamp")

    input_start = cutoff - pd.DateOffset(months=INPUT_MONTHS - 1)
    history = frame.loc[input_start:cutoff]
    if len(history) != INPUT_MONTHS or history.index.max() != cutoff:
        raise ValueError(
            f"Expected {INPUT_MONTHS} monthly copper observations from {input_start.date()} "
            f"through {cutoff.date()}, found {len(history)}."
        )
    return frame


def build_cutoff_service(copper: pd.DataFrame, cutoff: pd.Timestamp) -> DataService:
    """Register the 36-month input window and later actuals in a cutoff-safe service."""
    input_start = cutoff - pd.DateOffset(months=INPUT_MONTHS - 1)
    service = DataService()
    service.register(
        COPPER_SERIES_ID,
        StaticFrameAdapter(copper.loc[input_start:].reset_index()),
        SeriesMetadata(
            series_id=COPPER_SERIES_ID,
            description="Global monthly copper price",
            source="FRED PCOPPUSDM cache",
            units="USD per metric ton",
            frequency="MS",
        ),
    )
    visible = service.context(as_of=cutoff).get_series(COPPER_SERIES_ID)
    if len(visible) != INPUT_MONTHS:
        raise ValueError(f"Expected {INPUT_MONTHS} cutoff-safe observations, found {len(visible)}.")
    return service


def load_fixed_asset_growth(path: Path = NBS_PATH) -> pd.Series:
    """Load the total Chinese fixed-asset accumulated growth series."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    records = [
        record
        for dataset in payload["datasets"]
        if dataset["dataset"] == "Monthly Investment in Fixed Assets"
        for record in dataset["records"]
        if record["indicator"] == NBS_INDICATOR
    ]
    if not records:
        raise ValueError(f"NBS indicator not found: {NBS_INDICATOR}")

    frame = pd.DataFrame(records)
    frame["period"] = pd.to_datetime(frame["period"], format="%Y-%m")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["period"].duplicated().any():
        raise ValueError(f"NBS indicator contains duplicate months: {NBS_INDICATOR}")
    return frame.set_index("period")["value"].sort_index().rename("fixed_asset_growth_yoy_percent")


def prepare_nbs_context(
    fixed_asset_growth: pd.Series,
    copper: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.Series, list[dict[str, Any]], float, int]:
    """Create the 36-month NBS ledger and cutoff-safe six-month correlation."""
    input_start = cutoff - pd.DateOffset(months=INPUT_MONTHS - 1)
    context_months = pd.date_range(input_start, cutoff, freq="MS")
    latest_assumed_available = cutoff - pd.DateOffset(months=1)
    available_growth = fixed_asset_growth.loc[:latest_assumed_available]
    window = available_growth.reindex(context_months)

    copper_values = copper["value"].astype(float)
    copper_six_month_forward_return = (copper_values.shift(-6) / copper_values - 1) * 100
    latest_pair_month = cutoff - pd.DateOffset(months=6)
    paired = (
        pd.concat(
            [
                window.rename("fixed_asset_growth"),
                copper_six_month_forward_return.rename("copper_six_month_forward_return"),
            ],
            axis=1,
            join="inner",
            sort=False,
        )
        .loc[:latest_pair_month]
        .dropna()
    )
    if len(paired) < 12:
        raise InsufficientNBSContextError(f"At least 12 paired NBS observations are required; found {len(paired)}.")

    correlation = float(paired["fixed_asset_growth"].corr(paired["copper_six_month_forward_return"]))
    if not np.isfinite(correlation):
        raise ValueError("The observed six-month correlation is not finite.")

    history = [
        {
            "month": str(month.date()),
            "growth_yoy_percent": None if pd.isna(value) else round(float(value), 3),
        }
        for month, value in window.items()
    ]
    if len(history) != INPUT_MONTHS:
        raise AssertionError(f"Expected {INPUT_MONTHS} NBS calendar rows, found {len(history)}.")
    return window, history, correlation, len(paired)


def point_method_forecast(train: pd.Series, horizon: int, method: str) -> np.ndarray:
    """Fit one point-only panel method and return its forecast path."""
    if method == "last_value":
        forecast = np.repeat(train.iloc[-1], horizon)
    elif method == "historical_mean":
        forecast = np.repeat(train.mean(), horizon)
    elif method == "drift":
        slope = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
        forecast = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    elif method == "seasonal_naive":
        forecast = np.resize(train.iloc[-12:].to_numpy(), horizon)
    elif method == "moving_average":
        forecast = np.repeat(train.iloc[-12:].mean(), horizon)
    elif method == "ets":
        forecast = (
            ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            .fit()
            .forecast(horizon)
        )
    elif method == "holt_damped":
        forecast = Holt(train, damped_trend=True, initialization_method="estimated").fit().forecast(horizon)
    elif method == "autoreg":
        forecast = AutoReg(train, lags=min(6, len(train) // 4), old_names=False, trend="ct").fit().forecast(horizon)
    else:
        raise ValueError(f"Unknown point method: {method}")
    return np.asarray(forecast, dtype=float)


def build_numerical_panel(
    train: pd.Series,
    task: ForecastingTask,
    service: DataService,
    cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Build the same 36-month numerical panel used by the revised notebook agent."""
    panel_rows: list[dict[str, Any]] = []
    validation_paths: dict[str, np.ndarray] = {}

    for model_name, method in POINT_METHODS.items():
        full_forecast = point_method_forecast(train, max(HORIZONS), method)
        validation_path = point_method_forecast(
            train.iloc[:-VALIDATION_MONTHS],
            VALIDATION_MONTHS,
            method,
        )
        validation_paths[model_name] = validation_path
        validation_actuals = train.iloc[-VALIDATION_MONTHS:].to_numpy()
        validation_mae = float(np.mean(np.abs(validation_path - validation_actuals)))
        panel_rows.extend(
            {
                "model": model_name,
                "horizon_months": horizon,
                "forecast": round(float(point), 2),
                "past_validation_mae": round(validation_mae, 2),
                "impossible_price": bool(point <= 0),
            }
            for horizon in HORIZONS
            for point in [full_forecast[horizon - 1]]
        )

    probabilistic_factories: dict[str, Callable[[], Predictor]] = {
        "Prophet": lambda: CopperProphetPredictor(interval_width=0.80, min_history=24),
        "AutoARIMA": lambda: DartsAutoARIMAPredictor(num_samples=NUMERICAL_SAMPLES),
        "Kalman": lambda: DartsKalmanForecasterPredictor(num_samples=NUMERICAL_SAMPLES),
    }
    context = service.context(as_of=cutoff)
    for model_name, factory in probabilistic_factories.items():
        np.random.seed(42)
        predictions = factory().predict(task, context)
        if len(predictions) != len(HORIZONS):
            raise ValueError(f"{model_name} returned {len(predictions)} forecasts; expected {len(HORIZONS)}.")
        for horizon, prediction in zip(HORIZONS, predictions, strict=True):
            payload = cast(ContinuousForecast, prediction.payload)
            panel_rows.append(
                {
                    "model": model_name,
                    "horizon_months": horizon,
                    "forecast": round(float(payload.point_forecast), 2),
                    "range_10_to_90": [
                        round(float(payload.quantiles[0.10]), 2),
                        round(float(payload.quantiles[0.90]), 2),
                    ],
                    "past_validation_mae": None,
                    "impossible_price": payload.point_forecast <= 0,
                }
            )

    validation_actuals = train.iloc[-VALIDATION_MONTHS:].to_numpy()
    validation_matrix = np.array(list(validation_paths.values()), dtype=float)
    panel_frame = pd.DataFrame(panel_rows)
    for ensemble_name, reducer in (("Mean ensemble", np.mean), ("Median ensemble", np.median)):
        validation_path = reducer(validation_matrix, axis=0)
        validation_mae = float(np.mean(np.abs(validation_path - validation_actuals)))
        for horizon in HORIZONS:
            eligible = panel_frame.loc[
                (panel_frame["horizon_months"] == horizon)
                & panel_frame["forecast"].gt(0)
                & panel_frame["forecast"].notna(),
                "forecast",
            ]
            panel_rows.append(
                {
                    "model": ensemble_name,
                    "horizon_months": horizon,
                    "forecast": round(float(reducer(eligible.to_numpy(dtype=float))), 2),
                    "past_validation_mae": round(validation_mae, 2),
                    "impossible_price": False,
                }
            )
    return panel_rows


def predictions_to_frame(predictions: list[Any], copper: pd.DataFrame) -> pd.DataFrame:
    """Combine agent predictions with available target-month actuals and errors."""
    rows = []
    for prediction in predictions:
        payload = cast(ContinuousForecast, prediction.payload)
        forecast_date = pd.Timestamp(prediction.forecast_date)
        actual = float(copper.loc[forecast_date, "value"]) if forecast_date in copper.index else np.nan
        point_forecast = float(payload.point_forecast)
        rows.append(
            {
                "forecast_date": forecast_date,
                "point_forecast": point_forecast,
                "q10": float(payload.quantiles[0.10]),
                "q90": float(payload.quantiles[0.90]),
                "actual": actual,
                "error": point_forecast - actual if np.isfinite(actual) else np.nan,
                "absolute_error": abs(point_forecast - actual) if np.isfinite(actual) else np.nan,
                "status": "Scored" if np.isfinite(actual) else "Pending",
                "rationale": " ".join(
                    part
                    for part in (
                        str(prediction.metadata.get("rationale", "")),
                        str(prediction.metadata.get("horizon_rationale", "")),
                    )
                    if part
                ).strip(),
            }
        )
    frame = pd.DataFrame(rows).sort_values("forecast_date").reset_index(drop=True)
    if len(frame) != len(HORIZONS):
        raise ValueError(f"Agent returned {len(frame)} forecasts; expected {len(HORIZONS)}.")
    return frame


def build_chart(
    copper_history: pd.Series,
    nbs_window: pd.Series,
    results: pd.DataFrame,
    cutoff: pd.Timestamp,
    nbs_context_used: bool,
) -> go.Figure:
    """Plot copper and NBS trends plus forecast errors."""
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            "Copper history, six-month-ahead forecast, and observed price",
            (
                "Chinese fixed-asset accumulated growth"
                if nbs_context_used
                else "Chinese fixed-asset accumulated growth (unavailable at cutoff)"
            ),
            "Absolute forecast error by observed target month",
        ),
        row_heights=[0.52, 0.25, 0.23],
    )
    figure.add_trace(
        go.Scatter(
            x=copper_history.index,
            y=copper_history,
            name="Copper history",
            mode="lines+markers",
            line={"color": "#4b5563", "width": 3},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=results["forecast_date"],
            y=results["q90"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=results["forecast_date"],
            y=results["q10"],
            name="Agent 80% range",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(166, 93, 53, 0.16)",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[cutoff, *results["forecast_date"].tolist()],
            y=[float(copper_history.iloc[-1]), *results["point_forecast"].tolist()],
            name="Revised news agent",
            mode="lines+markers",
            line={"color": "#a65d35", "width": 4},
        ),
        row=1,
        col=1,
    )
    observed = results.loc[results["actual"].notna()]
    figure.add_trace(
        go.Scatter(
            x=[cutoff, *observed["forecast_date"].tolist()],
            y=[float(copper_history.iloc[-1]), *observed["actual"].tolist()],
            name="Observed copper",
            mode="lines+markers",
            line={"color": "#111111", "width": 4},
        ),
        row=1,
        col=1,
    )
    figure.add_vline(
        x=cutoff.timestamp() * 1000,
        line_color="#111111",
        line_dash="dot",
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=nbs_window.index,
            y=nbs_window,
            name="NBS fixed-asset growth",
            mode="lines+markers",
            connectgaps=False,
            line={"color": "#16827c", "width": 3},
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_color="#777777", line_width=1, row=2, col=1)
    figure.add_trace(
        go.Bar(
            x=observed["forecast_date"],
            y=observed["absolute_error"],
            name="Absolute error",
            marker_color="#c2413b",
        ),
        row=3,
        col=1,
    )

    figure.update_yaxes(title_text="USD per metric ton", row=1, col=1)
    figure.update_yaxes(title_text="YoY growth (%)", row=2, col=1)
    figure.update_yaxes(title_text="Absolute error", row=3, col=1)
    figure.update_xaxes(title_text="Month", row=3, col=1)
    figure.update_layout(
        title=(
            f"Revised {cutoff.strftime('%B %Y')} copper forecast "
            + (
                "with high-weight Chinese fixed-asset context"
                if nbs_context_used
                else "without available Chinese fixed-asset context"
            )
        ),
        template="plotly_white",
        height=980,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.06, "x": 0},
        margin={"l": 80, "r": 40, "t": 130, "b": 70},
    )
    return figure


def parse_cutoff(value: str) -> pd.Timestamp:
    """Parse a monthly cutoff represented by the first day of a month."""
    try:
        cutoff = pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid cutoff date: {value}") from exc
    if cutoff != cutoff.to_period("M").to_timestamp():
        raise argparse.ArgumentTypeError("Cutoff must be the first day of a month (YYYY-MM-01).")
    return cutoff


def default_output_dir(cutoff: pd.Timestamp) -> Path:
    """Return the standard artifact directory for a cutoff."""
    cutoff_slug = cutoff.strftime("%b_%Y").lower()
    return ROOT / "data" / "predictions" / f"copper_revised_news_{cutoff_slug}_h6"


def prepare_optional_nbs_context(
    copper: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.Series, list[dict[str, Any]], float | None, int]:
    """Return cutoff-safe NBS context or an explicit unavailable result."""
    fixed_asset_growth = load_fixed_asset_growth()
    try:
        return prepare_nbs_context(fixed_asset_growth, copper, cutoff)
    except InsufficientNBSContextError as exc:
        print(f"NBS context unavailable: {exc}")
        return pd.Series(dtype=float, name="fixed_asset_growth_yoy_percent"), [], None, 0


def build_task(cutoff: pd.Timestamp) -> ForecastingTask:
    """Build the horizon-six copper task for a monthly cutoff."""
    target_date = cutoff + pd.DateOffset(months=6)
    cutoff_slug = cutoff.strftime("%b_%Y").lower()
    return ForecastingTask(
        task_id=f"copper_{cutoff_slug}_six_month_horizon",
        target_series_id=COPPER_SERIES_ID,
        horizons=list(HORIZONS),
        frequency="MS",
        description=(
            f"Global monthly copper price for {target_date.strftime('%B %Y')}, "
            f"six months after the {cutoff.strftime('%B %Y')} cutoff."
        ),
    )


def build_agent_prompt(
    task: ForecastingTask,
    context: Any,
    model_panels: dict[str, list[dict[str, Any]]],
    nbs_history: list[dict[str, Any]],
    correlation: float | None,
    correlation_pairs: int,
) -> str:
    """Build the model-panel prompt, including NBS context when available."""
    if correlation is None:
        prompt_builder = CopperModelPanelPromptBuilder(model_panels=model_panels)
    else:
        prompt_builder = CopperNBSModelPanelPromptBuilder(
            model_panels=model_panels,
            fixed_asset_history=nbs_history,
            six_month_correlation=correlation,
            correlation_pairs=correlation_pairs,
            nbs_signal_weight=NBS_SIGNAL_WEIGHT,
        )
    return prompt_builder(task=task, context=context)


def build_agent_predictor(
    config: Any,
    model_panels: dict[str, list[dict[str, Any]]],
    nbs_history: list[dict[str, Any]],
    correlation: float | None,
    correlation_pairs: int,
) -> Predictor:
    """Build the revised predictor, including NBS context when available."""
    if correlation is None:
        return build_copper_model_panel_predictor(config, model_panels)
    return build_copper_nbs_model_panel_predictor(
        config,
        model_panels,
        fixed_asset_history=nbs_history,
        six_month_correlation=correlation,
        correlation_pairs=correlation_pairs,
        nbs_signal_weight=NBS_SIGNAL_WEIGHT,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff",
        type=parse_cutoff,
        default=DEFAULT_CUTOFF,
        help=f"Monthly forecast cutoff (default: {DEFAULT_CUTOFF.date()}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the numerical panel and prompt without calling the agent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for prompt, forecast, metrics, and chart files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one horizon-six revised-agent experiment."""
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    cutoff = cast(pd.Timestamp, args.cutoff)
    output_dir = cast(Path | None, args.output_dir) or default_output_dir(cutoff)
    output_dir.mkdir(parents=True, exist_ok=True)

    copper = load_copper_prices(cutoff)
    service = build_cutoff_service(copper, cutoff)
    context = service.context(as_of=cutoff)
    copper_history = context.get_series(COPPER_SERIES_ID).set_index("timestamp")["value"].astype(float)
    nbs_window, nbs_history, correlation, correlation_pairs = prepare_optional_nbs_context(copper, cutoff)
    nbs_context_used = correlation is not None

    task = build_task(cutoff)
    panel_rows = build_numerical_panel(copper_history, task, service, cutoff)
    model_panels = {str(cutoff.date()): panel_rows}
    prompt = build_agent_prompt(
        task,
        context,
        model_panels,
        nbs_history,
        correlation,
        correlation_pairs,
    )
    prompt_path = output_dir / "agent_prompt.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Model: {ADVANCED_MODEL}")
    print(f"Cutoff: {cutoff.date()} | history: {INPUT_MONTHS} months | horizons: {list(HORIZONS)}")
    if nbs_context_used:
        print(
            f"NBS six-month correlation: {cast(float, correlation):.3f} across {correlation_pairs} paired observations"
        )
        print(f"NBS signal weight: {NBS_SIGNAL_WEIGHT}")
    if args.dry_run:
        print(f"Dry run complete. Prompt written to {prompt_path}")
        return

    config = build_copper_news_config(
        model=ADVANCED_MODEL,
        search_model=ADVANCED_MODEL,
        verifier_model=ADVANCED_MODEL,
        nbs_signal_weight=NBS_SIGNAL_WEIGHT if nbs_context_used else "standard",
    )
    predictor = build_agent_predictor(
        config,
        model_panels,
        nbs_history,
        correlation,
        correlation_pairs,
    )
    predictions = predictor.predict(task, context)
    results = predictions_to_frame(predictions, copper)
    scored = results.loc[results["status"] == "Scored"]
    if scored.empty:
        raise ValueError("No target-month actuals are available, so MAE cannot be reported.")
    mae = float(scored["absolute_error"].mean())

    forecast_path = output_dir / "forecasts.csv"
    results.to_csv(forecast_path, index=False)
    metrics = {
        "model": ADVANCED_MODEL,
        "cutoff": str(cutoff.date()),
        "input_months": INPUT_MONTHS,
        "horizons_months": list(HORIZONS),
        "nbs_six_month_correlation": correlation,
        "nbs_correlation_pairs": correlation_pairs,
        "nbs_signal_weight": NBS_SIGNAL_WEIGHT if nbs_context_used else "unavailable",
        "mae_usd_per_metric_ton": mae,
        "scored_forecasts": int(len(scored)),
        "pending_forecasts": int((results["status"] == "Pending").sum()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    chart = build_chart(copper_history, nbs_window, results, cutoff, nbs_context_used)
    chart_path = output_dir / "forecast_trends_and_error.html"
    chart.write_html(chart_path, include_plotlyjs=True)

    display_columns = ["forecast_date", "point_forecast", "actual", "absolute_error", "status"]
    display_frame = results[display_columns].copy()
    display_frame[["point_forecast", "actual", "absolute_error"]] = display_frame[
        ["point_forecast", "actual", "absolute_error"]
    ].round(1)
    print(display_frame.to_string(index=False))
    print(f"MAE over {len(scored)} observed target months: ${mae:,.1f} per metric ton")
    print(f"Forecasts: {forecast_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
