from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.evaluation.task import ForecastingTask
from copper_forecasting.agent import (
    CopperModelPanelPromptBuilder,
    CopperNBSModelPanelPromptBuilder,
    build_copper_model_panel_config,
    build_copper_news_config,
)
from copper_forecasting.data import COPPER_SERIES_ID, build_copper_service
from copper_forecasting.prophet_baseline import CopperProphetPredictor
from copper_forecasting.run_revised_news_agent import (
    DEFAULT_CUTOFF,
    HORIZONS,
    INPUT_MONTHS,
    InsufficientNBSContextError,
    prepare_nbs_context,
)


def _write_cache(cache_dir: Path, periods: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=periods, freq="MS")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "value": 7000 + np.arange(periods) * 20 + 200 * np.sin(np.arange(periods) * np.pi / 6),
            "released_at": dates,
        }
    )
    frame.to_parquet(cache_dir / "PCOPPUSDM.parquet", index=False)
    return frame


def test_copper_service_reads_cache_without_api_key(tmp_path: Path, monkeypatch) -> None:
    expected = _write_cache(tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    actual = build_copper_service(tmp_path).get_series(
        COPPER_SERIES_ID,
        as_of=pd.Timestamp("2024-12-01"),
    )

    assert actual["value"].tolist() == expected["value"].tolist()


def test_prophet_forecast_dates_follow_monthly_task(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    service = build_copper_service(tmp_path)
    task = ForecastingTask(
        task_id="copper_test",
        target_series_id=COPPER_SERIES_ID,
        horizons=[1, 3, 6],
        frequency="MS",
        description="Synthetic monthly copper price forecast.",
    )
    as_of = pd.Timestamp("2024-12-01")

    predictions = CopperProphetPredictor().predict(task, service.context(as_of=as_of))

    assert [pd.Timestamp(prediction.forecast_date) for prediction in predictions] == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
    ]


def test_prophet_minimum_history_is_configurable(tmp_path: Path) -> None:
    _write_cache(tmp_path, periods=24)
    service = build_copper_service(tmp_path)
    task = ForecastingTask(
        task_id="copper_short_history_test",
        target_series_id=COPPER_SERIES_ID,
        horizons=[1, 2, 3, 4, 5, 6],
        frequency="MS",
        description="Copper forecast using a fixed two-year input window.",
    )
    as_of = pd.Timestamp("2020-12-01")

    predictions = CopperProphetPredictor(min_history=24).predict(task, service.context(as_of=as_of))

    assert len(predictions) == 6


def test_model_panel_prompt_uses_results_for_current_cutoff(tmp_path: Path) -> None:
    _write_cache(tmp_path, periods=24)
    service = build_copper_service(tmp_path)
    task = ForecastingTask(
        task_id="copper_model_panel_test",
        target_series_id=COPPER_SERIES_ID,
        horizons=[1, 2],
        frequency="MS",
        description="Copper forecast informed by numerical model results.",
    )
    as_of = pd.Timestamp("2020-12-01")
    expected_panel = [{"model": "AutoARIMA", "horizon": 1, "forecast": 8100.0, "past_mae": 250.0}]
    builder = CopperModelPanelPromptBuilder(model_panels={"2020-12-01": expected_panel})

    payload = json.loads(builder(task=task, context=service.context(as_of=as_of)))

    assert payload["numerical_model_results"] == expected_panel
    assert "news" not in payload


def test_nbs_model_panel_prompt_includes_fixed_asset_context(tmp_path: Path) -> None:
    _write_cache(tmp_path, periods=24)
    service = build_copper_service(tmp_path)
    task = ForecastingTask(
        task_id="copper_nbs_panel_test",
        target_series_id=COPPER_SERIES_ID,
        horizons=[1, 2, 3, 4, 5, 6],
        frequency="MS",
        description="Copper forecast informed by Chinese fixed-asset growth.",
    )
    as_of = pd.Timestamp("2020-12-01")
    fixed_asset_history = [
        {"month": f"2020-{month:02d}-01", "growth_yoy_percent": value}
        for month, value in zip(range(6, 12), [5.1, 4.3, 3.2, 1.8, 0.4, -1.2], strict=True)
    ]
    builder = CopperNBSModelPanelPromptBuilder(
        model_panels={"2020-12-01": []},
        fixed_asset_history=fixed_asset_history,
        six_month_correlation=-0.45,
        correlation_pairs=28,
        nbs_signal_weight="high",
    )

    payload = json.loads(builder(task=task, context=service.context(as_of=as_of)))
    nbs_context = payload["chinese_nbs_fixed_asset_growth"]

    assert nbs_context["history"] == fixed_asset_history
    assert nbs_context["six_month_relationship"]["correlation"] == -0.45
    assert nbs_context["six_month_relationship"]["pairs"] == 28
    assert "descriptive, not causal" in nbs_context["six_month_relationship"]["guidance"]
    assert nbs_context["signal_weight"] == "high"
    assert nbs_context["recent_trend"]["direction"] == "falling"
    assert "falling fixed-asset growth is a bullish" in nbs_context["six_month_relationship"]["guidance"]


def test_revised_runner_defaults_to_january_cutoff_and_only_horizon_six() -> None:
    assert pd.Timestamp("2026-01-01") == DEFAULT_CUTOFF
    assert HORIZONS == (6,)


def test_prepare_nbs_context_uses_cumulative_six_month_copper_return() -> None:
    dates = pd.date_range(
        DEFAULT_CUTOFF - pd.DateOffset(months=INPUT_MONTHS - 1),
        DEFAULT_CUTOFF,
        freq="MS",
    )
    copper_values = pd.Series(8000 + np.arange(INPUT_MONTHS) ** 2 * 8, index=dates, dtype=float)
    six_month_forward_return = (copper_values.shift(-6) / copper_values - 1) * 100
    fixed_asset_growth = (-six_month_forward_return).fillna(0).rename("fixed_asset_growth_yoy_percent")
    copper = pd.DataFrame({"value": copper_values}, index=dates)

    window, history, correlation, pairs = prepare_nbs_context(
        fixed_asset_growth,
        copper,
        DEFAULT_CUTOFF,
    )

    assert len(window) == INPUT_MONTHS
    assert len(history) == INPUT_MONTHS
    assert pairs == INPUT_MONTHS - 6
    assert correlation == pytest.approx(-1.0)


def test_prepare_nbs_context_rejects_insufficient_history() -> None:
    dates = pd.date_range(
        DEFAULT_CUTOFF - pd.DateOffset(months=INPUT_MONTHS - 1),
        DEFAULT_CUTOFF,
        freq="MS",
    )
    copper = pd.DataFrame({"value": np.arange(INPUT_MONTHS, dtype=float)}, index=dates)
    fixed_asset_growth = pd.Series(dtype=float, name="fixed_asset_growth_yoy_percent")

    with pytest.raises(InsufficientNBSContextError, match="found 0"):
        prepare_nbs_context(fixed_asset_growth, copper, DEFAULT_CUTOFF)


def test_model_panel_agent_has_no_news_retrieval() -> None:
    config = build_copper_model_panel_config()

    assert config.context_retrieval.enabled is False


def test_news_agent_reports_signals_used() -> None:
    config = build_copper_news_config()

    assert config.context_retrieval.enabled is True
    assert "Global signals used:" in config.instruction
    assert "global inventory" in config.instruction
    assert "USD-per-metric-ton impact" in config.instruction
    assert "Alternative interpretation:" in config.instruction
    assert "geographic relocation" in config.context_retrieval.instruction


def test_news_agent_can_give_nbs_signal_high_weight() -> None:
    config = build_copper_news_config(nbs_signal_weight="high")

    assert "Give `chinese_nbs_fixed_asset_growth` high weight" in config.instruction
    assert "Apply the sign exactly" in config.instruction
