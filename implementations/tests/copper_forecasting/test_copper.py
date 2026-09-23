from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.task import ForecastingTask
from copper_forecasting.agent import (
    CopperModelPanelPromptBuilder,
    build_copper_adaptive_config,
    build_copper_model_panel_config,
    build_copper_news_config,
)
from copper_forecasting.data import COPPER_SERIES_ID, build_copper_service
from copper_forecasting.prophet_baseline import CopperProphetPredictor


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


def test_model_panel_agent_has_no_news_retrieval() -> None:
    config = build_copper_model_panel_config()

    assert config.context_retrieval.enabled is False


def test_adaptive_strategy_is_frozen_during_protected_evaluation(tmp_path: Path) -> None:
    seed_dir = tmp_path / "copper-strategy"
    trained_dir = tmp_path / "copper-strategy-trained"
    seed_dir.mkdir()
    trained_dir.mkdir()

    seed_config = build_copper_adaptive_config(seed_dir)
    trained_config = build_copper_adaptive_config(trained_dir)

    assert list(seed_config.skills_dirs) == [seed_dir]
    assert seed_config.extra_tools == ()
    assert seed_config.context_retrieval.enabled is False
    assert "read-only" in seed_config.instruction
    assert seed_config.name != trained_config.name


def test_news_agent_reports_signals_used() -> None:
    config = build_copper_news_config()

    assert config.context_retrieval.enabled is True
    assert "Global signals used:" in config.instruction
