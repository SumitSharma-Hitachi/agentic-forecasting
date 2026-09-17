from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.task import ForecastingTask
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
