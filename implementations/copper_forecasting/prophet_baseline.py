"""Prophet baseline for monthly copper price forecasts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import scipy.stats
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from prophet import Prophet


class CopperProphetPredictor(Predictor):
    """Prophet trend and yearly-seasonality baseline for monthly copper prices."""

    def __init__(self, *, interval_width: float = 0.80, min_history: int = 36) -> None:
        self._interval_width = interval_width
        self._min_history = min_history

    @property
    def predictor_id(self) -> str:
        return "copper_prophet_monthly"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit through the information cutoff and forecast each requested horizon."""
        history = context.get_series(task.target_series_id)
        if len(history) < self._min_history:
            return []

        training = history.loc[:, ["timestamp", "value"]].rename(columns={"timestamp": "ds", "value": "y"})
        training["ds"] = pd.to_datetime(training["ds"])

        logging.getLogger("prophet").setLevel(logging.ERROR)
        model = Prophet(
            interval_width=self._interval_width,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.fit(training)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        forecast_dates = [pd.Timestamp(context.as_of) + offset * horizon for horizon in task.horizons]
        future = pd.DataFrame({"ds": forecast_dates})
        forecast = model.predict(future).set_index("ds")
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        predictions: list[Prediction] = []
        for forecast_date in forecast_dates:
            row = forecast.loc[forecast_date]
            point_forecast = float(row["yhat"])
            sigma = max(
                (float(row["yhat_upper"]) - float(row["yhat_lower"])) / (2 * 1.96),
                1e-4,
            )
            quantiles = {
                quantile: float(scipy.stats.norm.ppf(quantile, loc=point_forecast, scale=sigma))
                for quantile in STANDARD_QUANTILES
            }
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date.to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=point_forecast, quantiles=quantiles),
                )
            )

        return predictions


__all__ = ["CopperProphetPredictor"]
