"""FRED data-service setup for monthly copper price forecasting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter


COPPER_FRED_ID = "PCOPPUSDM"
"""FRED series: global copper price, monthly, USD per metric ton."""

COPPER_SERIES_ID = "global_copper_price_usd_per_metric_ton"
"""Canonical target series ID used by the notebook and backtest spec."""

DEFAULT_CACHE_DIR = Path("data/fred")


def naive_utc_now() -> datetime:
    """Return the timezone-naive UTC timestamp expected by ``DataService``."""
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def build_copper_service(
    cache_dir: Path | None = None,
    *,
    refresh: bool = False,
) -> DataService:
    """Return a data service backed by FRED's monthly global copper series.

    A populated ``PCOPPUSDM.parquet`` cache is read without consulting the
    ``FRED_API_KEY`` environment variable. Set ``refresh=True`` only when a key
    is configured and a fresh API request is intended.
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    service = DataService()
    service.register(
        COPPER_SERIES_ID,
        FREDAdapter(COPPER_FRED_ID, cache_dir=resolved_cache_dir, refresh=refresh),
        SeriesMetadata(
            series_id=COPPER_SERIES_ID,
            description="Global price of copper from FRED series PCOPPUSDM",
            source="FRED (IMF primary commodity prices)",
            units="USD per metric ton",
            frequency="MS",
        ),
    )
    return service


__all__ = [
    "COPPER_FRED_ID",
    "COPPER_SERIES_ID",
    "DEFAULT_CACHE_DIR",
    "build_copper_service",
    "naive_utc_now",
]
