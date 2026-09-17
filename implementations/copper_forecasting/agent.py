"""Agent configuration and prompt builder for copper price forecasting."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, ContextRetrievalConfig
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import BaseModel


def compress_copper_history(history: pd.DataFrame, *, max_months: int = 180) -> str:
    """Serialize the most recent monthly observations as compact CSV."""
    recent = history.tail(max_months)
    rows = ["date,price_usd_per_metric_ton"]
    rows.extend(
        f"{pd.Timestamp(row.timestamp).date()},{float(row.value):.2f}" for row in recent.itertuples(index=False)
    )
    return "\n".join(rows)


class CopperForecastPromptBuilder(BaseModel):
    """Build a structured payload from cutoff-safe copper price history."""

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        history = context.get_series(task.target_series_id)
        trailing_year = history["value"].tail(12)
        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons_months": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_price_usd_per_metric_ton": float(history["value"].iloc[-1]),
                "last_observation": str(pd.Timestamp(history["timestamp"].iloc[-1]).date()),
                "observations": int(len(history)),
                "trailing_12m_high": float(trailing_year.max()),
                "trailing_12m_low": float(trailing_year.min()),
            },
            "target_history_csv": compress_copper_history(history),
        }
        return json.dumps(payload, indent=2)


def _analyst_instruction() -> str:
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "You are an expert copper market analyst producing calibrated probabilistic "
        "forecasts for the global monthly copper price in USD per metric ton. Ground "
        "your forecast in the supplied price history and, when available, retrieved "
        "evidence about mine supply, inventories, Chinese demand, the US dollar, and "
        "energy-transition demand.\n\n"
        "Produce one forecast for every value in `horizons_months`. Use exactly the "
        "provided quantile levels, make the 0.50 quantile equal the point forecast, "
        "and keep quantiles non-decreasing. Put assumptions and uncertainty drivers "
        "in the rationale fields.\n\n"
        "If `set_model_response` is available, call it once with the complete JSON as "
        "`json_response`; otherwise return only JSON matching this schema:\n" + schema
    )


_SEARCH_INSTRUCTION = """\
You are a copper market research specialist. Search for evidence available on or
before the supplied cutoff date about copper mine supply and disruptions, LME/COMEX
inventories, Chinese industrial demand, the US dollar, treatment charges, and
published copper outlooks. Return a concise sourced summary. Exclude any fact that
cannot be confidently placed on or before the cutoff date.
"""

_SEARCH_SUPPLEMENT = """\

Call `search_web` before forecasting. Pass the payload's `as_of` value unchanged as
`cutoff_date`. Search separately for (1) copper supply and inventories and (2) China
demand, the US dollar, and current analyst outlooks. If verification fails, use only
the supplied history and state that limitation in the rationale.
"""


def build_copper_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build a history-only copper analyst configuration."""
    return AgentConfig(name="copper_analyst_basic", model=model, instruction=_analyst_instruction())


def build_copper_news_config(
    model: str = LITE_MODEL,
    *,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
) -> AgentConfig:
    """Build a copper analyst with cutoff-aware web context retrieval."""
    return AgentConfig(
        name="copper_analyst_news",
        model=model,
        instruction=_analyst_instruction() + _SEARCH_SUPPLEMENT,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_SEARCH_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
        ),
    )


def build_copper_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a copper agent configuration in the standard predictor interface."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=CopperForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "CopperForecastPromptBuilder",
    "build_copper_agent_predictor",
    "build_copper_basic_config",
    "build_copper_news_config",
    "compress_copper_history",
]
