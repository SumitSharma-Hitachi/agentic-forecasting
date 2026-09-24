"""Agent configuration and prompt builder for copper price forecasting."""

from __future__ import annotations

import json
from typing import Any, Literal

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


class CopperModelPanelPromptBuilder(BaseModel):
    """Add cutoff-safe numerical forecasts and past errors to the agent prompt."""

    model_config = {"extra": "forbid"}

    model_panels: dict[str, list[dict[str, Any]]]

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(CopperForecastPromptBuilder()(task=task, context=context))
        cutoff = str(pd.Timestamp(context.as_of).date())
        try:
            payload["numerical_model_results"] = self.model_panels[cutoff]
        except KeyError as exc:
            raise KeyError(f"No numerical model results were supplied for cutoff {cutoff}.") from exc
        payload["numerical_model_results_note"] = (
            "Each row contains a candidate forecast and, when available, its mean absolute error "
            "on a six-month check period that ended at the cutoff. No target-period values or news are included."
        )
        return json.dumps(payload, indent=2)


class CopperNBSModelPanelPromptBuilder(CopperModelPanelPromptBuilder):
    """Add Chinese fixed-asset history and its six-month relationship to copper."""

    fixed_asset_history: list[dict[str, Any]]
    six_month_correlation: float
    correlation_pairs: int
    nbs_signal_weight: Literal["standard", "high"] = "standard"

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(super().__call__(task=task, context=context))
        observed_growth = [
            float(row["growth_yoy_percent"])
            for row in self.fixed_asset_history
            if row.get("growth_yoy_percent") is not None
        ]
        recent_growth = observed_growth[-6:]
        recent_change = recent_growth[-1] - recent_growth[0]
        trend = "falling" if recent_change < 0 else "rising" if recent_change > 0 else "flat"
        if self.six_month_correlation < 0:
            relationship_guidance = (
                "The observed relationship is negative, so falling fixed-asset growth is a bullish "
                "directional signal for the six-month copper price level and rising growth is bearish. "
            )
        elif self.six_month_correlation > 0:
            relationship_guidance = (
                "The observed relationship is positive, so rising fixed-asset growth is a bullish "
                "directional signal for the six-month copper price level and falling growth is bearish. "
            )
        else:
            relationship_guidance = (
                "The observed relationship is zero, so fixed-asset growth supplies no directional "
                "signal for the six-month copper price level. "
            )
        payload["chinese_nbs_fixed_asset_growth"] = {
            "indicator": "Investment in Fixed Assets, Accumulated Growth Rate(%)",
            "unit": "year-over-year percent",
            "source": "National Bureau of Statistics of China",
            "history": self.fixed_asset_history,
            "signal_weight": self.nbs_signal_weight,
            "recent_trend": {
                "direction": trend,
                "observations": len(recent_growth),
                "first_growth_yoy_percent": recent_growth[0],
                "latest_growth_yoy_percent": recent_growth[-1],
                "change_percentage_points": recent_change,
            },
            "six_month_relationship": {
                "definition": (
                    "Pearson correlation between fixed-asset growth in month t and "
                    "the cumulative copper price return from month t through month t+6"
                ),
                "correlation": self.six_month_correlation,
                "pairs": self.correlation_pairs,
                "guidance": (
                    relationship_guidance
                    + f"Apply {self.nbs_signal_weight} weight to this directional prior. Treat the relationship "
                    "as descriptive, not causal, and do not linearly extrapolate its magnitude. Override it "
                    "only when stronger cutoff-safe evidence supports the opposite direction."
                ),
            },
            "availability_note": (
                "The source file has observation months but no publication timestamps. Missing "
                "months are null, and a one-month publication-lag assumption is used."
            ),
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
published copper outlooks. For inventories, distinguish a change in globally
available metal from geographic relocation caused by tariffs or arbitrage. Return a
concise sourced summary with publication title, publisher, and date. Exclude any fact
that cannot be confidently placed on or before the cutoff date.
"""

_SEARCH_SUPPLEMENT = """\

Call `search_web` before forecasting. Pass the payload's `as_of` value unchanged as
`cutoff_date`. Search separately for (1) copper supply and inventories and (2) China
demand, the US dollar, and current analyst outlooks. If verification fails, use only
the supplied history and state that limitation in the rationale.

Build the forecast from the price-history path, then adjust it only for verified
evidence. In the top-level rationale, include a `Global signals used:` section. For
each signal, state the observed fact, whether it changes global supply/demand or only
location/timing, its expected direction, the horizons it should affect, an approximate
USD-per-metric-ton impact, confidence, publication title, publisher, and date. Do not
list a signal unless it affected the forecast.

Do not count related observations as independent evidence: for example, low treatment
charges may confirm concentrate tightness already represented by a mine-supply signal.
Do not interpret a tariff-driven transfer into US warehouses as a global inventory
surplus without evidence that total available stocks increased. Give near-term
observable indicators more weight than structural multi-year narratives for this
six-month task. Include an `Alternative interpretation:` sentence naming the strongest
counter-case and explain why the final path gives it less weight.
"""


def build_copper_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build a history-only copper analyst configuration."""
    return AgentConfig(name="copper_analyst_basic", model=model, instruction=_analyst_instruction())


def build_copper_news_config(
    model: str = LITE_MODEL,
    *,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    nbs_signal_weight: Literal["standard", "high"] = "standard",
) -> AgentConfig:
    """Build a copper analyst with cutoff-aware web context retrieval."""
    nbs_instruction = ""
    if nbs_signal_weight == "high":
        nbs_instruction = (
            "\n\nGive `chinese_nbs_fixed_asset_growth` high weight as a directional prior for "
            "the six-month forecast. Apply the sign exactly as described in the supplied "
            "`six_month_relationship.guidance`. Anchor on the cutoff price, state how this "
            "prior changed the point forecast, and do not linearly extrapolate the correlation into "
            "an implausible price change. Override the direction only for stronger cutoff-safe evidence, "
            "and explain that evidence explicitly."
        )
    return AgentConfig(
        name="copper_analyst_news",
        model=model,
        instruction=_analyst_instruction() + _SEARCH_SUPPLEMENT + nbs_instruction,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_SEARCH_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
        ),
    )


def build_copper_model_panel_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build a no-news analyst that reviews cutoff-safe numerical model results."""
    return AgentConfig(
        name="copper_analyst_model_panel",
        model=model,
        instruction=(
            _analyst_instruction() + "\n\nReview `numerical_model_results` before forecasting. Compare the candidate "
            "paths, their disagreement, and their past-only mean absolute errors. Produce your "
            "own forecast rather than automatically selecting or averaging one model. You have "
            "no news or web-search tools, so do not claim knowledge of market events not present "
            "in the supplied data."
        ),
    )


def build_copper_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a copper agent configuration in the standard predictor interface."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=CopperForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


def build_copper_model_panel_predictor(
    config: AgentConfig,
    model_panels: dict[str, list[dict[str, Any]]],
) -> AgentPredictor:
    """Wrap a model-panel configuration with its cutoff-keyed numerical results."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=CopperModelPanelPromptBuilder(model_panels=model_panels),
        output_schema=ContinuousAgentForecastOutput,
    )


def build_copper_nbs_model_panel_predictor(
    config: AgentConfig,
    model_panels: dict[str, list[dict[str, Any]]],
    *,
    fixed_asset_history: list[dict[str, Any]],
    six_month_correlation: float,
    correlation_pairs: int,
    nbs_signal_weight: Literal["standard", "high"] = "standard",
) -> AgentPredictor:
    """Wrap a news agent with numerical and Chinese fixed-asset context."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=CopperNBSModelPanelPromptBuilder(
            model_panels=model_panels,
            fixed_asset_history=fixed_asset_history,
            six_month_correlation=six_month_correlation,
            correlation_pairs=correlation_pairs,
            nbs_signal_weight=nbs_signal_weight,
        ),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "CopperForecastPromptBuilder",
    "CopperModelPanelPromptBuilder",
    "CopperNBSModelPanelPromptBuilder",
    "build_copper_agent_predictor",
    "build_copper_basic_config",
    "build_copper_model_panel_config",
    "build_copper_model_panel_predictor",
    "build_copper_nbs_model_panel_predictor",
    "build_copper_news_config",
    "compress_copper_history",
]
