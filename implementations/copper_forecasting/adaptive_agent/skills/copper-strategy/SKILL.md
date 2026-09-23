---
name: copper-strategy
description: >-
  Seed strategy for the adaptive monthly copper-price analyst. Load this skill
  before every prediction that uses the adaptive strategy configuration.
---

# Copper Forecasting Strategy

## Approach

Treat the numerical model panel as evidence, not a vote. Start from the robust
center of the eligible positive forecasts, then use only cutoff-safe diagnostics
to decide whether a different path is justified.

At one- and two-month horizons, emphasize the latest price level and methods with
lower past-only validation error. At three- through six-month horizons, reduce
confidence in extrapolated trends and give more weight to agreement across
independent model families. When candidate paths disagree materially, widen the
forecast distribution instead of expressing false precision.

Do not treat a missing validation score as evidence that a model is better or
worse. Do not infer news, supply disruptions, demand shocks, or policy events
from model disagreement alone. The protected backtest provides no news context.

## Active calibration corrections

No learned calibration corrections are present in the seed strategy.

## Training boundary

This seed contains domain priors only. A trained replacement must be produced
from study data that ends before every evaluation target and then frozen before
the protected backtest runs.