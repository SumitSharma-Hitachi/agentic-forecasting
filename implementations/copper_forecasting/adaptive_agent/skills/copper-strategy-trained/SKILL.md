---
name: copper-strategy-trained
description: >-
  Experimental trained strategy for the adaptive monthly copper-price analyst.
  Load this skill before every prediction that uses the trained adaptive strategy
  configuration.
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

### Sudden-fall rebound correction

When the latest observed price fall is unusually large relative to recent
monthly changes, do not extrapolate the full decline through all six horizons.
Keep the post-fall level influential, but include a plausible partial rebound in
the central path when supported by the numerical panel. Widen the forecast range
to cover both continued weakness and rebound risk; do not assume an immediate
return to the pre-fall price.

### Validated trend-agreement correction

When several independently fitted trend-oriented models produce upward paths
and the models with available past-only validation scores are among the stronger
recent performers, preserve a meaningful upward slope in the central forecast.
Temper an extreme slope when model families disagree or validation is weak, but
do not flatten a well-supported upward trend back to a nearly constant path.

## Training boundary

These corrections are an exploratory, frozen variant proposed after inspecting
the same three scenarios used in the copper backtest. Results from those
scenarios are in-sample evidence and must not be presented as protected or
unbiased validation. The evaluation remains read-only and cannot modify this
skill from target-period outcomes.