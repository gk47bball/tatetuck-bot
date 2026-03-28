# Alpha Stack v2 Methodology

The **Alpha Stack v2** operates entirely within `strategy.py`. It is a sophisticated, pure-math (no external ML libs) multi-factor model for biopharma valuation. The entire signal pipeline is meticulously bound and scaled to prevent extreme outliers while retaining deep variance.

## 1. Dynamic POS (Probability of Success)
Instead of static phase-transition statistics, the model dynamically boosts or penalizes POS:
*   **Trial Volume Boost**: Modeled via `math.log10(phase_trial_count + 1)`. Companies running multiple concurrent trials in the same phase represent higher conviction.
*   **Enrollment Power**: Identifies the maximum enrollment of any single active trial. Large pivotal trials represent massive capital commitment, mapped via logarithmic scaling.
*   **Literature Backing**: Normalizes the volume of PubMed publications related to the active pipeline, acting as a proxy for academic and peer-reviewed conviction.
*   **Diversity Index**: Rewards pipelines targeting multiple unique conditions (shots on goal).

## 2. Advanced rNPV & Commercial Differentiation
*   **Clinical Stage**: Values assets through a standard discounted cash flow (DCF), mapping theoretical peak revenue scaled by disease-specific penetration assumptions, burning capital until theoretical launch.
*   **Commercial Stage**: Cross-references trailing `totalRevenue`. If revenue exceeds a commercial threshold (e.g., $25M), the valuation shifts to a growth-perpetuity model representing actual cashflow ramps rather than theoretical clinical probabilities.

## 3. The 5 Orthogonal Sub-Signals
The engine outputs a discrete signal bounded between `[-1.0, 1.0]`. It achieves this by squashing five independent sub-signals:

1.  **Fundamental Value**: Focuses on `rNPV / Market Cap`. Bounded via a sigmoid function (`2.0 / (1.0 + math.exp(-x)) - 1.0`).
2.  **Clinical Score**: Heavy interaction term combining raw enrollment mass and literature, normalized around typical biotech baselines to achieve a zero-centered output.
3.  **FDA Safety Penalty**: Isolates `serious` and `death` adverse events from openFDA, strongly divided by the enrollment base to calculate an actual risk-ratio. Mapped through an exponential decay penalty.
4.  **Financial Health**: Computes `(Cash - Debt) / Enterprise Value`. Shifted to a 0.30 baseline (typical clinical biotech ratio) to effectively reward robust balance sheets and penalize heavy leverage.
5.  **Market Regime & Autocorrelation**: Uses a trailing 6-month continuous return proxy and 3-month momentum interlocked with linear volatility decay.

## 4. Signal Synthesis
The signals are blended using conviction-weighted multipliers. The returned dictionary includes an `alpha_breakdown` mapping every specific signal component for full diagnostic transparency.
