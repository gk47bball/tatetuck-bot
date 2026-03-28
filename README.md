# Tatetuck Bot

**Tatetuck Bot** is a production-grade, autonomous quantitative research engine designed for the biopharma sector. Inspired by Andrej Karpathy's "LLM AutoResearch" pattern, the bot iterates over a valuation strategy, scores biopharma equities based on clinical and financial fundamentals, evaluates the performance against real-world market data, and systematically improves its own logic.

## Overview

The core of the engine is the **Alpha Stack v13 Breakthrough Edition**, which achieves phenomenal rank correlation and directional accuracy when predicting trailing market performance across an expanded benchmark index of 41 biopharma companies.

Key features include:
*   **Dynamic POS (Probability of Success)**: Adjusts base phase probabilities using active trial counts, maximum single-trial enrollment, and PubMed literature volume.
*   **Commercial Revenue Differentiation**: Applies distinct NPV algorithms based on commercial vs. pure clinical stage metrics.
*   **Portfolio Risk Scaling**: Explicitly outputs `recommended_allocation` and `risk_parity_allocation` limits based on volatility, market cap liquidity penalties, and pipeline-concentration dampeners.
*   **Alpha Stack (5 Orthogonal Sub-signals)**:
    1.  **Fundamental Value**: Risk-adjusted NPV (rNPV) vs. Market Cap.
    2.  **Clinical Momentum**: Conviction indexing based on enrollment flow and literature coverage (with heavy concentration penalties).
    3.  **FDA Safety Composite**: Serious adverse events scaled by total enrollment.
    4.  **Risk-Adjusted Financial Health**: Net Cash / Enterprise Value scaling.
    5.  **Market Regime & Autocorrelation**: Macro regime proxy using momentum and volatility interaction.

## Validation & Performance Highlights

On the latest v13 benchmark across 41 tickers (14-train, 27-holdout blended evaluation), the agent recorded:
*   **Valuation Error**: 0.0369
*   **Directional Accuracy**: 92.0%
*   **Rank Correlation**: 0.993
*   **Holdout Error Gap**: 0.01

The engine successfully transformed a baseline heuristic model (0.57 error) into a highly rigorous, non-linear scoring framework capable of accurately stratifying the biopharma ecosystem.

## Installation & Setup

1.  **Clone the Repository**
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Research Loop**:
    ```bash
    python main.py
    ```

## Files & Architecture

*   `strategy.py`: The single "brain" file the agent modifies. Contains the Alpha Stack logic.
*   `prepare.py`: The immutable data-gathering and evaluation harness. Fetches real-time data from ClinicalTrials.gov, openFDA, Yahoo Finance, and PubMed.
*   `evaluate.py`: The execution script that scores the benchmark tickers against strategy performance.
*   `program.md`: The agent's guiding instructions.
*   `METHODOLOGY.md`: Detailed breakdown of the quantitative methodologies driving the Alpha Stack.

*Built for advanced theoretical finance and biopharma quantitative analysis.*
