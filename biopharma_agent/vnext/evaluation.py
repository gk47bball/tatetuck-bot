from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .models import EventDrivenEnsemble
from .storage import LocalResearchStore


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return 0.0
    return float(a.rank().corr(b.rank(), method="pearson") or 0.0)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(slots=True)
class WalkForwardSummary:
    num_rows: int
    num_windows: int
    rank_ic: float
    hit_rate: float
    top_bottom_spread: float
    turnover: float
    max_drawdown: float
    beta_adjusted_return: float
    calibrated_brier: float
    leakage_passed: bool
    message: str = ""


class WalkForwardEvaluator:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()

    def build_training_frame(self) -> pd.DataFrame:
        features = self.store.read_table("feature_vectors")
        snapshots = self.store.read_table("company_snapshots")
        if features.empty or snapshots.empty:
            return pd.DataFrame()

        labels = self._build_labels(snapshots)
        if labels.empty:
            return pd.DataFrame()
        return features.merge(labels, on=["ticker", "as_of"], how="left")

    def _build_labels(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for ticker in sorted(set(snapshots["ticker"].tolist())):
            ticker_snaps = snapshots[snapshots["ticker"] == ticker].copy()
            if ticker_snaps.empty:
                continue
            start = (pd.to_datetime(ticker_snaps["as_of"]).min() - timedelta(days=14)).date().isoformat()
            end = (pd.to_datetime(ticker_snaps["as_of"]).max() + timedelta(days=220)).date().isoformat()
            history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if history.empty:
                continue
            close = history["Close"]
            close.index = pd.to_datetime(close.index, utc=True).tz_convert(None).astype("datetime64[ns]")
            for as_of in ticker_snaps["as_of"].tolist():
                timestamp = pd.Timestamp(as_of)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert(None)
                snapshot_ts = timestamp.to_datetime64().astype("datetime64[ns]")
                base_idx = close.index.values.searchsorted(snapshot_ts, side="left")
                if base_idx >= len(close):
                    continue
                base_price = float(close.iloc[base_idx])
                row = {"ticker": ticker, "as_of": as_of}
                for horizon, days in (("30d", 30), ("90d", 90), ("180d", 180)):
                    target_ts = snapshot_ts + np.timedelta64(days, "D")
                    future_idx = close.index.values.searchsorted(target_ts, side="left")
                    if future_idx >= len(close) or base_price <= 0:
                        row[f"target_return_{horizon}"] = np.nan
                    else:
                        future_price = float(close.iloc[future_idx])
                        row[f"target_return_{horizon}"] = (future_price / base_price) - 1.0
                catalyst_label = row.get("target_return_90d")
                row["target_catalyst_success"] = int(pd.notna(catalyst_label) and float(catalyst_label) > 0.08)
                rows.append(row)
        return pd.DataFrame(rows)

    def leakage_audit(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        forbidden_prefixes = ("target_",)
        feature_columns = [col for col in frame.columns if col not in {"entity_id", "ticker", "as_of", "thesis_horizon"}]
        if any(col.startswith(forbidden_prefixes) and col in feature_columns for col in feature_columns):
            return False
        if "market_flow_return_6mo_legacy" in feature_columns:
            return False
        return True

    def evaluate(self, min_train_rows: int = 24) -> WalkForwardSummary:
        frame = self.build_training_frame()
        if frame.empty:
            return WalkForwardSummary(
                num_rows=0,
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=False,
                message="No archived snapshots with forward labels yet. Collect snapshots over time, then rerun.",
            )

        frame["as_of"] = pd.to_datetime(frame["as_of"])
        frame = frame.sort_values("as_of")
        unique_dates = sorted(frame["as_of"].drop_duplicates().tolist())
        if len(unique_dates) < 3:
            return WalkForwardSummary(
                num_rows=len(frame),
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=self.leakage_audit(frame),
                message="Need at least three distinct as-of dates for walk-forward evaluation.",
            )

        ensemble = EventDrivenEnsemble(store=self.store)
        rank_ics: list[float] = []
        hit_rates: list[float] = []
        spreads: list[float] = []
        turnovers: list[float] = []
        briers: list[float] = []
        cumulative_returns: list[float] = []
        previous_top: set[str] = set()
        num_windows = 0

        for split_idx in range(2, len(unique_dates)):
            train_dates = unique_dates[:split_idx]
            test_date = unique_dates[split_idx]
            train = frame[frame["as_of"].isin(train_dates)].copy()
            test = frame[frame["as_of"] == test_date].copy()
            if len(train) < min_train_rows or test.empty:
                continue

            ensemble.fit(train)
            predictions = ensemble.score(
                [
                    self._row_to_feature_vector(row)
                    for _, row in test.iterrows()
                ]
            )
            pred_df = pd.DataFrame([pred.to_record() for pred in predictions])
            merged = test.merge(pred_df, on=["ticker", "as_of"], how="inner")
            if merged.empty:
                continue
            merged = merged.dropna(subset=["target_return_90d"])
            if merged.empty:
                continue

            num_windows += 1
            rank_ics.append(_spearman(merged["expected_return"], merged["target_return_90d"]))
            hit_rates.append(float((np.sign(merged["expected_return"]) == np.sign(merged["target_return_90d"])).mean()))
            top = merged.nlargest(max(1, len(merged) // 5), "expected_return")
            bottom = merged.nsmallest(max(1, len(merged) // 5), "expected_return")
            spreads.append(float(top["target_return_90d"].mean() - bottom["target_return_90d"].mean()))
            current_top = set(top["ticker"].tolist())
            turnovers.append(1.0 if not previous_top else 1.0 - (len(current_top & previous_top) / max(len(current_top | previous_top), 1)))
            previous_top = current_top
            briers.append(float(((merged["catalyst_success_prob"] - merged["target_catalyst_success"]) ** 2).mean()))
            cumulative_returns.append(float(top["target_return_90d"].mean()))

        if not num_windows:
            return WalkForwardSummary(
                num_rows=len(frame),
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=self.leakage_audit(frame),
                message="Not enough labeled windows to fit and test the ensemble yet.",
            )

        curve = pd.Series(cumulative_returns).cumsum()
        peak = curve.cummax()
        drawdown = (curve - peak).min()
        return WalkForwardSummary(
            num_rows=len(frame),
            num_windows=num_windows,
            rank_ic=_safe_mean(rank_ics),
            hit_rate=_safe_mean(hit_rates),
            top_bottom_spread=_safe_mean(spreads),
            turnover=_safe_mean(turnovers),
            max_drawdown=float(abs(drawdown)),
            beta_adjusted_return=_safe_mean(cumulative_returns),
            calibrated_brier=_safe_mean(briers),
            leakage_passed=self.leakage_audit(frame),
            message="ok",
        )

    @staticmethod
    def _row_to_feature_vector(row: pd.Series):
        from .entities import FeatureVector

        features = {
            key: float(row[key])
            for key in row.index
            if key
            not in {
                "entity_id",
                "ticker",
                "as_of",
                "thesis_horizon",
                "target_return_30d",
                "target_return_90d",
                "target_return_180d",
                "target_catalyst_success",
            }
            and not key.startswith("meta_")
        }
        metadata = {key[5:]: row[key] for key in row.index if key.startswith("meta_")}
        return FeatureVector(
            entity_id=row["entity_id"],
            ticker=row["ticker"],
            as_of=str(row["as_of"]),
            thesis_horizon=row["thesis_horizon"],
            feature_family=features,
            metadata=metadata,
        )
