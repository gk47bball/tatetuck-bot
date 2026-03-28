from __future__ import annotations

import math

import pandas as pd

from .entities import CompanySnapshot, FeatureVector, Program


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class FeatureEngineer:
    def build_program_features(self, snapshot: CompanySnapshot, program: Program) -> FeatureVector:
        lead_trial = program.trials[0] if program.trials else None
        enrollment = float(lead_trial.enrollment if lead_trial else 0)
        top_catalyst = program.catalyst_events[0] if program.catalyst_events else None
        runway_months = float(snapshot.metadata.get("runway_months", 0.0) or 0.0)
        revenue = snapshot.revenue
        market_cap = max(snapshot.market_cap, 1.0)
        tam = max(program.tam_estimate, 1.0)
        sec_revenue_ttm = float(snapshot.metadata.get("sec_revenue_ttm", revenue) or revenue or 0.0)
        sec_operating_cashflow = float(snapshot.metadata.get("sec_operating_cashflow", 0.0) or 0.0)
        recent_offering_signal = float(snapshot.metadata.get("recent_offering_signal", 0.0) or 0.0)
        filing_freshness_days = self._filing_freshness_days(snapshot)
        op_cf_margin = sec_operating_cashflow / max(sec_revenue_ttm, 1.0) if sec_revenue_ttm > 0 else 0.0

        features = {
            "program_quality_pos_prior": program.pos_prior,
            "program_quality_log_enrollment": math.log10(enrollment + 1.0),
            "program_quality_trial_count": float(len(program.trials)),
            "program_quality_tam_to_cap": math.log10(tam / market_cap),
            "program_quality_phase_score": self._phase_score(program.phase),
            "program_quality_endpoint_score": self._endpoint_score(lead_trial.primary_outcomes if lead_trial else []),
            "catalyst_timing_horizon_days": float(top_catalyst.horizon_days if top_catalyst else 180.0),
            "catalyst_timing_probability": float(top_catalyst.probability if top_catalyst else 0.35),
            "catalyst_timing_importance": float(top_catalyst.importance if top_catalyst else 0.40),
            "catalyst_timing_crowdedness": float(top_catalyst.crowdedness if top_catalyst else 0.30),
            "catalyst_timing_filing_freshness_days": filing_freshness_days,
            "commercial_execution_revenue_scale": math.log10(revenue + 1.0),
            "commercial_execution_revenue_to_cap": math.log10(max(revenue / market_cap, 1e-6)),
            "commercial_execution_has_product": 1.0 if snapshot.approved_products else 0.0,
            "commercial_execution_sec_revenue_scale": math.log10(sec_revenue_ttm + 1.0),
            "balance_sheet_cash_to_cap": snapshot.cash / market_cap,
            "balance_sheet_debt_to_cap": snapshot.debt / market_cap,
            "balance_sheet_runway_months": runway_months,
            "balance_sheet_financing_pressure": 1.0 if snapshot.financing_events else 0.0,
            "balance_sheet_operating_cashflow_margin": op_cf_margin,
            "balance_sheet_recent_offering_signal": recent_offering_signal,
            "market_flow_momentum_3mo": float(snapshot.momentum_3mo or 0.0),
            "market_flow_volatility": float(snapshot.volatility or 0.0),
        }

        features["balance_sheet_runway_score"] = _clamp(runway_months / 24.0, 0.0, 2.0)
        features["program_quality_modality_risk"] = self._modality_risk(program.modality)
        features["commercial_execution_growth_signal"] = (
            snapshot.approved_products[0].growth_signal if snapshot.approved_products else 0.0
        )

        return FeatureVector(
            entity_id=program.program_id,
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            thesis_horizon=self._horizon_label(top_catalyst.horizon_days if top_catalyst else 180),
            feature_family=features,
            metadata={
                "program_name": program.name,
                "phase": program.phase,
                "modality": program.modality,
            },
        )

    def build_company_aggregate_features(self, snapshot: CompanySnapshot) -> FeatureVector:
        top_horizon = min((event.horizon_days for event in snapshot.catalyst_events), default=180)
        aggregate = {
            "program_quality_program_count": float(len(snapshot.programs)),
            "program_quality_approved_product_count": float(len(snapshot.approved_products)),
            "catalyst_timing_company_event_count": float(len(snapshot.catalyst_events)),
            "catalyst_timing_nearest_event_days": float(top_horizon),
            "catalyst_timing_filing_freshness_days": self._filing_freshness_days(snapshot),
            "commercial_execution_revenue_scale": math.log10(snapshot.revenue + 1.0),
            "commercial_execution_sec_revenue_scale": math.log10(float(snapshot.metadata.get("sec_revenue_ttm", snapshot.revenue) or snapshot.revenue or 0.0) + 1.0),
            "balance_sheet_cash_to_cap": snapshot.cash / max(snapshot.market_cap, 1.0),
            "balance_sheet_debt_to_cap": snapshot.debt / max(snapshot.market_cap, 1.0),
            "balance_sheet_runway_months": float(snapshot.metadata.get("runway_months", 0.0) or 0.0),
            "balance_sheet_recent_offering_signal": float(snapshot.metadata.get("recent_offering_signal", 0.0) or 0.0),
            "market_flow_momentum_3mo": float(snapshot.momentum_3mo or 0.0),
            "market_flow_volatility": float(snapshot.volatility or 0.0),
        }
        return FeatureVector(
            entity_id=f"{snapshot.ticker}:company",
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            thesis_horizon=self._horizon_label(top_horizon),
            feature_family=aggregate,
            metadata={"aggregate": True},
        )

    def build_all(self, snapshot: CompanySnapshot) -> list[FeatureVector]:
        vectors = [self.build_program_features(snapshot, program) for program in snapshot.programs]
        vectors.append(self.build_company_aggregate_features(snapshot))
        return vectors

    @staticmethod
    def _phase_score(phase: str) -> float:
        scores = {
            "EARLY_PHASE1": 0.1,
            "PHASE1": 0.2,
            "PHASE2": 0.45,
            "PHASE3": 0.75,
            "NDA_BLA": 0.9,
            "APPROVED": 1.0,
        }
        return scores.get(phase, 0.2)

    @staticmethod
    def _endpoint_score(outcomes: list[str]) -> float:
        outcome_text = " ".join(outcomes).lower()
        score = 0.25
        if any(keyword in outcome_text for keyword in ("overall survival", "pfs", "response rate", "hemoglobin")):
            score += 0.35
        if any(keyword in outcome_text for keyword in ("safety", "adverse", "tolerability")):
            score += 0.10
        return min(score, 1.0)

    @staticmethod
    def _modality_risk(modality: str) -> float:
        risk_map = {
            "gene editing": 0.80,
            "gene therapy": 0.70,
            "cell therapy": 0.65,
            "antibody": 0.30,
            "small molecule": 0.25,
            "vaccine": 0.45,
            "rna": 0.55,
            "platform": 0.60,
        }
        return risk_map.get(modality, 0.50)

    @staticmethod
    def _horizon_label(days: int) -> str:
        if days <= 45:
            return "30d"
        if days <= 120:
            return "90d"
        return "180d"

    @staticmethod
    def _filing_freshness_days(snapshot: CompanySnapshot) -> float:
        as_of = snapshot.as_of.split("T")[0]
        filing_date = snapshot.metadata.get("last_10q_date") or snapshot.metadata.get("last_10k_date")
        if not filing_date:
            return 180.0
        try:
            as_of_dt = pd.Timestamp(as_of)
            filing_dt = pd.Timestamp(filing_date)
            return float(max((as_of_dt - filing_dt).days, 0))
        except Exception:
            return 180.0
