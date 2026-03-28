from __future__ import annotations

import math

import pandas as pd

from .entities import CompanySnapshot, FeatureVector, Program
from .market_profile import build_snapshot_profile
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority, is_clinical_event_type


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


EVENT_TYPE_FEATURES = (
    "phase1_readout",
    "phase2_readout",
    "phase3_readout",
    "pdufa",
    "commercial_update",
    "earnings",
)


class FeatureEngineer:
    def build_program_features(self, snapshot: CompanySnapshot, program: Program) -> FeatureVector:
        profile = build_snapshot_profile(snapshot)
        lead_trial = program.trials[0] if program.trials else None
        enrollment = float(lead_trial.enrollment if lead_trial else 0)
        top_catalyst = program.catalyst_events[0] if program.catalyst_events else None
        company_primary_event = self._primary_company_event(snapshot)
        runway_months = float(snapshot.metadata.get("runway_months", 0.0) or 0.0)
        revenue = snapshot.revenue
        market_cap = max(snapshot.market_cap, 1.0)
        tam = max(program.tam_estimate, 1.0)
        sec_revenue_ttm = float(snapshot.metadata.get("sec_revenue_ttm", revenue) or revenue or 0.0)
        sec_operating_cashflow = float(snapshot.metadata.get("sec_operating_cashflow", 0.0) or 0.0)
        recent_offering_signal = float(snapshot.metadata.get("recent_offering_signal", 0.0) or 0.0)
        filing_freshness_days = self._filing_freshness_days(snapshot)
        op_cf_margin = sec_operating_cashflow / max(sec_revenue_ttm, 1.0) if sec_revenue_ttm > 0 else 0.0
        top_event_type = top_catalyst.event_type if top_catalyst else None
        company_event_type = company_primary_event.event_type if company_primary_event else None
        commercial_presence = bool(snapshot.approved_products) or bool(snapshot.metadata.get("commercial_revenue_present"))

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
            "catalyst_timing_expected_value": float(
                (top_catalyst.probability * top_catalyst.importance * (1.0 - top_catalyst.crowdedness))
                if top_catalyst
                else 0.10
            ),
            "catalyst_timing_clinical_focus": 1.0 if is_clinical_event_type(top_event_type) else 0.0,
            "catalyst_timing_company_event_priority": float(event_type_priority(company_event_type)),
            "catalyst_timing_company_event_clinical": 1.0 if is_clinical_event_type(company_event_type) else 0.0,
            "catalyst_timing_company_event_earnings": 1.0 if company_event_type == "earnings" else 0.0,
            "commercial_execution_revenue_scale": math.log10(revenue + 1.0),
            "commercial_execution_revenue_to_cap": math.log10(max(revenue / market_cap, 1e-6)),
            "commercial_execution_has_product": 1.0 if commercial_presence else 0.0,
            "commercial_execution_sec_revenue_scale": math.log10(sec_revenue_ttm + 1.0),
            "balance_sheet_cash_to_cap": snapshot.cash / market_cap,
            "balance_sheet_debt_to_cap": snapshot.debt / market_cap,
            "balance_sheet_runway_months": runway_months,
            "balance_sheet_financing_pressure": 1.0 if snapshot.financing_events else 0.0,
            "balance_sheet_operating_cashflow_margin": op_cf_margin,
            "balance_sheet_recent_offering_signal": recent_offering_signal,
            "market_flow_momentum_3mo": float(snapshot.momentum_3mo or 0.0),
            "market_flow_volatility": float(snapshot.volatility or 0.0),
            "state_profile_pre_commercial": 1.0 if profile["company_state"] == "pre_commercial" else 0.0,
            "state_profile_commercial_launch": 1.0 if profile["company_state"] == "commercial_launch" else 0.0,
            "state_profile_commercialized": 1.0 if profile["company_state"] == "commercialized" else 0.0,
            "state_profile_competition_intensity": float(profile["competition_intensity"]),
            "state_profile_floor_support_pct": float(profile["floor_support_pct"]),
            "state_profile_launch_progress_pct": float(profile["launch_progress_pct"]),
            "state_profile_lifecycle_management_score": float(profile["lifecycle_management_score"]),
            "state_profile_pipeline_optionality_score": float(profile["pipeline_optionality_score"]),
            "state_profile_capital_deployment_score": float(profile["capital_deployment_score"]),
            "state_profile_hard_catalyst_presence": 1.0 if profile["has_near_term_hard_catalyst"] else 0.0,
            "state_profile_precommercial_value_gap": math.log10(max(tam / market_cap, 1e-6)),
        }

        features["balance_sheet_runway_score"] = _clamp(runway_months / 24.0, 0.0, 2.0)
        features["program_quality_modality_risk"] = self._modality_risk(program.modality)
        features["commercial_execution_growth_signal"] = (
            snapshot.approved_products[0].growth_signal
            if snapshot.approved_products
            else 0.0
        )
        features.update(self._event_type_features(top_event_type, prefix="catalyst_timing_event"))
        features.update(self._event_type_features(company_event_type, prefix="catalyst_timing_company_event"))

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
                "event_type": top_event_type,
                "event_bucket": event_type_bucket(top_event_type),
                "event_status": top_catalyst.status if top_catalyst else None,
                "event_expected_date": top_catalyst.expected_date if top_catalyst else None,
                "company_primary_event_type": company_event_type,
                "company_primary_event_status": company_primary_event.status if company_primary_event else None,
                "company_primary_event_expected_date": company_primary_event.expected_date if company_primary_event else None,
                "company_state": profile["company_state"],
                "primary_indication": profile["primary_indication"],
            },
        )

    def build_company_aggregate_features(self, snapshot: CompanySnapshot) -> FeatureVector:
        profile = build_snapshot_profile(snapshot)
        company_primary_event = self._primary_company_event(snapshot)
        top_horizon = min((event.horizon_days for event in snapshot.catalyst_events), default=180)
        company_event_type = company_primary_event.event_type if company_primary_event else None
        aggregate = {
            "program_quality_program_count": float(len(snapshot.programs)),
            "program_quality_approved_product_count": float(len(snapshot.approved_products)),
            "catalyst_timing_company_event_count": float(len(snapshot.catalyst_events)),
            "catalyst_timing_nearest_event_days": float(top_horizon),
            "catalyst_timing_filing_freshness_days": self._filing_freshness_days(snapshot),
            "catalyst_timing_clinical_focus": 1.0 if is_clinical_event_type(company_event_type) else 0.0,
            "catalyst_timing_company_event_priority": float(event_type_priority(company_event_type)),
            "catalyst_timing_company_event_earnings": 1.0 if company_event_type == "earnings" else 0.0,
            "commercial_execution_revenue_scale": math.log10(snapshot.revenue + 1.0),
            "commercial_execution_sec_revenue_scale": math.log10(float(snapshot.metadata.get("sec_revenue_ttm", snapshot.revenue) or snapshot.revenue or 0.0) + 1.0),
            "commercial_execution_has_product": 1.0 if (snapshot.approved_products or snapshot.metadata.get("commercial_revenue_present")) else 0.0,
            "commercial_execution_launch_progress_pct": float(profile["launch_progress_pct"]),
            "commercial_execution_lifecycle_management_score": float(profile["lifecycle_management_score"]),
            "balance_sheet_cash_to_cap": snapshot.cash / max(snapshot.market_cap, 1.0),
            "balance_sheet_debt_to_cap": snapshot.debt / max(snapshot.market_cap, 1.0),
            "balance_sheet_runway_months": float(snapshot.metadata.get("runway_months", 0.0) or 0.0),
            "balance_sheet_recent_offering_signal": float(snapshot.metadata.get("recent_offering_signal", 0.0) or 0.0),
            "balance_sheet_floor_support_pct": float(profile["floor_support_pct"]),
            "balance_sheet_capital_deployment_score": float(profile["capital_deployment_score"]),
            "market_flow_momentum_3mo": float(snapshot.momentum_3mo or 0.0),
            "market_flow_volatility": float(snapshot.volatility or 0.0),
            "state_profile_pre_commercial": 1.0 if profile["company_state"] == "pre_commercial" else 0.0,
            "state_profile_commercial_launch": 1.0 if profile["company_state"] == "commercial_launch" else 0.0,
            "state_profile_commercialized": 1.0 if profile["company_state"] == "commercialized" else 0.0,
            "state_profile_competition_intensity": float(profile["competition_intensity"]),
            "state_profile_pipeline_optionality_score": float(profile["pipeline_optionality_score"]),
            "state_profile_hard_catalyst_presence": 1.0 if profile["has_near_term_hard_catalyst"] else 0.0,
        }
        aggregate.update(self._event_type_features(company_event_type, prefix="catalyst_timing_company_event"))
        return FeatureVector(
            entity_id=f"{snapshot.ticker}:company",
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            thesis_horizon=self._horizon_label(top_horizon),
            feature_family=aggregate,
            metadata={
                "aggregate": True,
                "event_type": company_event_type,
                "event_bucket": event_type_bucket(company_event_type),
                "event_status": company_primary_event.status if company_primary_event else None,
                "event_expected_date": company_primary_event.expected_date if company_primary_event else None,
                "company_state": profile["company_state"],
                "primary_indication": profile["primary_indication"],
            },
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

    @staticmethod
    def _event_type_features(event_type: str | None, prefix: str) -> dict[str, float]:
        payload = {f"{prefix}_{name}": 0.0 for name in EVENT_TYPE_FEATURES}
        if event_type in EVENT_TYPE_FEATURES:
            payload[f"{prefix}_{event_type}"] = 1.0
        return payload

    @staticmethod
    def _primary_company_event(snapshot: CompanySnapshot):
        if not snapshot.catalyst_events:
            return None
        return min(
            snapshot.catalyst_events,
            key=lambda event: (
                -event_timing_priority(event.status, event.expected_date, event.title),
                -event_type_priority(event.event_type),
                event.horizon_days,
                -event.importance,
                event.crowdedness,
            ),
        )
