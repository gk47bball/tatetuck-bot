from __future__ import annotations

from functools import lru_cache
import math
import os
import re

import pandas as pd

from .entities import CompanySnapshot, FeatureVector, Program
from .graph import select_lead_trial
from .market_profile import build_snapshot_profile, classify_company_state
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority, is_clinical_event_type, is_synthetic_event


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


EVENT_TYPE_FEATURES = (
    "phase1_readout",
    "phase2_readout",
    "phase3_readout",
    "pdufa",
    "adcom",
    "regulatory_update",
    "strategic_transaction",
    "portfolio_repositioning",
    "label_expansion",
    "capital_allocation",
    "commercial_update",
    "earnings",
)


class FeatureEngineer:
    def build_program_features(self, snapshot: CompanySnapshot, program: Program) -> FeatureVector:
        profile = build_snapshot_profile(snapshot)
        lead_trial = select_lead_trial(program)
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

        # ── Interaction features ───────────────────────────────────────────────
        # Ridge regression is a linear model; it cannot discover non-linear
        # relationships between features on its own.  These four interactions
        # capture the most important product relationships in biotech alpha:
        #
        # 1. quality × timing  — high POS AND near-term catalyst is the
        #    archetypal pre-catalyst long.  POS alone (far-away event) and
        #    near-term alone (low-POS Phase 1) each have modest expected return;
        #    their product identifies the genuine setup.
        #
        # 2. phase × competition  — a Phase 3 drug in a crowded indication is
        #    worth far less than a Phase 3 drug with clear competitive moat.
        #    The linear model gives credit to both independently; the product
        #    captures the moat premium.
        #
        # 3. value_gap × catalyst  — a deep TAM/cap discount with a near-term
        #    clinical event is the classic "catalyst to unlock value" trade.
        #    Neither feature alone guarantees alpha; the combination does.
        #
        # 4. cash_stress × horizon  — financing pressure + short runway + near-
        #    term event = a company that needs this catalyst to survive.
        #    This triple-threat flag identifies high-risk binary outcomes that
        #    should be sized very small regardless of headline expected_return.
        horizon_days = float(top_catalyst.horizon_days if top_catalyst else 180.0)
        phase_score = self._phase_score(program.phase)
        competition = float(profile["competition_intensity"])
        clinical_focus = 1.0 if is_clinical_event_type(top_event_type) else 0.0
        tam_to_cap_raw = math.log10(max(tam / market_cap, 1e-6))
        runway_score = _clamp(runway_months / 24.0, 0.0, 2.0)
        financing_pressure = 1.0 if snapshot.financing_events else 0.0
        near_term = 1.0 if horizon_days <= 90.0 else 0.0

        features["interaction_quality_x_timing"] = (
            program.pos_prior * max(1.0 - horizon_days / 365.0, 0.0)
        )
        features["interaction_phase_x_moat"] = phase_score * (1.0 - competition)
        features["interaction_value_gap_x_catalyst"] = (
            _clamp(tam_to_cap_raw, 0.0, 3.0) * clinical_focus
        )
        features["interaction_cash_stress_x_horizon"] = (
            financing_pressure * max(1.0 - runway_score / 2.0, 0.0) * near_term
        )

        # KOL proxy features — academic site quality as a public substitute for
        # proprietary key opinion leader relationships
        from .kol_proxy import program_site_quality
        site_quality = program_site_quality(program.trials)
        features["program_quality_amc_site_score"] = site_quality["amc_site_score"]
        features["program_quality_site_diversity"] = site_quality["site_count_score"]
        features["program_quality_international_reach"] = site_quality["international_score"]
        # Composite KOL proxy: AMC presence × site diversity
        features["program_quality_kol_proxy"] = (
            site_quality["amc_site_score"] * 0.60
            + site_quality["site_count_score"] * 0.25
            + site_quality["international_score"] * 0.15
        )

        # ── Options IV implied move (best-effort) ─────────────────────────────
        # The options market prices binary catalyst risk directly and in real
        # time.  An ATM straddle price / spot gives the market-consensus expected
        # move for the catalyst window.  When our model's expected_return exceeds
        # the options-implied move, the setup is genuinely underpriced.  When it
        # is below, we may be paying up for a catalyst the options market already
        # fully reflects.
        catalyst_date = top_catalyst.expected_date if top_catalyst else None
        features["catalyst_timing_iv_implied_move"] = self._fetch_iv_implied_move(
            snapshot.ticker, catalyst_date
        )

        # ── Black-Scholes straddle proxy for historical snapshots ─────────────
        # Live options chains are unavailable for past dates, so
        # catalyst_timing_iv_implied_move is 0.0 for all training rows.  An ATM
        # straddle's fair value under log-normal assumptions equals:
        #   straddle ≈ 2 × S × N(σ√T / 2) ≈ S × σ√T × 0.798
        # where 0.798 ≈ 2/√(2π) (Brenner-Subrahmanyam 1988 approximation).
        # Dividing by S gives the implied-move fraction: σ√T × 0.798.
        # This is NOT the options-market signal (which reflects future realised
        # vol + risk premium); it is a model-based lower bound that gives the
        # ridge regression a meaningful non-zero training signal to learn from.
        if features["catalyst_timing_iv_implied_move"] == 0.0:
            horizon_days = top_catalyst.horizon_days if top_catalyst else 0
            vol = float(snapshot.volatility or 0.0)
            if vol > 0.0 and horizon_days > 0:
                T = horizon_days / 252.0
                bs_implied_move = vol * math.sqrt(T) * 0.798
                features["catalyst_timing_iv_implied_move"] = _clamp(bs_implied_move, 0.0, 2.0)

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
        """
        Score endpoint quality based on FDA approvability and regulatory history.

        Hard clinical endpoints (OS, PFS) are the gold standard — they directly
        measure patient benefit and have the highest FDA acceptance rate.
        Surrogate endpoints (response rate, biomarkers) require additional
        validation and face more complete-response letters.
        Safety-only endpoints (Phase 1) are informative but not approvable.

        Scores calibrated to FDA Complete Response Letter (CRL) frequency by
        endpoint type: OS-powered trials have ~10% CRL rate, surrogate-only
        trials have ~35% CRL rate, safety trials are not yet at approval stage.
        """
        endpoint_scores = [
            FeatureEngineer._single_endpoint_score(str(outcome or ""))
            for outcome in outcomes
            if str(outcome or "").strip()
        ]
        if not endpoint_scores:
            return 0.40
        if len(endpoint_scores) == 1:
            return endpoint_scores[0]

        # Co-primary designs are constrained by the weakest clinically relevant
        # endpoint, so we do not let a single strong endpoint fully dominate.
        weakest = min(endpoint_scores)
        strongest = max(endpoint_scores)
        return _clamp((0.65 * weakest) + (0.35 * strongest), 0.25, 0.90)

    @staticmethod
    def _single_endpoint_score(outcome: str) -> float:
        outcome_text = f" {outcome.lower()} "

        # Hard clinical endpoints — direct patient benefit, FDA gold standard
        if "overall survival" in outcome_text or re.search(r"\bos\b", outcome_text) or "death from any" in outcome_text:
            return 0.90
        if any(
            marker in outcome_text
            for marker in ("progression-free survival", "event-free survival", "relapse-free", "forced vital capacity")
        ) or re.search(r"\bpfs\b", outcome_text) or re.search(r"\befs\b", outcome_text):
            return 0.75
        # Disease-specific hard endpoints with established FDA precedent
        if any(marker in outcome_text for marker in ("hemoglobin", "vaso-occlusive", "transfusion independence", "bleed")):
            return 0.80
        if any(marker in outcome_text for marker in ("forced expiratory", "fev1", "lung function")):
            return 0.75
        if any(marker in outcome_text for marker in ("major adverse cardiovascular", "cardiovascular death")) or re.search(r"\bmace\b", outcome_text):
            return 0.80
        # Functional / patient-reported outcomes — FDA accepted but higher CRL risk
        if any(marker in outcome_text for marker in ("functional independence", "activities of daily", "disability")):
            return 0.65
        # Surrogate endpoints — approvable but vulnerable to confirmatory failure
        if any(marker in outcome_text for marker in ("objective response", "response rate", "complete response", "partial response")) or re.search(r"\borr\b", outcome_text):
            return 0.55
        if any(marker in outcome_text for marker in ("viral load", "hba1c", "ldl", "bone mineral density")):
            return 0.60
        # Biomarker / pharmacodynamic — typically Phase 1/2; not independently approvable
        if any(marker in outcome_text for marker in ("pharmacokinetic", "pharmacodynamic", "biomarker", "dose-limiting")):
            return 0.30
        # Safety / tolerability only — Phase 1 standard; no efficacy claim
        if any(marker in outcome_text for marker in ("safety", "adverse", "tolerability", "maximum tolerated")):
            return 0.25

        return 0.40  # unknown endpoint — conservative prior

    @staticmethod
    def _modality_risk(modality: str) -> float:
        # Risk scores calibrated to recent regulatory track record (2018-2024).
        # "gene editing": reduced from 0.80 — CASGEVY approval demonstrates commercial
        #   viability; primary risks now execution (manufacturing, delivery) not biology.
        # "vaccine": reduced from 0.45 — mRNA platform has proven Phase 3 reliability;
        #   traditional vaccine Phase 3 LoA is one of the highest across modalities.
        risk_map = {
            "gene editing": 0.65,   # post-CASGEVY; key risk is delivery & durability
            "gene therapy": 0.70,   # AAV immunogenicity & redosing remain open issues
            "cell therapy": 0.65,   # solid tumor penetration still uncertain; heme success high
            "antibody": 0.30,       # well-characterised modality, high Phase 3 LoA
            "small molecule": 0.25, # most predictable from Phase 2 data
            "vaccine": 0.35,        # mRNA platform reliable; adjuvanted vaccines also strong
            "rna": 0.55,            # siRNA/ASO successes offset CNS delivery challenges
            "platform": 0.60,       # unknown modality — apply conservative prior
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
    @lru_cache(maxsize=512)
    def _fetch_iv_implied_move(ticker: str, catalyst_date: str | None) -> float:
        """
        Estimate the options market's implied move for the catalyst window.

        Returns straddle_price / spot_price as a fraction (e.g. 0.25 = ±25%
        expected move).  Falls back to 0.0 on any data failure so it never
        blocks snapshot building.

        Method: find the listed expiry immediately after catalyst_date (or the
        nearest front-month if no date is known), fetch the ATM call + put at
        that expiry, return (call_ask + put_ask) / spot.  Using ask-side gives a
        conservative (buy-side) implied move estimate — appropriate since we are
        evaluating whether buying the catalyst is attractively priced.
        """
        live_market_data_enabled = os.environ.get("TATETUCK_ENABLE_LIVE_MARKET_DATA", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not live_market_data_enabled:
            return 0.0
        try:
            import yfinance as yf
            from datetime import date, timedelta

            tk = yf.Ticker(ticker)
            spot_hist = tk.history(period="2d", auto_adjust=True)
            if spot_hist.empty:
                return 0.0
            spot = float(spot_hist["Close"].iloc[-1])
            if spot <= 0:
                return 0.0

            expiries = tk.options  # tuple of "YYYY-MM-DD" strings
            if not expiries:
                return 0.0

            # Choose the expiry that best brackets the catalyst date
            today = date.today()
            if catalyst_date:
                try:
                    cat_dt = date.fromisoformat(catalyst_date[:10])
                except ValueError:
                    cat_dt = today + timedelta(days=90)
            else:
                cat_dt = today + timedelta(days=90)

            # Prefer the first expiry ON OR AFTER the catalyst date; fall back
            # to the nearest available expiry (front month) if none qualifies.
            target = None
            for exp in expiries:
                try:
                    exp_dt = date.fromisoformat(exp)
                except ValueError:
                    continue
                if exp_dt >= cat_dt:
                    target = exp
                    break
            if target is None:
                target = expiries[-1]  # use furthest available

            chain = tk.option_chain(target)
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                return 0.0

            # Find ATM strike (closest to spot)
            strikes = calls["strike"].values
            atm_idx = int(abs(strikes - spot).argmin())
            atm_strike = float(strikes[atm_idx])

            call_row = calls[calls["strike"] == atm_strike]
            put_row = puts[puts["strike"] == atm_strike]
            if call_row.empty or put_row.empty:
                return 0.0

            call_ask = float(call_row["ask"].iloc[0])
            put_ask = float(put_row["ask"].iloc[0])
            if call_ask <= 0 or put_ask <= 0:
                return 0.0

            implied_move = (call_ask + put_ask) / spot
            return float(_clamp(implied_move, 0.0, 2.0))
        except Exception:
            return 0.0

    @staticmethod
    def _primary_company_event(snapshot: CompanySnapshot):
        if not snapshot.catalyst_events:
            return None
        candidates = list(snapshot.catalyst_events)
        reference_ts = pd.to_datetime(snapshot.as_of, errors="coerce", utc=True, format="mixed")
        if not pd.isna(reference_ts):
            reference_ts = reference_ts.tz_convert(None)
            upcoming = []
            for event in candidates:
                event_ts = pd.to_datetime(event.expected_date, errors="coerce", utc=True, format="mixed")
                if pd.isna(event_ts):
                    continue
                event_ts = event_ts.tz_convert(None)
                if "T" not in str(event.expected_date or "") and len(str(event.expected_date or "")) <= 10:
                    if event_ts.date() >= reference_ts.date():
                        upcoming.append(event)
                elif event_ts >= reference_ts:
                    upcoming.append(event)
            if upcoming:
                candidates = upcoming
        state = classify_company_state(snapshot)
        if state in {"commercial_launch", "commercialized"}:
            strategic = [
                event
                for event in candidates
                if event_type_bucket(event.event_type) == "strategic"
                and not is_synthetic_event(event.status, event.title)
                and event.horizon_days <= 180
            ]
            exact_hard = [
                event
                for event in candidates
                if event_type_bucket(event.event_type) in {"clinical", "regulatory"}
                and not is_synthetic_event(event.status, event.title)
                and event.horizon_days <= 180
            ]
            if strategic:
                strategic_with_timing = [
                    event
                    for event in strategic
                    if event_timing_priority(event.status, event.expected_date, event.title) >= 1
                ]
                candidates = strategic_with_timing or strategic
            elif exact_hard:
                candidates = exact_hard
            else:
                near_term = [event for event in candidates if event.horizon_days <= 90]
                if near_term:
                    candidates = near_term
        return min(
            candidates,
            key=lambda event: (
                -event_timing_priority(event.status, event.expected_date, event.title),
                -event_type_priority(event.event_type),
                event.horizon_days,
                -event.importance,
                event.crowdedness,
            ),
        )
