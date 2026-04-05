from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from ..agents.pubmed_agent import AutoResearchAgent
from .entities import CompanyAnalysis, CompanySnapshot, ModelPrediction
from .features import FeatureEngineer
from .graph import select_lead_trial
from .market_profile import build_expectation_lens, build_snapshot_profile, classify_company_state
from .models import EventDrivenEnsemble
from .portfolio import PortfolioConstructor, aggregate_signal
from .replay import snapshot_from_dict
from .sources import IngestionService
from .storage import LocalResearchStore
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority, is_synthetic_event

PHASE_RANK = {
    "EARLY_PHASE1": 1,
    "PHASE1": 2,
    "PHASE2": 3,
    "PHASE3": 4,
    "NDA_BLA": 5,
    "APPROVED": 6,
}


class TatetuckPlatform:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()
        self.ingestion = IngestionService(store=self.store)
        self.features = FeatureEngineer()
        self.ensemble = EventDrivenEnsemble(store=self.store)
        self.portfolio = PortfolioConstructor(store=self.store)
        self.literature = AutoResearchAgent()

    def analyze_ticker(
        self,
        ticker: str,
        company_name: str | None = None,
        include_literature: bool = True,
        as_of: datetime | None = None,
        prefer_archive: bool = False,
        fallback_to_archive: bool = True,
        persist: bool = True,
        max_archive_age_days: int | None = None,
        allow_stale_archive_fallback: bool = True,
    ) -> CompanyAnalysis:
        analysis_time = as_of or datetime.now(timezone.utc)
        snapshot: CompanySnapshot | None = None
        analysis_source = "live"
        archived_snapshot = self._load_archived_snapshot(ticker)
        archived_snapshot_age_days: int | None = None
        archived_snapshot_is_stale = False
        if archived_snapshot is not None:
            archived_snapshot_age_days = self._snapshot_age_days(archived_snapshot, reference_time=analysis_time)
            archived_snapshot_is_stale = (
                max_archive_age_days is not None
                and archived_snapshot_age_days is not None
                and archived_snapshot_age_days > max_archive_age_days
            )

        if prefer_archive:
            if archived_snapshot is not None and not archived_snapshot_is_stale:
                snapshot = archived_snapshot
                analysis_source = "archive"

        if snapshot is None:
            try:
                snapshot = self.ingestion.ingest_company(ticker, company_name, as_of=as_of, persist=persist)
            except Exception as exc:
                if not fallback_to_archive:
                    raise
                snapshot = archived_snapshot
                if snapshot is None:
                    raise
                if archived_snapshot_is_stale and not allow_stale_archive_fallback:
                    age_text = (
                        f"{archived_snapshot_age_days} days old"
                        if archived_snapshot_age_days is not None
                        else "too old"
                    )
                    raise RuntimeError(
                        f"Live ingestion failed for {ticker} and the latest archived snapshot is {age_text}."
                    ) from exc
                snapshot.metadata["live_ingestion_error"] = f"{type(exc).__name__}: {exc}"
                analysis_source = "archive_fallback"
            else:
                analysis_source = "live"

        feature_vectors = self.features.build_all(snapshot)
        if persist:
            self.store.write_snapshot(snapshot)
            self.store.write_feature_vectors(feature_vectors)

        program_vectors = [vector for vector in feature_vectors if not vector.metadata.get("aggregate")]
        if not program_vectors:
            program_vectors = [vector for vector in feature_vectors if vector.metadata.get("aggregate")]
        program_predictions = self.ensemble.score(program_vectors, persist=persist)
        company_state = classify_company_state(snapshot)
        signal = self._aggregate_company_signal(snapshot, program_predictions, company_state=company_state)
        self._apply_special_situation_adjustments(snapshot, signal)
        signal.company_state = company_state
        primary_event = self._primary_event(snapshot, preferred_event_type=signal.primary_event_type)
        peer_context = self._peer_context(snapshot)
        expectation_lens = build_expectation_lens(snapshot, signal, primary_event, peer_context)
        signal.setup_type = str(expectation_lens["setup_type"])
        signal.internal_value = float(expectation_lens["internal_value"])
        signal.internal_price_target = (
            None if expectation_lens["internal_price_target"] is None else float(expectation_lens["internal_price_target"])
        )
        signal.internal_upside_pct = float(expectation_lens["internal_upside_pct"])
        signal.floor_support_pct = float(expectation_lens["floor_support_pct"])
        portfolio_rec = self.portfolio.recommend(signal)
        literature_review = ""
        if include_literature:
            drug_names = [program.name for program in snapshot.programs[:3]]
            conditions = [condition for program in snapshot.programs[:2] for condition in program.conditions[:2]]
            literature_review = self.literature.generate_literature_review(snapshot.company_name, drug_names, conditions)

        profile = build_snapshot_profile(snapshot)
        expectations_summary = self._expectations_summary(snapshot, signal, primary_event)
        kill_points = self._kill_points(snapshot, signal, primary_event)
        why_now = self._why_now(snapshot, signal, primary_event, expectations_summary, peer_context["summary"], expectation_lens["state_focus"])
        combined_risk_flags: list[str] = []
        for flag in [*portfolio_rec.risk_flags, *kill_points]:
            if flag and flag not in combined_risk_flags:
                combined_risk_flags.append(flag)
        portfolio_rec.risk_flags = combined_risk_flags[:6]

        if persist:
            self.store.write_signal_artifacts([signal])
            self.store.write_portfolio_recommendations([portfolio_rec])

        return CompanyAnalysis(
            snapshot=snapshot,
            signal=signal,
            portfolio=portfolio_rec,
            feature_vectors=feature_vectors,
            program_predictions=program_predictions,
            literature_review=literature_review,
            metadata={
                "store_dir": str(self.store.base_dir),
                "analysis_source": analysis_source,
                "persisted": persist,
                "analysis_as_of": analysis_time.isoformat(),
                "snapshot_age_days": self._snapshot_age_days(snapshot, reference_time=analysis_time),
                "archived_snapshot_age_days": archived_snapshot_age_days,
                "stale_archive_available": archived_snapshot_is_stale,
                "company_state": company_state,
                "setup_type": signal.setup_type,
                "primary_event_date": primary_event.expected_date if primary_event is not None else None,
                "primary_event_status": primary_event.status if primary_event is not None else None,
                "why_now": why_now,
                "expectations_summary": expectations_summary,
                "valuation_summary": peer_context["summary"],
                "peer_tickers": peer_context["peer_tickers"],
                "peer_stage": peer_context["peer_stage"],
                "primary_indication": expectation_lens["primary_indication"],
                "competitive_summary": expectation_lens["competitive_summary"],
                "differentiation_focus": expectation_lens["differentiation_focus"],
                "state_focus": expectation_lens["state_focus"],
                "market_view": expectation_lens["market_view"],
                "asymmetry_summary": expectation_lens["asymmetry_summary"],
                "asymmetry_label": expectation_lens["asymmetry_label"],
                "peer_gap_pct": expectation_lens.get("peer_gap_pct"),
                "peer_anchor_value": expectation_lens.get("peer_anchor_value"),
                "risk_adjusted_sales": expectation_lens.get("risk_adjusted_sales"),
                "cash_floor_value": expectation_lens.get("cash_floor_value"),
                "value_method": expectation_lens.get("value_method"),
                "internal_value": signal.internal_value,
                "internal_price_target": signal.internal_price_target,
                "internal_upside_pct": signal.internal_upside_pct,
                "floor_support_pct": signal.floor_support_pct,
                "market_leaders": profile["market_leaders"],
                "special_situation": snapshot.metadata.get("special_situation"),
                "special_situation_label": snapshot.metadata.get("special_situation_label"),
                "special_situation_reason": snapshot.metadata.get("special_situation_reason"),
                "driver_label": snapshot.metadata.get("driver_label"),
                "driver_indication": snapshot.metadata.get("driver_indication"),
                "kill_points": kill_points,
            },
        )

    def analyze_universe(
        self,
        universe: Iterable[tuple[str, str]],
        include_literature: bool = False,
        as_of: datetime | None = None,
        prefer_archive: bool = False,
        fallback_to_archive: bool = True,
        persist: bool = True,
        max_archive_age_days: int | None = None,
        allow_stale_archive_fallback: bool = True,
    ) -> list[CompanyAnalysis]:
        results: list[CompanyAnalysis] = []
        for ticker, company_name in universe:
            results.append(
                self.analyze_ticker(
                    ticker,
                    company_name=company_name,
                    include_literature=include_literature,
                    as_of=as_of,
                    prefer_archive=prefer_archive,
                    fallback_to_archive=fallback_to_archive,
                    persist=persist,
                    max_archive_age_days=max_archive_age_days,
                    allow_stale_archive_fallback=allow_stale_archive_fallback,
                )
            )
        return results

    def _load_archived_snapshot(self, ticker: str) -> CompanySnapshot | None:
        payload = self.store.read_latest_raw_payload("snapshots", f"{ticker}_")
        if not isinstance(payload, dict):
            return None
        return snapshot_from_dict(payload)

    @staticmethod
    def _primary_event(snapshot: CompanySnapshot, preferred_event_type: str | None = None):
        if not snapshot.catalyst_events:
            return None
        candidates = snapshot.catalyst_events
        reference_time = TatetuckPlatform._snapshot_reference_time(snapshot)
        upcoming = [item for item in candidates if TatetuckPlatform._is_upcoming_event(item.expected_date, reference_time)]
        if upcoming:
            candidates = upcoming
        state = classify_company_state(snapshot)
        if state in {"commercial_launch", "commercialized"}:
            strategic = [
                item
                for item in candidates
                if event_type_bucket(item.event_type) == "strategic"
                and not is_synthetic_event(item.status, item.title)
                and item.horizon_days <= 180
            ]
            exact_hard = [
                item
                for item in candidates
                if event_type_bucket(item.event_type) in {"clinical", "regulatory"}
                and not is_synthetic_event(item.status, item.title)
                and item.horizon_days <= 180
            ]
            if strategic:
                strategic_with_timing = [
                    item
                    for item in strategic
                    if event_timing_priority(item.status, item.expected_date, item.title) >= 1
                ]
                candidates = strategic_with_timing or strategic
            elif exact_hard:
                candidates = exact_hard
            else:
                near_term = [item for item in candidates if item.horizon_days <= 90]
                if near_term:
                    candidates = near_term
        best_overall = min(
            candidates,
            key=lambda item: (
                -event_timing_priority(item.status, item.expected_date, item.title),
                -event_type_priority(item.event_type),
                item.horizon_days,
                -item.importance,
            ),
        )
        if preferred_event_type:
            typed_candidates = [item for item in snapshot.catalyst_events if item.event_type == preferred_event_type]
            if typed_candidates:
                best_typed = min(
                    typed_candidates,
                    key=lambda item: (
                        -event_timing_priority(item.status, item.expected_date, item.title),
                        -event_type_priority(item.event_type),
                        item.horizon_days,
                        -item.importance,
                    ),
                )
                if event_timing_priority(best_typed.status, best_typed.expected_date, best_typed.title) >= event_timing_priority(
                    best_overall.status,
                    best_overall.expected_date,
                    best_overall.title,
                ):
                    return best_typed
        return best_overall

    @staticmethod
    def _snapshot_reference_time(snapshot: CompanySnapshot) -> datetime:
        ts = pd.to_datetime(snapshot.as_of, errors="coerce", utc=True, format="mixed")
        if pd.isna(ts):
            return datetime.now(timezone.utc)
        return ts.to_pydatetime()

    @staticmethod
    def _is_upcoming_event(expected_date: str | None, reference_time: datetime) -> bool:
        if not expected_date:
            return False
        event_ts = pd.to_datetime(expected_date, errors="coerce", utc=True, format="mixed")
        if pd.isna(event_ts):
            return False
        event_dt = event_ts.to_pydatetime()
        if "T" not in str(expected_date) and len(str(expected_date)) <= 10:
            return event_dt.date() >= reference_time.date()
        return event_dt >= reference_time

    @staticmethod
    def _snapshot_age_days(snapshot: CompanySnapshot, reference_time: datetime | None = None) -> int | None:
        snapshot_ts = pd.to_datetime(snapshot.as_of, errors="coerce", utc=True, format="mixed")
        if pd.isna(snapshot_ts):
            return None
        reference = reference_time or datetime.now(timezone.utc)
        return max((reference.date() - snapshot_ts.to_pydatetime().date()).days, 0)

    def _peer_context(self, snapshot: CompanySnapshot) -> dict[str, object]:
        companies = self.store.read_table("company_snapshots")
        programs = self.store.read_table("programs")
        if companies.empty or programs.empty:
            return {
                "summary": "Peer context unavailable until more archived snapshots are loaded.",
                "peer_tickers": [],
                "peer_stage": "unknown",
            }

        latest_companies = companies.sort_values("as_of").drop_duplicates(subset=["ticker"], keep="last").copy()
        latest_programs = programs.sort_values("as_of").drop_duplicates(subset=["ticker", "program_id"], keep="last").copy()
        latest_programs["phase_rank"] = latest_programs["phase"].map(PHASE_RANK).fillna(0.0)
        top_programs = (
            latest_programs.sort_values(["ticker", "phase_rank", "tam_estimate"], ascending=[True, False, False])
            .drop_duplicates(subset=["ticker"], keep="first")
            [["ticker", "phase", "phase_rank", "tam_estimate", "name"]]
        )
        latest_companies = latest_companies.merge(top_programs, on="ticker", how="left")
        latest_companies = latest_companies[latest_companies["ticker"] != snapshot.ticker].copy()
        if latest_companies.empty:
            return {
                "summary": "Peer context unavailable until more archived snapshots are loaded.",
                "peer_tickers": [],
                "peer_stage": "unknown",
                "valuation_posture": "unknown",
                "current_multiple": None,
                "median_multiple": None,
                "metric_label": "value",
            }

        company_state = classify_company_state(snapshot)
        stage = (
            "commercial"
            if company_state in {"commercial_launch", "commercialized"}
            else "late_stage"
            if any(PHASE_RANK.get(program.phase, 0) >= 4 for program in snapshot.programs)
            else "clinical"
        )
        if stage == "commercial":
            peers = latest_companies[latest_companies["revenue"].fillna(0.0) > 10_000_000].copy()
            current_multiple = snapshot.enterprise_value / max(snapshot.revenue, 1.0)
            if not peers.empty:
                peers["comparison_metric"] = peers["enterprise_value"].fillna(0.0) / peers["revenue"].clip(lower=1.0)
            metric_label = "EV/revenue"
        else:
            peer_threshold = 4 if stage == "late_stage" else 0
            peers = latest_companies[latest_companies["revenue"].fillna(0.0) <= 10_000_000].copy()
            if stage == "late_stage":
                peers = peers[peers["phase_rank"].fillna(0.0) >= peer_threshold]
            else:
                peers = peers[peers["phase_rank"].fillna(0.0) < 4]
            top_tam = max((program.tam_estimate for program in snapshot.programs), default=0.0)
            current_multiple = snapshot.market_cap / max(top_tam, 1.0)
            if not peers.empty:
                peers["comparison_metric"] = peers["market_cap"].fillna(0.0) / peers["tam_estimate"].clip(lower=1.0)
            metric_label = "market-cap/TAM"

        if peers.empty or "comparison_metric" not in peers:
            return {
                "summary": "Peer context is still sparse for this stage bucket.",
                "peer_tickers": [],
                "peer_stage": stage,
                "valuation_posture": "unknown",
                "current_multiple": None,
                "median_multiple": None,
                "metric_label": metric_label,
            }

        peers = peers.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna(subset=["comparison_metric"])
        if peers.empty:
            return {
                "summary": "Peer context is still sparse for this stage bucket.",
                "peer_tickers": [],
                "peer_stage": stage,
                "valuation_posture": "unknown",
                "current_multiple": current_multiple,
                "median_multiple": None,
                "metric_label": metric_label,
            }

        median_multiple = float(peers["comparison_metric"].median())
        percentile = float((peers["comparison_metric"] <= current_multiple).mean() * 100.0)
        peers["distance"] = (peers["comparison_metric"] - current_multiple).abs()
        peer_tickers = peers.nsmallest(3, "distance")["ticker"].astype(str).tolist()

        valuation_posture = "rich" if current_multiple > (median_multiple * 1.15) else "discounted" if current_multiple < (median_multiple * 0.85) else "near peer median"
        summary = (
            f"{stage.replace('_', ' ')} peers trade around {median_multiple:.2f}x {metric_label}; "
            f"{snapshot.ticker} screens at {current_multiple:.2f}x ({valuation_posture}, {percentile:.0f}th percentile)."
        )
        return {
            "summary": summary,
            "peer_tickers": peer_tickers,
            "peer_stage": stage,
            "valuation_posture": valuation_posture,
            "current_multiple": current_multiple,
            "median_multiple": median_multiple,
            "metric_label": metric_label,
        }

    @staticmethod
    def _apply_special_situation_adjustments(snapshot: CompanySnapshot, signal) -> None:
        reason = str(snapshot.metadata.get("special_situation_reason") or "").strip()
        confidence_haircut = float(snapshot.metadata.get("confidence_haircut", 1.0) or 1.0)
        expected_return_haircut = float(snapshot.metadata.get("expected_return_haircut", 1.0) or 1.0)
        catalyst_success_prob_haircut = float(snapshot.metadata.get("catalyst_success_prob_haircut", 1.0) or 1.0)
        if confidence_haircut < 1.0:
            signal.confidence = max(0.0, min(signal.confidence * confidence_haircut, 0.99))
        if expected_return_haircut < 1.0:
            signal.expected_return = float(signal.expected_return) * expected_return_haircut
        if catalyst_success_prob_haircut < 1.0:
            signal.catalyst_success_prob = max(
                0.0,
                min(float(signal.catalyst_success_prob) * catalyst_success_prob_haircut, 0.99),
            )
        if reason:
            note = f"Special situation overlay: {reason}"
            if note not in signal.rationale:
                signal.rationale.append(note)

    @staticmethod
    def _expectations_summary(snapshot: CompanySnapshot, signal, primary_event) -> str:
        momentum = float(snapshot.momentum_3mo or 0.0)
        if momentum >= 0.25:
            price_setup = "hot into the catalyst"
        elif momentum <= -0.10:
            price_setup = "washed out into the catalyst"
        else:
            price_setup = "fairly balanced into the catalyst"

        crowding = "high" if signal.crowding_risk >= 0.65 else "moderate" if signal.crowding_risk >= 0.40 else "low"
        financing = "elevated" if signal.financing_risk >= 0.55 else "manageable"
        event_label = primary_event.event_type if primary_event is not None else "no clear dated catalyst"
        return (
            f"Shares look {price_setup}; crowding risk is {crowding} and financing pressure is {financing} "
            f"heading into {event_label}. Peer-anchored value gap screens at {(signal.internal_upside_pct or 0.0) * 100:+.1f}% "
            f"with a floor support estimate of {(signal.floor_support_pct or 0.0) * 100:.1f}%."
        )

    @staticmethod
    def _kill_points(snapshot: CompanySnapshot, signal, primary_event) -> list[str]:
        kill_points: list[str] = []
        short_thesis = signal.expected_return < 0.0 and (signal.internal_upside_pct or 0.0) < 0.0
        special_situation = str(snapshot.metadata.get("special_situation") or "")
        special_reason = str(snapshot.metadata.get("special_situation_reason") or "")
        if special_situation == "pending_transaction":
            kill_points.append(
                special_reason
                or "Announced transaction caps upside largely to the deal spread instead of standalone execution."
            )
        elif special_situation == "partnered_royalty_transition":
            kill_points.append(
                special_reason
                or "Commercial economics are shifting to partner-led milestones and royalties, which can break simple product-franchise comps."
            )
        elif special_situation == "partner_search_overhang":
            kill_points.append(
                special_reason
                or "Commercialization rights and partner appetite are unresolved, so approval does not automatically translate into clean launch economics."
            )
        elif special_situation == "regulatory_overhang":
            kill_points.append(
                special_reason
                or "An active regulatory dispute can dominate the stock even if routine commercial metrics look stable."
            )
        runway = float(snapshot.metadata.get("runway_months", 0.0) or 0.0)
        if runway and runway < 12.0:
            kill_points.append("Runway is under 12 months, so dilution risk can swamp a good setup.")
        if signal.financing_risk >= 0.55:
            kill_points.append("Financing risk is elevated, which can cap upside or create event overhang.")
        if primary_event is None or not primary_event.expected_date:
            kill_points.append("Catalyst timing is still estimated rather than firmly dated.")
        elif primary_event.status in {"phase_timing_estimate", "calendar_estimate", "guided_company_event"}:
            kill_points.append("Catalyst timing is only partially confirmed and could slip.")
        if signal.crowding_risk >= 0.65 and not short_thesis:
            kill_points.append("Crowding is high enough that a good outcome could still meet sell-the-news pressure.")
        if short_thesis and primary_event is not None:
            kill_points.append("A cleaner-than-feared catalyst or strategic surprise can squeeze a short violently.")
        if short_thesis and signal.company_state in {"commercial_launch", "commercialized"}:
            kill_points.append("Commercial franchises can stay rich longer than fundamentals justify without a clean negative unlock.")
        if snapshot.revenue > 10_000_000 and not snapshot.approved_products and not snapshot.metadata.get("driver_label"):
            kill_points.append("Commercial revenue is present, but the exact product contribution still needs verification.")
        if signal.company_state == "pre_commercial" and signal.setup_type == "asymmetry_without_near_term_catalyst":
            kill_points.append("There is no clean hard catalyst yet, so time and sentiment can dominate before fundamentals close the gap.")
        return kill_points[:3] or ["No single red flag dominates the setup right now."]

    @staticmethod
    def _why_now(snapshot: CompanySnapshot, signal, primary_event, expectations_summary: str, valuation_summary: str, state_focus: str) -> str:
        special_situation = str(snapshot.metadata.get("special_situation") or "")
        special_reason = str(snapshot.metadata.get("special_situation_reason") or "")
        driver_label = str(snapshot.metadata.get("driver_label") or "").strip()
        driver_indication = str(snapshot.metadata.get("driver_indication") or "").strip()
        short_thesis = signal.expected_return < 0.0 and (signal.internal_upside_pct or 0.0) < 0.0
        lead_program = snapshot.programs[0] if snapshot.programs else None
        lead_anchor = (
            driver_label
            if driver_label
            else snapshot.approved_products[0].name
            if snapshot.approved_products
            else (lead_program.name if lead_program else snapshot.ticker)
        )
        lead_descriptor = lead_anchor if not driver_indication else f"{lead_anchor} ({driver_indication})"
        if special_situation == "pending_transaction":
            conviction_line = (
                f"The model sees {signal.expected_return * 100:+.1f}% expected 90-day return with "
                f"{signal.catalyst_success_prob * 100:.0f}% catalyst success probability."
            )
            return f"{state_focus} {special_reason or f'{lead_descriptor} is now a pending transaction rather than an open-ended standalone thesis.'} {conviction_line} {expectations_summary} {valuation_summary}"
        if special_situation == "partner_search_overhang":
            conviction_line = (
                f"The model sees {signal.expected_return * 100:+.1f}% expected 90-day return with "
                f"{signal.catalyst_success_prob * 100:.0f}% catalyst success probability after a special-situation haircut."
            )
            return f"{state_focus} {special_reason or f'{lead_descriptor} still needs a real commercialization partner and cleaner breadth of benefit.'} {conviction_line} {expectations_summary} {valuation_summary}"
        if special_situation == "regulatory_overhang":
            conviction_line = (
                f"The model sees {signal.expected_return * 100:+.1f}% expected 90-day return with "
                f"{signal.catalyst_success_prob * 100:.0f}% catalyst success probability."
            )
            return f"{state_focus} {special_reason or f'{lead_descriptor} now carries an explicit regulatory overhang.'} {conviction_line} {expectations_summary} {valuation_summary}"
        if short_thesis:
            if primary_event is not None:
                catalyst_line = (
                    f"{lead_descriptor} is where the Street may still be too generous, while the current model is focused on "
                    f"{primary_event.event_type} on {primary_event.expected_date if primary_event.expected_date else 'TBD'} as the negative unlock."
                )
            else:
                catalyst_line = (
                    f"{lead_descriptor} still looks richer than the peer-anchored science and commercial reality, even without a clean dated negative unlock."
                )
            conviction_line = (
                f"The model sees {signal.expected_return * 100:+.1f}% expected 90-day return and only "
                f"{signal.catalyst_success_prob * 100:.0f}% catalyst success probability."
            )
            return f"{state_focus} {catalyst_line} {conviction_line} {expectations_summary} {valuation_summary}"
        if signal.setup_type == "asymmetry_without_near_term_catalyst":
            catalyst_line = (
                f"{lead_descriptor} is currently an asymmetry setup rather than a clean dated catalyst trade."
            )
        elif signal.setup_type == "sentiment_floor":
            catalyst_line = (
                f"{lead_descriptor} is currently trading more off sentiment and downside floor than a single hard catalyst."
            )
        elif snapshot.approved_products or driver_label:
            if special_situation == "partnered_royalty_transition":
                catalyst_line = (
                    f"{lead_descriptor} frames the thesis more than a standalone owned-product launch, while the current model is focused on "
                    f"{primary_event.event_type if primary_event is not None else 'an undated catalyst'} "
                    f"on {primary_event.expected_date if primary_event is not None and primary_event.expected_date else 'TBD'}."
                )
            else:
                catalyst_line = (
                    f"{lead_descriptor} anchors the franchise while the current model is focused on "
                    f"{primary_event.event_type if primary_event is not None else 'an undated catalyst'} "
                    f"on {primary_event.expected_date if primary_event is not None and primary_event.expected_date else 'TBD'}."
                )
        else:
            catalyst_line = (
                f"{lead_descriptor} is driving the setup into "
                f"{primary_event.event_type if primary_event is not None else 'an undated catalyst'} "
                f"on {primary_event.expected_date if primary_event is not None and primary_event.expected_date else 'TBD'}."
            )
        conviction_line = (
            f"The model sees {signal.expected_return * 100:+.1f}% expected 90-day return with "
            f"{signal.catalyst_success_prob * 100:.0f}% catalyst success probability."
        )
        return f"{state_focus} {catalyst_line} {conviction_line} {expectations_summary} {valuation_summary}"

    def _aggregate_company_signal(self, snapshot: CompanySnapshot, predictions: list[ModelPrediction], company_state: str | None = None):
        primary_event = self._primary_event(snapshot)
        special_reason = str(snapshot.metadata.get("special_situation_reason") or "")
        driver_label = str(snapshot.metadata.get("driver_label") or "").strip()
        driver_indication = str(snapshot.metadata.get("driver_indication") or "").strip()
        if special_reason:
            commercial_truth = special_reason
        elif driver_label and not snapshot.approved_products:
            commercial_truth = f"Strategic/commercial driver: {driver_label}{f' ({driver_indication})' if driver_indication else ''}."
        elif snapshot.approved_products:
            approved_names = ", ".join(item.name for item in snapshot.approved_products[:2])
            if driver_label and driver_label.lower() not in approved_names.lower():
                commercial_truth = (
                    f"Approved/commercial products: {approved_names}. "
                    f"Current PM driver: {driver_label}{f' ({driver_indication})' if driver_indication else ''}."
                )
            else:
                commercial_truth = f"Approved/commercial products: {approved_names}."
        else:
            commercial_truth = (
                "Commercial revenue is present, but product mapping is not yet verified."
                if snapshot.metadata.get("commercial_revenue_present")
                else "No approved-product revenue base identified."
            )
        rationale = [
            f"{len(snapshot.programs)} active programs normalized into the company-program-catalyst graph.",
            f"{len(snapshot.catalyst_events)} anticipated catalysts inside the 1-6 month event horizon.",
            (
                f"Primary event is {primary_event.event_type} on {primary_event.expected_date}."
                if primary_event is not None
                else "No primary catalyst has been identified."
            ),
            commercial_truth,
            (
                "Runway is capped at 120+ months in reporting."
                if snapshot.metadata.get("runway_months_capped")
                else f"Estimated runway is {float(snapshot.metadata.get('runway_months', 0.0) or 0.0):.1f} months."
            ),
        ]
        signal = aggregate_signal(
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            predictions=predictions,
            evidence_rationale=rationale,
            evidence=snapshot.evidence[:5],
            company_state=company_state,
        )
        if primary_event is not None:
            signal.primary_event_type = primary_event.event_type
            signal.primary_event_bucket = event_type_bucket(primary_event.event_type, primary_event.title)
            signal.primary_event_status = primary_event.status
            signal.primary_event_date = primary_event.expected_date
            signal.primary_event_exact = bool(event_timing_priority(primary_event.status, primary_event.expected_date, primary_event.title) >= 2)
            signal.primary_event_synthetic = is_synthetic_event(primary_event.status, primary_event.title)
            signal.primary_event_source = getattr(primary_event, "source", None)
            if primary_event.horizon_days <= 45:
                signal.thesis_horizon = "30d"
            elif primary_event.horizon_days <= 120:
                signal.thesis_horizon = "90d"
            else:
                signal.thesis_horizon = "180d"
        return signal

    def build_legacy_report(self, ticker: str, company_name: str | None = None) -> dict:
        analysis = self.analyze_ticker(ticker, company_name=company_name, include_literature=True)
        snapshot = analysis.snapshot
        primary_program = snapshot.programs[0] if snapshot.programs else None

        valuation = {
            "tam": primary_program.tam_estimate if primary_program else 1_000_000_000.0,
            "penetration_rate": analysis.signal.catalyst_success_prob,
            "peak_revenue": (primary_program.tam_estimate if primary_program else 1_000_000_000.0) * 0.1,
            "net_revenue_at_peak": (primary_program.tam_estimate if primary_program else 1_000_000_000.0) * 0.06,
            "unadjusted_npv": analysis.signal.expected_return * max(snapshot.market_cap, 1.0),
            "probability_of_success": analysis.signal.catalyst_success_prob,
            "rNPV": analysis.signal.expected_return * max(snapshot.market_cap, 1.0) + snapshot.cash,
            "current_market_cap": snapshot.market_cap,
            "implied_upside_pct": analysis.signal.expected_return * 100.0,
            "signal": analysis.portfolio.scenario,
        }

        heuristic_analysis = {
            "heuristic_pos": primary_program.pos_prior if primary_program else 0.10,
            "phase_label": primary_program.phase if primary_program else "Unknown",
            "disease_multiplier": 1.0,
            "years_remaining": min((event.horizon_days for event in snapshot.catalyst_events), default=180) / 365.0,
            "lead_trial": select_lead_trial(primary_program).title if primary_program and primary_program.trials else "N/A",
            "details": analysis.signal.rationale[0],
        }

        pos_analysis = {
            "probability_of_success": analysis.signal.catalyst_success_prob,
            "estimated_tam": primary_program.tam_estimate if primary_program else 1_000_000_000.0,
            "reasoning": " ".join(analysis.signal.rationale),
        }

        finance_data = {
            "ticker": snapshot.ticker,
            "shortName": snapshot.company_name,
            "marketCap": snapshot.market_cap,
            "enterpriseValue": snapshot.enterprise_value,
            "totalRevenue": snapshot.revenue,
            "cash": snapshot.cash,
            "debt": snapshot.debt,
            "description": snapshot.metadata.get("description"),
        }

        trials_data = [asdict(select_lead_trial(program)) for program in snapshot.programs if select_lead_trial(program) is not None]
        return {
            "company": snapshot.company_name,
            "ticker": snapshot.ticker,
            "finance_data": finance_data,
            "trials_data": trials_data,
            "drug_names": [program.name for program in snapshot.programs],
            "conditions": [condition for program in snapshot.programs for condition in program.conditions],
            "fda_records": 0,
            "heuristic_analysis": heuristic_analysis,
            "literature_review": analysis.literature_review,
            "pos_analysis": pos_analysis,
            "valuation": valuation,
            "signal_artifact": asdict(analysis.signal),
            "portfolio_recommendation": asdict(analysis.portfolio),
            "company_snapshot": asdict(snapshot),
        }
