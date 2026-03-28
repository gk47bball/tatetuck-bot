from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Iterable

from ..agents.pubmed_agent import AutoResearchAgent
from .entities import CompanyAnalysis, CompanySnapshot, ModelPrediction
from .features import FeatureEngineer
from .models import EventDrivenEnsemble
from .portfolio import PortfolioConstructor, aggregate_signal
from .replay import snapshot_from_dict
from .sources import IngestionService
from .storage import LocalResearchStore


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
    ) -> CompanyAnalysis:
        snapshot: CompanySnapshot | None = None
        analysis_source = "live"

        if prefer_archive:
            snapshot = self._load_archived_snapshot(ticker)
            if snapshot is not None:
                analysis_source = "archive"

        if snapshot is None:
            try:
                snapshot = self.ingestion.ingest_company(ticker, company_name, as_of=as_of)
            except Exception as exc:
                if not fallback_to_archive:
                    raise
                snapshot = self._load_archived_snapshot(ticker)
                if snapshot is None:
                    raise
                snapshot.metadata["live_ingestion_error"] = f"{type(exc).__name__}: {exc}"
                analysis_source = "archive_fallback"
            else:
                analysis_source = "live"

        self.store.write_snapshot(snapshot)
        feature_vectors = self.features.build_all(snapshot)
        self.store.write_feature_vectors(feature_vectors)

        program_vectors = [vector for vector in feature_vectors if not vector.metadata.get("aggregate")]
        program_predictions = self.ensemble.score(program_vectors)
        signal = self._aggregate_company_signal(snapshot, program_predictions)
        portfolio_rec = self.portfolio.recommend(signal)
        self.store.write_signal_artifacts([signal])
        self.store.write_portfolio_recommendations([portfolio_rec])
        literature_review = ""
        if include_literature:
            drug_names = [program.name for program in snapshot.programs[:3]]
            conditions = [condition for program in snapshot.programs[:2] for condition in program.conditions[:2]]
            literature_review = self.literature.generate_literature_review(snapshot.company_name, drug_names, conditions)

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
            },
        )

    def analyze_universe(
        self,
        universe: Iterable[tuple[str, str]],
        include_literature: bool = False,
        as_of: datetime | None = None,
        prefer_archive: bool = False,
        fallback_to_archive: bool = True,
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
                )
            )
        return results

    def _load_archived_snapshot(self, ticker: str) -> CompanySnapshot | None:
        payload = self.store.read_latest_raw_payload("snapshots", f"{ticker}_")
        if not isinstance(payload, dict):
            return None
        return snapshot_from_dict(payload)

    def _aggregate_company_signal(self, snapshot: CompanySnapshot, predictions: list[ModelPrediction]):
        rationale = [
            f"{len(snapshot.programs)} active programs normalized into the company-program-catalyst graph.",
            f"{len(snapshot.catalyst_events)} anticipated catalysts inside the 1-6 month event horizon.",
            f"Estimated runway is {float(snapshot.metadata.get('runway_months', 0.0) or 0.0):.1f} months.",
        ]
        return aggregate_signal(
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            predictions=predictions,
            evidence_rationale=rationale,
            evidence=snapshot.evidence[:5],
        )

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
            "lead_trial": primary_program.trials[0].title if primary_program and primary_program.trials else "N/A",
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

        trials_data = [asdict(program.trials[0]) for program in snapshot.programs if program.trials]
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
