import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from biopharma_agent.vnext.entities import (
    CatalystEvent,
    CompanyAnalysis,
    CompanySnapshot,
    PortfolioRecommendation,
    SignalArtifact,
)
from discord_bot import (
    brief_event_text,
    brief_pick_profile,
    brief_setup_driver_text,
    build_bot,
    build_dashboard_embed,
    catalyst_alpha_gate,
    collect_fresh_analyses,
    idea_action_label,
    live_first_analysis_kwargs,
    rank_deployable_ideas,
    rank_current_catalyst_events,
    rank_top_idea_analyses,
    rank_top_catalyst_ideas,
    qualifies_top_idea,
    upcoming_event_text,
)


def make_analysis(
    ticker: str,
    *,
    setup_type: str,
    scenario: str,
    event_type: str,
    event_status: str,
    event_source: str,
    event_exact: bool,
    event_horizon_days: int,
    expected_return: float,
    confidence: float,
    catalyst_success_prob: float,
    target_weight: float,
    company_state: str = "pre_commercial",
    expected_date: str = "2026-04-15",
    stance: str = "long",
    internal_upside_pct: float = 0.22,
):
    event = CatalystEvent(
        event_id=f"{ticker}-evt",
        program_id=None,
        event_type=event_type,
        title=f"{ticker} event",
        expected_date=expected_date,
        horizon_days=event_horizon_days,
        probability=0.7,
        importance=0.8,
        crowdedness=0.2,
        status=event_status,
        source=event_source,
        timing_exact=event_exact,
        timing_synthetic=not event_exact,
    )
    snapshot = CompanySnapshot(
        ticker=ticker,
        company_name=ticker,
        as_of="2026-03-29T00:00:00+00:00",
        market_cap=2_000_000_000,
        enterprise_value=1_800_000_000,
        revenue=250_000_000,
        cash=500_000_000,
        debt=100_000_000,
        momentum_3mo=0.1,
        trailing_6mo_return=0.12,
        volatility=0.04,
        programs=[],
        approved_products=[],
        catalyst_events=[event],
        financing_events=[],
        metadata={},
    )
    signal = SignalArtifact(
        ticker=ticker,
        as_of=snapshot.as_of,
        expected_return=expected_return,
        catalyst_success_prob=catalyst_success_prob,
        confidence=confidence,
        crowding_risk=0.2,
        financing_risk=0.2,
        thesis_horizon="90d",
        primary_event_type=event_type,
        primary_event_bucket="clinical" if event_type != "commercial_update" else "commercial",
        primary_event_status=event_status,
        primary_event_date=event.expected_date,
        primary_event_exact=event_exact,
        primary_event_synthetic=not event_exact,
        primary_event_source=event_source,
        company_state=company_state,
        setup_type=setup_type,
        internal_upside_pct=internal_upside_pct,
        floor_support_pct=0.12,
    )
    portfolio = PortfolioRecommendation(
        ticker=ticker,
        as_of=snapshot.as_of,
        stance=stance,
        target_weight=target_weight,
        max_weight=8.0,
        confidence=confidence,
        scenario=scenario,
        thesis_horizon="90d",
        primary_event_type=event_type,
        company_state=company_state,
        setup_type=setup_type,
        risk_flags=[],
    )
    return CompanyAnalysis(
        snapshot=snapshot,
        signal=signal,
        portfolio=portfolio,
        feature_vectors=[],
        program_predictions=[],
        metadata={"setup_type": setup_type, "company_state": company_state},
    )


class TestDiscordBotCatalystRanking(unittest.TestCase):
    @staticmethod
    def catalyst_audit_fixture(pre_pass: bool = True, family_pass: bool = True) -> dict:
        return {
            "sleeve_scorecards": {
                "pre_event_long": {
                    "rows": 132.0,
                    "windows": 24.0,
                    "rank_ic": 0.18,
                    "hit_rate": 0.59,
                    "cost_adjusted_top_bottom_spread": 0.11,
                    "spread_ci_low": 0.02,
                },
                "post_event_reaction_long": {
                    "rows": 108.0,
                    "windows": 22.0,
                    "rank_ic": 0.13,
                    "cost_adjusted_top_bottom_spread": 0.09,
                },
                "event_short_or_pairs": {
                    "rows": 18.0,
                    "windows": 6.0,
                    "rank_ic": -0.02,
                    "cost_adjusted_top_bottom_spread": -0.01,
                },
            },
            "family_scorecards": {
                "phase3_readout": {
                    "rows": 35.0,
                    "windows": 10.0,
                    "rank_ic": 0.12,
                    "cost_adjusted_top_bottom_spread": 0.08,
                },
                "pdufa": {
                    "rows": 29.0,
                    "windows": 9.0,
                    "rank_ic": 0.10,
                    "cost_adjusted_top_bottom_spread": 0.09,
                },
            },
            "gates": {
                "pre_event_long": {"passed": pre_pass, "reason": "pre-event gate"},
                "post_event_reaction_long": {"passed": False, "reason": "post gate"},
                "event_short_or_pairs": {"passed": False, "reason": "short gate"},
                "family_depth": {"passed": family_pass, "qualifying_families": 2.0, "required_families": 2.0},
                "overall_catalyst_bot": {"passed": False},
            },
            "exact_timing_rate": 0.98,
            "exact_outcome_rate": 0.95,
        }

    @staticmethod
    def catalyst_event_master_fixture() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": "VRTX",
                    "event_family": "pdufa",
                    "event_bucket": "regulatory",
                    "expected_date": "2026-04-20T00:00:00+00:00",
                    "last_seen_as_of": "2026-03-29T00:00:00+00:00",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "deployable_pre_event": True,
                    "pre_event_score": 0.92,
                    "confidence_proxy": 0.81,
                    "success_probability_proxy": 0.76,
                    "floor_support_pct": 0.22,
                    "financing_risk_proxy": 0.18,
                    "importance": 0.75,
                    "source": "eodhd_news",
                },
                {
                    "ticker": "REGN",
                    "event_family": "phase3_readout",
                    "event_bucket": "clinical",
                    "expected_date": "2026-05-15T00:00:00+00:00",
                    "last_seen_as_of": "2026-03-29T00:00:00+00:00",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "deployable_pre_event": True,
                    "pre_event_score": 0.88,
                    "confidence_proxy": 0.78,
                    "success_probability_proxy": 0.70,
                    "floor_support_pct": 0.18,
                    "financing_risk_proxy": 0.21,
                    "importance": 0.80,
                    "source": "sec",
                },
                {
                    "ticker": "OLD",
                    "event_family": "phase2_readout",
                    "event_bucket": "clinical",
                    "expected_date": "2026-03-20T00:00:00+00:00",
                    "last_seen_as_of": "2026-03-29T00:00:00+00:00",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "deployable_pre_event": True,
                    "pre_event_score": 0.95,
                    "confidence_proxy": 0.90,
                    "success_probability_proxy": 0.82,
                    "floor_support_pct": 0.15,
                    "financing_risk_proxy": 0.15,
                    "importance": 0.70,
                    "source": "eodhd_news",
                },
            ]
        )

    def test_rank_top_catalyst_ideas_excludes_non_catalyst_setups(self):
        hard = make_analysis(
            "HARD",
            setup_type="hard_catalyst",
            scenario="pre-catalyst long",
            event_type="phase3_readout",
            event_status="exact_sec_filing",
            event_source="sec",
            event_exact=True,
            event_horizon_days=25,
            expected_return=0.18,
            confidence=0.78,
            catalyst_success_prob=0.66,
            target_weight=4.0,
        )
        compounder = make_analysis(
            "CASH",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="exact_sec_filing",
            event_source="sec",
            event_exact=True,
            event_horizon_days=5,
            expected_return=0.22,
            confidence=0.88,
            catalyst_success_prob=0.80,
            target_weight=5.0,
            company_state="commercialized",
        )

        ranked = rank_top_catalyst_ideas([compounder, hard])

        self.assertEqual([analysis.snapshot.ticker for analysis, _event, _score in ranked], ["HARD"])

    def test_rank_top_catalyst_ideas_prefers_exact_hard_catalysts(self):
        hard_exact = make_analysis(
            "EXACT",
            setup_type="hard_catalyst",
            scenario="pre-catalyst long",
            event_type="phase3_readout",
            event_status="exact_press_release",
            event_source="eodhd_news",
            event_exact=True,
            event_horizon_days=30,
            expected_return=0.14,
            confidence=0.74,
            catalyst_success_prob=0.61,
            target_weight=3.5,
        )
        soft_estimated = make_analysis(
            "SOFT",
            setup_type="soft_catalyst",
            scenario="pre-catalyst long",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=12,
            expected_return=0.20,
            confidence=0.82,
            catalyst_success_prob=0.70,
            target_weight=3.0,
        )

        ranked = rank_top_catalyst_ideas(
            [soft_estimated, hard_exact],
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(ranked[0][0].snapshot.ticker, "EXACT")

    def test_rank_top_catalyst_ideas_excludes_past_dated_events_even_if_horizon_is_zero(self):
        stale = make_analysis(
            "STALE",
            setup_type="hard_catalyst",
            scenario="pre-catalyst long",
            event_type="phase3_readout",
            event_status="exact_press_release",
            event_source="eodhd_news",
            event_exact=True,
            event_horizon_days=0,
            expected_return=0.20,
            confidence=0.80,
            catalyst_success_prob=0.72,
            target_weight=4.0,
            expected_date="2026-03-26T17:14:00",
        )
        future = make_analysis(
            "FUTR",
            setup_type="hard_catalyst",
            scenario="pre-catalyst long",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=0,
            expected_return=0.14,
            confidence=0.74,
            catalyst_success_prob=0.61,
            target_weight=3.0,
            expected_date="2026-04-02",
        )

        ranked = rank_top_catalyst_ideas(
            [stale, future],
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual([analysis.snapshot.ticker for analysis, _event, _score in ranked], ["FUTR"])

    def test_rank_top_catalyst_ideas_keeps_same_day_date_only_events(self):
        same_day = make_analysis(
            "TODAY",
            setup_type="soft_catalyst",
            scenario="pre-catalyst long",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=0,
            expected_return=0.11,
            confidence=0.71,
            catalyst_success_prob=0.58,
            target_weight=2.5,
            expected_date="2026-03-29",
        )

        ranked = rank_top_catalyst_ideas(
            [same_day],
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual([analysis.snapshot.ticker for analysis, _event, _score in ranked], ["TODAY"])

    def test_upcoming_event_text_hides_past_events(self):
        stale = make_analysis(
            "STALE",
            setup_type="hard_catalyst",
            scenario="pre-catalyst long",
            event_type="phase3_readout",
            event_status="exact_press_release",
            event_source="eodhd_news",
            event_exact=True,
            event_horizon_days=0,
            expected_return=0.20,
            confidence=0.80,
            catalyst_success_prob=0.72,
            target_weight=4.0,
            expected_date="2026-03-26T17:14:00",
        )

        text = upcoming_event_text(
            stale.snapshot,
            stale.signal.primary_event_type,
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(text, "none upcoming")

    def test_brief_event_text_hides_synthetic_commercial_placeholder(self):
        compounder = make_analysis(
            "CASH",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.22,
            confidence=0.88,
            catalyst_success_prob=0.80,
            target_weight=5.0,
            company_state="commercialized",
        )

        text = brief_event_text(
            compounder.snapshot,
            compounder.signal.primary_event_type,
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(text, "franchise setup, no clean dated catalyst")

    def test_brief_setup_driver_text_prefers_approved_product(self):
        compounder = make_analysis(
            "BMRN",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.22,
            confidence=0.88,
            catalyst_success_prob=0.80,
            target_weight=5.0,
            company_state="commercialized",
        )
        compounder.snapshot.approved_products = [
            SimpleNamespace(name="VOXZOGO", indication="achondroplasia"),
        ]

        text = brief_setup_driver_text(
            compounder.snapshot,
            compounder.signal.primary_event_type,
        )

        self.assertEqual(text, "VOXZOGO | achondroplasia")

    def test_brief_setup_driver_text_prefers_curated_driver_override(self):
        compounder = make_analysis(
            "NVAX",
            setup_type="capital_allocation",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.22,
            confidence=0.88,
            catalyst_success_prob=0.80,
            target_weight=5.0,
            company_state="commercialized",
        )
        compounder.snapshot.approved_products = [
            SimpleNamespace(name="Nuvaxovid", indication="COVID-19"),
        ]
        compounder.snapshot.metadata["driver_label"] = "Matrix-M platform"
        compounder.snapshot.metadata["driver_indication"] = "partnered vaccines / Sanofi royalties"

        text = brief_setup_driver_text(
            compounder.snapshot,
            compounder.signal.primary_event_type,
        )

        self.assertEqual(text, "Matrix-M platform | partnered vaccines / Sanofi royalties")

    def test_brief_event_text_marks_longer_dated_catalyst(self):
        delayed = make_analysis(
            "EDIT",
            setup_type="soft_catalyst",
            scenario="pairs candidate",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=181,
            expected_return=0.20,
            confidence=0.82,
            catalyst_success_prob=0.70,
            target_weight=3.0,
            expected_date="2026-09-30",
        )

        text = brief_event_text(
            delayed.snapshot,
            delayed.signal.primary_event_type,
            now_dt=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(text, "phase2_readout on 2026-09-30 (longer-dated)")

    def test_brief_pick_profile_marks_longer_dated_catalyst(self):
        delayed = make_analysis(
            "EDIT",
            setup_type="soft_catalyst",
            scenario="pairs candidate",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=181,
            expected_return=0.20,
            confidence=0.82,
            catalyst_success_prob=0.70,
            target_weight=3.0,
            expected_date="2026-09-30",
        )

        profile = brief_pick_profile(
            delayed,
            now_dt=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(profile, "extended_catalyst")

    def test_brief_pick_profile_marks_pending_transaction_as_special_situation(self):
        analysis = make_analysis(
            "APLS",
            setup_type="capital_allocation",
            scenario="watchlist only",
            event_type="strategic_transaction",
            event_status="guided_company_event",
            event_source="company_curated",
            event_exact=False,
            event_horizon_days=170,
            expected_return=0.02,
            confidence=0.55,
            catalyst_success_prob=0.85,
            target_weight=0.0,
            company_state="commercialized",
            expected_date="2026-09-22",
        )
        analysis.snapshot.metadata["special_situation"] = "pending_transaction"
        analysis.snapshot.metadata["special_situation_label"] = "pending transaction"

        profile = brief_pick_profile(
            analysis,
            now_dt=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(profile, "special_situation")

    def test_rank_deployable_ideas_prefers_real_dated_events_over_synthetic_commercial_updates(self):
        catalyst = make_analysis(
            "EXACT",
            setup_type="soft_catalyst",
            scenario="pairs candidate",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=30,
            expected_return=0.16,
            confidence=0.76,
            catalyst_success_prob=0.61,
            target_weight=1.8,
        )
        compounder = make_analysis(
            "CASH",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.24,
            confidence=0.90,
            catalyst_success_prob=0.82,
            target_weight=2.5,
            company_state="commercialized",
        )
        deployable = [
            (SimpleNamespace(scaled_target_weight=1.8), catalyst),
            (SimpleNamespace(scaled_target_weight=2.5), compounder),
        ]

        ranked = rank_deployable_ideas(
            deployable,
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(ranked[0][1].snapshot.ticker, "EXACT")

    def test_rank_deployable_ideas_demotes_longer_dated_catalyst_below_franchise_setup(self):
        delayed = make_analysis(
            "EDIT",
            setup_type="soft_catalyst",
            scenario="pairs candidate",
            event_type="phase2_readout",
            event_status="calendar_estimate",
            event_source="estimated_calendar",
            event_exact=False,
            event_horizon_days=181,
            expected_return=0.34,
            confidence=0.90,
            catalyst_success_prob=0.70,
            target_weight=1.8,
            expected_date="2026-09-30",
        )
        compounder = make_analysis(
            "APLS",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.26,
            confidence=0.92,
            catalyst_success_prob=0.80,
            target_weight=2.1,
            company_state="commercialized",
        )
        compounder.snapshot.approved_products = [
            SimpleNamespace(name="Syfovre", indication="geographic atrophy"),
        ]
        deployable = [
            (SimpleNamespace(scaled_target_weight=1.8), delayed),
            (SimpleNamespace(scaled_target_weight=2.1), compounder),
        ]

        ranked = rank_deployable_ideas(
            deployable,
            now_dt=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(ranked[0][1].snapshot.ticker, "APLS")

    def test_idea_action_label_marks_short_recommendations(self):
        analysis = make_analysis(
            "BEAR",
            setup_type="hard_catalyst",
            scenario="pre-catalyst short",
            event_type="phase3_readout",
            event_status="exact_company_calendar",
            event_source="company_curated",
            event_exact=True,
            event_horizon_days=35,
            expected_return=-0.18,
            confidence=0.78,
            catalyst_success_prob=0.31,
            target_weight=2.2,
            stance="short",
            internal_upside_pct=-0.24,
        )

        self.assertEqual(idea_action_label(analysis), "short")

    def test_qualifies_top_idea_accepts_high_conviction_short(self):
        analysis = make_analysis(
            "BEAR",
            setup_type="hard_catalyst",
            scenario="pre-catalyst short",
            event_type="phase3_readout",
            event_status="exact_company_calendar",
            event_source="company_curated",
            event_exact=True,
            event_horizon_days=35,
            expected_return=-0.18,
            confidence=0.78,
            catalyst_success_prob=0.31,
            target_weight=2.2,
            stance="short",
            internal_upside_pct=-0.24,
        )

        self.assertTrue(qualifies_top_idea(analysis))

    def test_rank_top_idea_analyses_can_surface_shorts(self):
        short_idea = make_analysis(
            "BEAR",
            setup_type="hard_catalyst",
            scenario="pre-catalyst short",
            event_type="phase3_readout",
            event_status="exact_company_calendar",
            event_source="company_curated",
            event_exact=True,
            event_horizon_days=35,
            expected_return=-0.24,
            confidence=0.82,
            catalyst_success_prob=0.28,
            target_weight=2.4,
            stance="short",
            internal_upside_pct=-0.30,
        )
        long_idea = make_analysis(
            "BULL",
            setup_type="pipeline_optionality",
            scenario="commercial compounder",
            event_type="commercial_update",
            event_status="estimated_from_revenue",
            event_source="internal_inference",
            event_exact=False,
            event_horizon_days=45,
            expected_return=0.11,
            confidence=0.72,
            catalyst_success_prob=0.70,
            target_weight=1.5,
            company_state="commercialized",
            stance="long",
            internal_upside_pct=0.12,
        )

        ranked = rank_top_idea_analyses(
            [long_idea, short_idea],
            now_dt=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(ranked[0].snapshot.ticker, "BEAR")

    def test_catalyst_alpha_gate_blocks_without_enough_rows(self):
        allowed, scorecard, reason = catalyst_alpha_gate({})

        self.assertFalse(allowed)
        self.assertEqual(reason, "missing_scorecard")
        self.assertEqual(scorecard["pre_event_long"], {})

    def test_catalyst_alpha_gate_blocks_when_pre_event_gate_is_off(self):
        allowed, _scorecard, reason = catalyst_alpha_gate(self.catalyst_audit_fixture(pre_pass=False))

        self.assertFalse(allowed)
        self.assertEqual(reason, "pre_event_gated")

    def test_catalyst_alpha_gate_allows_positive_validated_sleeve(self):
        allowed, scorecard, reason = catalyst_alpha_gate(self.catalyst_audit_fixture())

        self.assertTrue(allowed)
        self.assertIsNone(reason)
        self.assertAlmostEqual(scorecard["pre_event_long"]["cost_adjusted_top_bottom_spread"], 0.11)

    def test_catalyst_alpha_gate_blocks_when_family_depth_gate_is_off(self):
        allowed, _scorecard, reason = catalyst_alpha_gate(self.catalyst_audit_fixture(family_pass=False))

        self.assertFalse(allowed)
        self.assertEqual(reason, "family_depth_gated")

    def test_rank_current_catalyst_events_uses_event_master_and_filters_past_rows(self):
        ranked = rank_current_catalyst_events(
            self.catalyst_event_master_fixture(),
            self.catalyst_audit_fixture(),
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual([row["ticker"] for row in ranked], ["VRTX", "REGN"])

    def test_rank_current_catalyst_events_prefers_latest_event_view_when_present(self):
        event_master = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "event_family": "phase3_readout",
                    "event_bucket": "clinical",
                    "expected_date": "2026-04-05T00:00:00+00:00",
                    "last_seen_as_of": "2026-03-29T00:00:00+00:00",
                    "timing_exact": False,
                    "timing_synthetic": True,
                    "deployable_pre_event": False,
                    "pre_event_score": 0.20,
                    "confidence_proxy": 0.20,
                    "success_probability_proxy": 0.20,
                    "floor_support_pct": 0.05,
                    "financing_risk_proxy": 0.60,
                    "importance": 0.40,
                    "source": "internal_inference",
                    "latest_expected_date": "2026-04-22T00:00:00+00:00",
                    "latest_timing_exact": True,
                    "latest_timing_synthetic": False,
                    "latest_deployable_pre_event": True,
                    "latest_pre_event_score": 0.88,
                    "latest_confidence_proxy": 0.79,
                    "latest_success_probability_proxy": 0.74,
                    "latest_floor_support_pct": 0.21,
                    "latest_financing_risk_proxy": 0.18,
                    "latest_importance": 0.80,
                    "latest_source": "sec",
                }
            ]
        )

        ranked = rank_current_catalyst_events(
            event_master,
            self.catalyst_audit_fixture(),
            now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["ticker"], "TEST")
        self.assertEqual(ranked[0]["source"], "sec")
        self.assertTrue(bool(ranked[0]["timing_exact"]))

    def test_build_dashboard_embed_surfaces_pm_and_gate_status(self):
        readiness = SimpleNamespace(status="production_ready", blockers=[])
        embed = build_dashboard_embed(
            {
                "generated_at": "2026-03-29T15:17:20.606537+00:00",
                "strict_rank_ic": 0.21,
                "exact_primary_event_rate": 0.97,
                "synthetic_primary_event_rate": 0.03,
                "strict_outcome_label_coverage": 0.995,
                "top_bottom_spread_ci_low": 0.01,
                "baseline_scorecards": {
                    "model": {
                        "windows": 35.0,
                        "rows": 2705.0,
                        "rank_ic": 0.25,
                        "hit_rate": 0.64,
                        "cost_adjusted_top_bottom_spread": 0.28,
                    }
                },
                "alpha_sleeve_scorecards": {
                    "pm_alpha": {"cost_adjusted_top_bottom_spread": 0.46, "rank_ic": 0.18, "rows": 197.0},
                    "franchise_alpha": {"cost_adjusted_top_bottom_spread": 0.39, "rank_ic": 0.14},
                    "catalyst_alpha": {"cost_adjusted_top_bottom_spread": 0.0, "rank_ic": 0.0},
                },
                "setup_type_scorecards": {
                    "launch_asymmetry": {"cost_adjusted_top_bottom_spread": 0.64, "rank_ic": 0.35},
                    "pipeline_optionality": {"cost_adjusted_top_bottom_spread": 0.29, "rank_ic": 0.28},
                    "hard_catalyst": {"cost_adjusted_top_bottom_spread": -0.06, "rank_ic": -0.08},
                },
                "stale_snapshot_scorecards": {
                    "fresh_0_7d": {"cost_adjusted_top_bottom_spread": 0.12, "rank_ic": 0.05},
                },
                "regime_scorecards": {
                    "positive_momentum": {"cost_adjusted_top_bottom_spread": 0.13, "rank_ic": 0.15},
                },
                "a_grade_gates": {
                    "a_grade_ready": {"passed": False},
                    "catalyst_surface": {"status": "gated"},
                },
            },
            self.catalyst_audit_fixture(),
            readiness,
        )

        self.assertEqual(embed.title, "PM Alpha Dashboard")
        field_names = [field.name for field in embed.fields]
        self.assertIn("Core Scorecard", field_names)
        self.assertIn("Catalyst Bot", field_names)
        self.assertIn("Alpha Sleeves", field_names)
        self.assertIn("Watch Items", field_names)
        alpha_field = next(field for field in embed.fields if field.name == "Alpha Sleeves")
        self.assertIn("PM:", alpha_field.value)
        self.assertIn("gated", alpha_field.value)

    def test_live_first_analysis_kwargs_disallow_stale_archive_fallback(self):
        kwargs = live_first_analysis_kwargs(include_literature=False, persist=False)

        self.assertFalse(bool(kwargs["prefer_archive"]))
        self.assertTrue(bool(kwargs["fallback_to_archive"]))
        self.assertEqual(kwargs["max_archive_age_days"], 14)
        self.assertFalse(bool(kwargs["allow_stale_archive_fallback"]))
        self.assertFalse(bool(kwargs["persist"]))

    def test_collect_fresh_analyses_uses_live_first_and_skips_failures(self):
        class FakePlatform:
            def __init__(self):
                self.calls = []

            def analyze_ticker(self, ticker, company_name=None, **kwargs):
                self.calls.append((ticker, company_name, kwargs))
                if ticker == "FAIL":
                    raise RuntimeError("offline")
                return make_analysis(
                    ticker,
                    setup_type="hard_catalyst",
                    scenario="pre-catalyst long",
                    event_type="phase3_readout",
                    event_status="exact_sec_filing",
                    event_source="sec",
                    event_exact=True,
                    event_horizon_days=25,
                    expected_return=0.18,
                    confidence=0.78,
                    catalyst_success_prob=0.66,
                    target_weight=4.0,
                )

        platform = FakePlatform()
        analyses = asyncio.run(
            collect_fresh_analyses(
                platform,
                [("HARD", "Hard Bio"), ("FAIL", "Fail Bio"), ("HARD", "Hard Bio Duplicate")],
                include_literature=False,
                persist=False,
                max_concurrency=2,
            )
        )

        self.assertEqual([analysis.snapshot.ticker for analysis in analyses], ["HARD"])
        self.assertEqual(len(platform.calls), 2)
        kwargs_by_ticker = {ticker: kwargs for ticker, _company_name, kwargs in platform.calls}
        self.assertFalse(bool(kwargs_by_ticker["HARD"]["prefer_archive"]))
        self.assertEqual(kwargs_by_ticker["HARD"]["max_archive_age_days"], 14)
        self.assertFalse(bool(kwargs_by_ticker["HARD"]["allow_stale_archive_fallback"]))

    def test_build_bot_registers_expected_commands(self):
        fake_platform = SimpleNamespace(analyze_ticker=lambda *args, **kwargs: None)
        fake_settings = SimpleNamespace()
        fake_store = SimpleNamespace()

        with patch("discord_bot.get_discord_token", return_value="test-token"), patch(
            "discord_bot.build_platform",
            return_value=(fake_platform, fake_settings, fake_store),
        ), patch("discord_bot.UniverseResolver"), patch("discord_bot.AlpacaPaperBroker"), patch("discord_bot.PMExecutionPlanner"):
            bot = build_bot()
            command_names = {command.name for command in bot.commands}

        self.assertIn("analyze", command_names)
        self.assertIn("top5", command_names)
        self.assertIn("guide", command_names)
        self.assertIn("channelid", command_names)
        self.assertIn("setup", command_names)
        self.assertIn("status", command_names)
        self.assertIn("help", command_names)
