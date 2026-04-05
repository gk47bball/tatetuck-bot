import os
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from biopharma_agent.vnext import TatetuckPlatform, archive_universe
from biopharma_agent.vnext.entities import CatalystEvent, ModelPrediction, SignalArtifact
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.features import FeatureEngineer
from biopharma_agent.vnext.graph import build_company_snapshot, select_lead_trial
from biopharma_agent.vnext.market_profile import build_expectation_lens, classify_company_state, primary_indication
from biopharma_agent.vnext.portfolio import PortfolioConstructor, aggregate_signal
from biopharma_agent.vnext.sources import IngestionService, enrich_snapshot_with_external_data
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.universe import UniverseResolver


def make_raw_company() -> dict:
    return {
        "ticker": "TEST",
        "company_name": "Test Therapeutics",
        "finance": {
            "marketCap": 1_500_000_000,
            "enterpriseValue": 1_250_000_000,
            "totalRevenue": 80_000_000,
            "cash": 500_000_000,
            "debt": 50_000_000,
            "momentum_3mo": 0.22,
            "trailing_6mo_return": 0.10,
            "volatility": 0.04,
            "description": "Precision oncology and gene editing company with a late-stage RCC program.",
        },
        "trials": [
            {
                "nct_id": "NCT-1",
                "title": "Lead RCC trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 420,
            },
            {
                "nct_id": "NCT-2",
                "title": "Expansion cohort",
                "overall_status": "ACTIVE_NOT_RECRUITING",
                "phase": ["Phase 2"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["response rate"],
                "enrollment": 115,
            },
        ],
        "num_trials": 2,
        "best_phase": "PHASE3",
        "num_papers": 6,
        "pubmed_papers": [
            {"pmid": "1", "title": "TTX-101 data", "abstract": "Strong oncology signal."},
        ],
    }


class TestVNextPlatform(unittest.TestCase):
    def test_graph_builder_filters_observational_programs_and_maps_phase4_to_approved(self):
        raw = make_raw_company()
        raw["trials"] = [
            {
                "nct_id": "NCT-LOW",
                "title": "Welcome to PrEP School Utilizing Peer Educators to Improve Uptake",
                "overall_status": "RECRUITING",
                "phase": [],
                "conditions": ["HIV Prevention Program"],
                "interventions": ["PrEP Awareness and Uptake Educational Program"],
                "primary_outcomes": ["PrEP uptake"],
                "enrollment": 120,
            },
            {
                "nct_id": "NCT-P4",
                "title": "A Study to Evaluate Avacopan in Participants With ANCA-associated Vasculitis",
                "overall_status": "RECRUITING",
                "phase": ["PHASE4"],
                "conditions": ["Antineutrophil Cytoplasmic Antibody-associated Vasculitis"],
                "interventions": ["Avacopan", "Placebo", "Standard of Care"],
                "primary_outcomes": ["Treatment-emergent adverse events"],
                "enrollment": 300,
            },
        ]

        snapshot = build_company_snapshot(raw)

        self.assertEqual(len(snapshot.programs), 1)
        self.assertEqual(snapshot.programs[0].name, "Avacopan")
        self.assertEqual(snapshot.programs[0].phase, "APPROVED")
        self.assertEqual(snapshot.programs[0].catalyst_events[0].event_type, "commercial_update")

    def test_graph_builder_filters_irrelevant_company_evidence(self):
        raw = make_raw_company()
        raw["ticker"] = "BPMC"
        raw["company_name"] = "Blueprint Medicines"
        raw["trials"] = [
            {
                "nct_id": "NCT-BLU",
                "title": "Study of BLU-808 in Chronic Spontaneous Urticaria",
                "overall_status": "RECRUITING",
                "phase": ["PHASE2"],
                "conditions": ["Chronic Spontaneous Urticaria"],
                "interventions": ["BLU-808"],
                "primary_outcomes": ["Urticaria activity score"],
                "enrollment": 180,
            }
        ]
        raw["pubmed_papers"] = [
            {
                "pmid": "11",
                "title": "A randomized controlled clinical trial of intranasal versus subcutaneous midazolam for agitation in terminal illness.",
                "abstract": "Terminal illness sedation protocol.",
            },
            {
                "pmid": "12",
                "title": "Effects of different doses of alfentanil on tracheal intubation stress responses.",
                "abstract": "Ambulatory surgery anesthesia comparison.",
            },
        ]

        snapshot = build_company_snapshot(raw)

        self.assertEqual(snapshot.evidence, [])

    def test_graph_builder_rejects_competitor_pubmed_and_adds_special_situation_for_arvn(self):
        raw = make_raw_company()
        raw["ticker"] = "ARVN"
        raw["company_name"] = "Arvinas"
        raw["finance"]["totalRevenue"] = 262_600_000
        raw["trials"] = [
            {
                "nct_id": "NCT-ARV-471",
                "title": "A Study to Learn About a New Medicine Called Vepdegestrant (ARV-471, PF-07850327)",
                "overall_status": "ACTIVE_NOT_RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Advanced Breast Cancer"],
                "interventions": ["ARV-471"],
                "primary_outcomes": ["progression-free survival"],
                "enrollment": 420,
            }
        ]
        raw["pubmed_papers"] = [
            {
                "pmid": "21",
                "title": "Bireociclib Plus Fulvestrant in Advanced Breast Cancer After Endocrine Progression",
                "abstract": "Competitor CDK4/6 regimen in breast cancer.",
            },
            {
                "pmid": "22",
                "title": "Imlunestrant plus abemaciclib versus fulvestrant plus abemaciclib in ER-positive advanced breast cancer",
                "abstract": "Indirect treatment comparison across phase III trials.",
            },
        ]

        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 2, tzinfo=timezone.utc))
        evidence_titles = [item.title for item in snapshot.evidence]

        self.assertFalse(any("Bireociclib" in title for title in evidence_titles))
        self.assertFalse(any("Imlunestrant" in title for title in evidence_titles))
        self.assertEqual(snapshot.metadata["special_situation"], "partner_search_overhang")
        self.assertTrue(snapshot.metadata["bear_case_flags"])

    def test_graph_builder_applies_curated_program_overrides_and_exclusions(self):
        raw = make_raw_company()
        raw["ticker"] = "GILD"
        raw["company_name"] = "Gilead Sciences"
        raw["finance"]["totalRevenue"] = 28_000_000_000
        raw["trials"] = [
            {
                "nct_id": "NCT-GILD-1",
                "title": "Immediate Initiation of Antiretroviral Therapy During Hyperacute HIV Infection",
                "overall_status": "RECRUITING",
                "phase": ["Phase 1"],
                "conditions": ["HIV"],
                "interventions": ["Dolutegravir"],
                "primary_outcomes": ["safety"],
                "enrollment": 80,
            },
            {
                "nct_id": "NCT-GILD-2",
                "title": "Welcome to PrEP School Utilizing Peer Educators",
                "overall_status": "RECRUITING",
                "phase": [],
                "conditions": ["HIV Prevention Program"],
                "interventions": ["PrEP Awareness and Uptake Educational Program"],
                "primary_outcomes": ["uptake"],
                "enrollment": 120,
            },
            {
                "nct_id": "NCT-GILD-3",
                "title": "Study of Bictegravir/Lenacapavir in Children and Adolescents With HIV-1",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["HIV-1 infection"],
                "interventions": ["Lenacapavir"],
                "primary_outcomes": ["virologic suppression"],
                "enrollment": 300,
            },
        ]

        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 2, tzinfo=timezone.utc))
        program_names = [program.name for program in snapshot.programs]

        self.assertIn("Lenacapavir", program_names)
        self.assertNotIn("Dolutegravir", program_names)
        self.assertFalse(any("PrEP Awareness" in name for name in program_names))

    def test_store_reads_latest_raw_payload_by_as_of_not_file_mtime(self):
        snapshot = build_company_snapshot(make_raw_company())
        older = asdict(snapshot)
        older["as_of"] = "2024-08-08T00:00:00+00:00"
        newer = asdict(snapshot)
        newer["as_of"] = "2026-03-28T16:45:12.853369+00:00"
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            older_path = store.write_raw_payload("snapshots", "TEST_2024-08-08T00-00-00+00-00", older)
            newer_path = store.write_raw_payload("snapshots", "TEST_2026-03-28T16-45-12.853369+00-00", newer)
            os.utime(older_path, (2_000_000_000, 2_000_000_000))
            os.utime(newer_path, (1_000_000_000, 1_000_000_000))

            payload = store.read_latest_raw_payload("snapshots", "TEST_")

            self.assertEqual(payload["as_of"], "2026-03-28T16:45:12.853369+00:00")

    def test_graph_builder_creates_programs_and_financing_events(self):
        raw = make_raw_company()
        snapshot = build_company_snapshot(raw)
        self.assertEqual(snapshot.ticker, "TEST")
        self.assertTrue(snapshot.programs)
        self.assertEqual(snapshot.programs[0].phase, "PHASE3")
        self.assertTrue(snapshot.catalyst_events)
        self.assertFalse(snapshot.approved_products)
        self.assertTrue(snapshot.metadata["commercial_revenue_present"])
        self.assertTrue(any(event.event_type == "commercial_update" for event in snapshot.catalyst_events))

    def test_graph_builder_uses_registry_for_known_products(self):
        raw = make_raw_company()
        raw["ticker"] = "CRSP"
        raw["company_name"] = "CRISPR Therapeutics"
        raw["finance"]["totalRevenue"] = 3_500_000
        snapshot = build_company_snapshot(raw)
        self.assertTrue(snapshot.approved_products)
        self.assertEqual(snapshot.approved_products[0].name, "CASGEVY")

    def test_primary_indication_prefers_lead_approved_product_for_commercial_names(self):
        raw = make_raw_company()
        raw["ticker"] = "BMRN"
        raw["company_name"] = "BioMarin Pharmaceutical"
        raw["finance"]["totalRevenue"] = 3_200_000_000
        snapshot = build_company_snapshot(raw)

        self.assertEqual(primary_indication(snapshot), "achondroplasia")

    def test_primary_indication_uses_curated_override_when_needed(self):
        raw = make_raw_company()
        raw["ticker"] = "PTCT"
        raw["company_name"] = "PTC Therapeutics"
        raw["finance"]["totalRevenue"] = 1_700_000_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 2, tzinfo=timezone.utc))

        self.assertEqual(primary_indication(snapshot), "Phenylketonuria")

    def test_market_profile_classifies_company_states(self):
        pre = build_company_snapshot(make_raw_company() | {"finance": {**make_raw_company()["finance"], "totalRevenue": 0.0}})
        launch_raw = make_raw_company()
        launch_raw["ticker"] = "CRSP"
        launch_raw["company_name"] = "CRISPR Therapeutics"
        launch_raw["finance"]["totalRevenue"] = 50_000_000
        launch = build_company_snapshot(launch_raw)
        mature_raw = make_raw_company()
        mature_raw["ticker"] = "MRNA"
        mature_raw["company_name"] = "Moderna"
        mature_raw["finance"]["totalRevenue"] = 1_500_000_000
        mature = build_company_snapshot(mature_raw)

        self.assertEqual(classify_company_state(pre), "pre_commercial")
        self.assertEqual(classify_company_state(launch), "commercial_launch")
        self.assertEqual(classify_company_state(mature), "commercialized")

    def test_market_profile_honors_company_state_override(self):
        raw = make_raw_company()
        raw["ticker"] = "ARVN"
        raw["company_name"] = "Arvinas"
        raw["finance"]["totalRevenue"] = 262_600_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 2, tzinfo=timezone.utc))

        self.assertEqual(classify_company_state(snapshot), "pre_commercial")

    def test_feature_engineer_outputs_non_leaky_vectors(self):
        snapshot = build_company_snapshot(make_raw_company())
        snapshot.metadata.update(
            {
                "sec_revenue_ttm": 120_000_000,
                "sec_operating_cashflow": 20_000_000,
                "last_10q_date": "2025-01-15",
                "recent_offering_signal": 1.0,
            }
        )
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)
        self.assertTrue(vectors)
        program_vector = vectors[0]
        self.assertNotIn("market_flow_return_6mo_legacy", program_vector.feature_family)
        self.assertIn("catalyst_timing_probability", program_vector.feature_family)
        self.assertIn("commercial_execution_sec_revenue_scale", program_vector.feature_family)
        self.assertIn("balance_sheet_recent_offering_signal", program_vector.feature_family)
        self.assertGreaterEqual(program_vector.feature_family["catalyst_timing_filing_freshness_days"], 0.0)

    def test_graph_builder_prioritizes_decision_relevant_trial_over_larger_phase2_study(self):
        raw = make_raw_company()
        raw["trials"] = [
            {
                "nct_id": "NCT-LARGE-P2",
                "title": "Large expansion cohort",
                "overall_status": "RECRUITING",
                "phase": ["Phase 2"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["objective response rate"],
                "enrollment": 650,
            },
            {
                "nct_id": "NCT-PIVOTAL-P3",
                "title": "Pivotal RCC trial",
                "overall_status": "ACTIVE_NOT_RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 320,
            },
        ]
        snapshot = build_company_snapshot(raw)
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)

        self.assertEqual(snapshot.programs[0].trials[0].trial_id, "NCT-PIVOTAL-P3")
        self.assertGreater(vectors[0].feature_family["program_quality_endpoint_score"], 0.80)

    def test_graph_builder_attaches_program_specific_evidence(self):
        raw = make_raw_company()
        raw["trials"] = [
            {
                "nct_id": "NCT-RCC",
                "title": "Lead RCC trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 420,
            },
            {
                "nct_id": "NCT-MEL",
                "title": "Melanoma expansion trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 2"],
                "conditions": ["Metastatic Melanoma"],
                "interventions": ["ABC-201"],
                "primary_outcomes": ["objective response rate"],
                "enrollment": 180,
            },
        ]
        raw["pubmed_papers"] = [
            {
                "pmid": "11",
                "title": "TTX-101 demonstrates survival benefit in RCC",
                "abstract": "Clear cell renal cell carcinoma data for TTX-101 were encouraging.",
            },
            {
                "pmid": "22",
                "title": "ABC-201 activity in metastatic melanoma",
                "abstract": "Melanoma patients treated with ABC-201 showed durable responses.",
            },
        ]

        snapshot = build_company_snapshot(raw)

        evidence_by_program = {program.name: [item.title for item in program.evidence] for program in snapshot.programs}
        self.assertIn("TTX-101 demonstrates survival benefit in RCC", evidence_by_program["TTX-101"])
        self.assertNotIn("ABC-201 activity in metastatic melanoma", evidence_by_program["TTX-101"])
        self.assertIn("ABC-201 activity in metastatic melanoma", evidence_by_program["ABC-201"])

    def test_external_enrichment_adds_sec_evidence_calendar_and_financing(self):
        snapshot = build_company_snapshot(
            make_raw_company(),
            as_of=datetime(2025, 2, 1, tzinfo=timezone.utc),
        )
        sec_payload = {
            "cik": "0000000123",
            "parsed": {
                "revenue_ttm": 150_000_000,
                "cash_latest": 600_000_000,
                "operating_cashflow": 30_000_000,
                "last_10q_date": "2025-02-01",
                "recent_filings": [
                    {
                        "form": "10-Q",
                        "filing_date": "2025-02-01",
                        "acceptance_datetime": "2025-02-01T21:05:00Z",
                        "accession_number": "0001",
                        "primary_document": "q1.htm",
                        "primary_doc_description": "Quarterly financial results and business update",
                        "url": "https://example.com/10q",
                    }
                ],
                "recent_offering_forms": [
                    {
                        "form": "S-3",
                        "filing_date": "2025-02-15",
                    }
                ],
            },
        }
        calendar_payload = {
            "events": [
                {
                    "event_type": "earnings",
                    "title": "TEST estimated quarterly update",
                    "expected_date": "2025-05-01",
                }
            ]
        }

        enriched = enrich_snapshot_with_external_data(snapshot, sec_payload, calendar_payload)
        self.assertEqual(enriched.metadata["sec_cik"], "0000000123")
        self.assertEqual(enriched.metadata["recent_offering_signal"], 1.0)
        self.assertTrue(any(item.source == "sec" for item in enriched.evidence))
        self.assertTrue(any(event.event_type == "earnings" for event in enriched.catalyst_events))
        self.assertTrue(any(event.status == "exact_sec_filing" for event in enriched.catalyst_events))
        self.assertTrue(any(event.event_type == "recent_offering_filing" for event in enriched.financing_events))

    def test_store_persists_company_level_catalysts(self):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_snapshot(snapshot)
            catalysts = store.read_table("catalysts")
            event_tape = store.read_table("event_tape")
            membership = store.read_table("universe_membership")
            self.assertIn("phase3_readout", catalysts["event_type"].tolist())
            self.assertIn("commercial_update", catalysts["event_type"].tolist())
            self.assertFalse(event_tape.empty)
            self.assertEqual(membership.iloc[0]["ticker"], "TEST")

    def test_portfolio_constructor_flags_financing_overhang(self):
        snapshot = build_company_snapshot(make_raw_company())
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)
        prediction_rows = []
        for idx, vector in enumerate(vectors[:-1]):
            prediction_rows.append(
                {
                    "entity_id": vector.entity_id,
                    "ticker": vector.ticker,
                    "as_of": vector.as_of,
                    "expected_return": 0.20 if idx == 0 else 0.08,
                    "catalyst_success_prob": 0.70,
                    "confidence": 0.75,
                    "crowding_risk": 0.35,
                    "financing_risk": 0.85,
                    "thesis_horizon": vector.thesis_horizon,
                    "model_name": "test",
                    "model_version": "v1",
                    "metadata": vector.metadata,
                }
            )
        from biopharma_agent.vnext.entities import ModelPrediction

        predictions = [ModelPrediction(**row) for row in prediction_rows]
        signal = aggregate_signal(snapshot.ticker, snapshot.as_of, predictions, ["test"], snapshot.evidence)
        recommendation = PortfolioConstructor().recommend(signal)
        self.assertEqual(recommendation.scenario, "avoid due to financing")
        self.assertEqual(recommendation.target_weight, 0.0)

    def test_portfolio_constructor_expands_high_conviction_weights(self):
        constructor = PortfolioConstructor()
        high_signal = SignalArtifact(
            ticker="HIGH",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.24,
            catalyst_success_prob=0.72,
            confidence=0.84,
            crowding_risk=0.22,
            financing_risk=0.18,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        medium_signal = SignalArtifact(
            ticker="MED",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.09,
            catalyst_success_prob=0.55,
            confidence=0.63,
            crowding_risk=0.35,
            financing_risk=0.30,
            thesis_horizon="90d",
            primary_event_type="phase2_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )

        high_rec = constructor.recommend(high_signal)
        medium_rec = constructor.recommend(medium_signal)

        self.assertEqual(high_rec.scenario, "pre-catalyst long")
        self.assertGreater(high_rec.target_weight, medium_rec.target_weight)
        self.assertGreaterEqual(high_rec.target_weight, 3.0)

    def test_portfolio_constructor_treats_near_term_exact_event_as_pre_catalyst_long(self):
        constructor = PortfolioConstructor()
        signal = SignalArtifact(
            ticker="ARVN",
            as_of="2026-04-02T00:00:00+00:00",
            expected_return=0.24,
            catalyst_success_prob=0.68,
            confidence=0.82,
            crowding_risk=0.28,
            financing_risk=0.22,
            thesis_horizon="180d",
            primary_event_type="pdufa",
            primary_event_bucket="regulatory",
            primary_event_status="exact_company_calendar",
            primary_event_date="2026-06-05",
            primary_event_exact=True,
            company_state="pre_commercial",
            setup_type="hard_catalyst",
            rationale=[],
            supporting_evidence=[],
        )

        recommendation = constructor.recommend(signal)

        self.assertEqual(recommendation.scenario, "pre-catalyst long")

    def test_portfolio_constructor_identifies_pre_catalyst_short(self):
        constructor = PortfolioConstructor()
        signal = SignalArtifact(
            ticker="SHORT",
            as_of="2026-04-02T00:00:00+00:00",
            expected_return=-0.19,
            catalyst_success_prob=0.34,
            confidence=0.79,
            crowding_risk=0.28,
            financing_risk=0.24,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            primary_event_status="exact_company_calendar",
            primary_event_date="2026-05-20",
            primary_event_exact=True,
            company_state="pre_commercial",
            setup_type="hard_catalyst",
            internal_upside_pct=-0.22,
            floor_support_pct=0.06,
            rationale=[],
            supporting_evidence=[],
        )

        recommendation = constructor.recommend(signal)

        self.assertEqual(recommendation.stance, "short")
        self.assertEqual(recommendation.scenario, "pre-catalyst short")
        self.assertGreater(recommendation.target_weight, 0.0)

    def test_aggregate_signal_prefers_nearer_prediction_horizon_over_longer_majority(self):
        predictions = [
            ModelPrediction(
                entity_id="ARVN:1",
                ticker="ARVN",
                as_of="2026-04-02T00:00:00+00:00",
                expected_return=0.32,
                catalyst_success_prob=0.72,
                confidence=0.80,
                crowding_risk=0.25,
                financing_risk=0.22,
                thesis_horizon="90d",
                model_name="test",
                model_version="v1",
                metadata={
                    "event_type": "phase3_readout",
                    "event_status": "phase_timing_estimate",
                    "event_expected_date": "2026-08-01",
                },
            ),
            ModelPrediction(
                entity_id="ARVN:2",
                ticker="ARVN",
                as_of="2026-04-02T00:00:00+00:00",
                expected_return=0.20,
                catalyst_success_prob=0.62,
                confidence=0.72,
                crowding_risk=0.28,
                financing_risk=0.24,
                thesis_horizon="180d",
                model_name="test",
                model_version="v1",
                metadata={
                    "event_type": "phase2_readout",
                    "event_status": "phase_timing_estimate",
                    "event_expected_date": "2026-09-30",
                },
            ),
            ModelPrediction(
                entity_id="ARVN:3",
                ticker="ARVN",
                as_of="2026-04-02T00:00:00+00:00",
                expected_return=0.16,
                catalyst_success_prob=0.58,
                confidence=0.66,
                crowding_risk=0.30,
                financing_risk=0.26,
                thesis_horizon="180d",
                model_name="test",
                model_version="v1",
                metadata={
                    "event_type": "phase1_readout",
                    "event_status": "phase_timing_estimate",
                    "event_expected_date": "2026-12-29",
                },
            ),
        ]

        signal = aggregate_signal("ARVN", "2026-04-02T00:00:00+00:00", predictions, ["test"], [])

        self.assertEqual(signal.thesis_horizon, "90d")

    def test_portfolio_constructor_uses_empirical_priors_to_favor_stronger_archetypes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "company_state_scorecards": {
                        "pre_commercial": {
                            "rows": 48.0,
                            "windows": 8.0,
                            "rank_ic": 0.16,
                            "hit_rate": 0.66,
                            "top_bottom_spread": 0.10,
                            "beta_adjusted_return": 0.06,
                        },
                        "commercialized": {
                            "rows": 52.0,
                            "windows": 8.0,
                            "rank_ic": 0.01,
                            "hit_rate": 0.49,
                            "top_bottom_spread": -0.01,
                            "beta_adjusted_return": 0.00,
                        },
                    },
                    "setup_type_scorecards": {
                        "hard_catalyst": {
                            "rows": 44.0,
                            "windows": 8.0,
                            "rank_ic": 0.18,
                            "hit_rate": 0.67,
                            "top_bottom_spread": 0.12,
                            "beta_adjusted_return": 0.08,
                        },
                        "capital_allocation": {
                            "rows": 40.0,
                            "windows": 8.0,
                            "rank_ic": -0.04,
                            "hit_rate": 0.46,
                            "top_bottom_spread": -0.04,
                            "beta_adjusted_return": -0.02,
                        },
                    },
                    "state_setup_scorecards": {
                        "pre_commercial|hard_catalyst": {
                            "rows": 32.0,
                            "windows": 8.0,
                            "rank_ic": 0.19,
                            "hit_rate": 0.69,
                            "top_bottom_spread": 0.14,
                            "beta_adjusted_return": 0.09,
                        },
                        "commercialized|capital_allocation": {
                            "rows": 34.0,
                            "windows": 8.0,
                            "rank_ic": -0.05,
                            "hit_rate": 0.45,
                            "top_bottom_spread": -0.05,
                            "beta_adjusted_return": -0.02,
                        },
                    },
                },
            )
            constructor = PortfolioConstructor(store=store)
            catalyst_signal = SignalArtifact(
                ticker="CATA",
                as_of="2025-01-01T00:00:00+00:00",
                expected_return=0.15,
                catalyst_success_prob=0.55,
                confidence=0.76,
                crowding_risk=0.25,
                financing_risk=0.20,
                thesis_horizon="90d",
                primary_event_type="phase3_readout",
                primary_event_bucket="clinical",
                company_state="pre_commercial",
                setup_type="hard_catalyst",
                rationale=[],
                supporting_evidence=[],
            )
            compounder_signal = SignalArtifact(
                ticker="FRAN",
                as_of="2025-01-01T00:00:00+00:00",
                expected_return=0.15,
                catalyst_success_prob=0.56,
                confidence=0.76,
                crowding_risk=0.25,
                financing_risk=0.20,
                thesis_horizon="90d",
                primary_event_type="capital_allocation",
                primary_event_bucket="commercial",
                company_state="commercialized",
                setup_type="capital_allocation",
                rationale=[],
                supporting_evidence=[],
            )

            catalyst_rec = constructor.recommend(catalyst_signal)
            compounder_rec = constructor.recommend(compounder_signal)

            self.assertEqual(catalyst_rec.scenario, "pre-catalyst long")
            self.assertGreater(catalyst_rec.target_weight, compounder_rec.target_weight)
            self.assertIn("strong historical archetype edge", catalyst_rec.risk_flags)
            self.assertIn("weak historical archetype edge", compounder_rec.risk_flags)

    def test_portfolio_constructor_ignores_stale_validation_payload_and_derives_priors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": "2024-01-01T00:00:00+00:00",
                    "setup_type_scorecards": {
                        "hard_catalyst": {
                            "rows": 40.0,
                            "windows": 8.0,
                            "rank_ic": -0.40,
                            "hit_rate": 0.35,
                            "top_bottom_spread": -0.12,
                            "beta_adjusted_return": -0.08,
                        }
                    },
                },
            )
            store.replace_table(
                "signal_artifacts",
                [
                    {
                        "ticker": "A1",
                        "as_of": "2026-01-01T00:00:00+00:00",
                        "expected_return": 0.08,
                        "catalyst_success_prob": 0.55,
                        "confidence": 0.65,
                        "crowding_risk": 0.20,
                        "financing_risk": 0.15,
                        "thesis_horizon": "90d",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                    },
                    {
                        "ticker": "A2",
                        "as_of": "2026-02-01T00:00:00+00:00",
                        "expected_return": 0.12,
                        "catalyst_success_prob": 0.60,
                        "confidence": 0.70,
                        "crowding_risk": 0.18,
                        "financing_risk": 0.14,
                        "thesis_horizon": "90d",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                    },
                    {
                        "ticker": "A3",
                        "as_of": "2026-03-01T00:00:00+00:00",
                        "expected_return": 0.16,
                        "catalyst_success_prob": 0.66,
                        "confidence": 0.74,
                        "crowding_risk": 0.16,
                        "financing_risk": 0.12,
                        "thesis_horizon": "90d",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                    },
                ],
            )
            store.replace_table(
                "labels",
                [
                    {"ticker": "A1", "as_of": "2026-01-01T00:00:00+00:00", "target_return_90d": 0.05, "target_catalyst_success": 1},
                    {"ticker": "A2", "as_of": "2026-02-01T00:00:00+00:00", "target_return_90d": 0.11, "target_catalyst_success": 1},
                    {"ticker": "A3", "as_of": "2026-03-01T00:00:00+00:00", "target_return_90d": 0.17, "target_catalyst_success": 1},
                ],
            )

            constructor = PortfolioConstructor(store=store)
            priors = constructor._validation_priors()

            self.assertGreater(priors["setup_type_scorecards"]["hard_catalyst"]["rank_ic"], 0.0)

    def test_portfolio_constructor_can_derive_empirical_priors_from_signal_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.replace_table(
                "signal_artifacts",
                [
                    {
                        "ticker": "CATA",
                        "as_of": "2025-01-01T00:00:00+00:00",
                        "expected_return": 0.16,
                        "catalyst_success_prob": 0.60,
                        "confidence": 0.72,
                        "crowding_risk": 0.20,
                        "financing_risk": 0.18,
                        "thesis_horizon": "90d",
                        "primary_event_type": "phase3_readout",
                        "primary_event_bucket": "clinical",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                    },
                    {
                        "ticker": "CATA",
                        "as_of": "2025-02-01T00:00:00+00:00",
                        "expected_return": 0.18,
                        "catalyst_success_prob": 0.62,
                        "confidence": 0.74,
                        "crowding_risk": 0.20,
                        "financing_risk": 0.18,
                        "thesis_horizon": "90d",
                        "primary_event_type": "phase3_readout",
                        "primary_event_bucket": "clinical",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                    },
                    {
                        "ticker": "FRAN",
                        "as_of": "2025-01-01T00:00:00+00:00",
                        "expected_return": 0.14,
                        "catalyst_success_prob": 0.56,
                        "confidence": 0.74,
                        "crowding_risk": 0.20,
                        "financing_risk": 0.18,
                        "thesis_horizon": "90d",
                        "primary_event_type": "capital_allocation",
                        "primary_event_bucket": "commercial",
                        "company_state": "commercialized",
                        "setup_type": "capital_allocation",
                    },
                    {
                        "ticker": "FRAN",
                        "as_of": "2025-02-01T00:00:00+00:00",
                        "expected_return": 0.13,
                        "catalyst_success_prob": 0.55,
                        "confidence": 0.73,
                        "crowding_risk": 0.20,
                        "financing_risk": 0.18,
                        "thesis_horizon": "90d",
                        "primary_event_type": "capital_allocation",
                        "primary_event_bucket": "commercial",
                        "company_state": "commercialized",
                        "setup_type": "capital_allocation",
                    },
                ],
            )
            store.replace_table(
                "labels",
                [
                    {"ticker": "CATA", "as_of": "2025-01-01T00:00:00+00:00", "target_return_90d": 0.22, "target_catalyst_success": 1},
                    {"ticker": "CATA", "as_of": "2025-02-01T00:00:00+00:00", "target_return_90d": 0.18, "target_catalyst_success": 1},
                    {"ticker": "FRAN", "as_of": "2025-01-01T00:00:00+00:00", "target_return_90d": -0.04, "target_catalyst_success": 0},
                    {"ticker": "FRAN", "as_of": "2025-02-01T00:00:00+00:00", "target_return_90d": -0.02, "target_catalyst_success": 0},
                ],
            )
            constructor = PortfolioConstructor(store=store)
            catalyst_signal = SignalArtifact(
                ticker="CATA",
                as_of="2025-03-01T00:00:00+00:00",
                expected_return=0.15,
                catalyst_success_prob=0.56,
                confidence=0.76,
                crowding_risk=0.25,
                financing_risk=0.20,
                thesis_horizon="90d",
                primary_event_type="phase3_readout",
                primary_event_bucket="clinical",
                company_state="pre_commercial",
                setup_type="hard_catalyst",
                rationale=[],
                supporting_evidence=[],
            )
            compounder_signal = SignalArtifact(
                ticker="FRAN",
                as_of="2025-03-01T00:00:00+00:00",
                expected_return=0.15,
                catalyst_success_prob=0.56,
                confidence=0.76,
                crowding_risk=0.25,
                financing_risk=0.20,
                thesis_horizon="90d",
                primary_event_type="capital_allocation",
                primary_event_bucket="commercial",
                company_state="commercialized",
                setup_type="capital_allocation",
                rationale=[],
                supporting_evidence=[],
            )

            priors = constructor._validation_priors()
            catalyst_edge = constructor._empirical_edge(catalyst_signal)
            compounder_edge = constructor._empirical_edge(compounder_signal)
            catalyst_rec = constructor.recommend(catalyst_signal)
            compounder_rec = constructor.recommend(compounder_signal)

            self.assertIn("hard_catalyst", priors["setup_type_scorecards"])
            self.assertIn("capital_allocation", priors["setup_type_scorecards"])
            self.assertGreater(
                priors["setup_type_scorecards"]["hard_catalyst"]["beta_adjusted_return"],
                priors["setup_type_scorecards"]["capital_allocation"]["beta_adjusted_return"],
            )
            self.assertGreater(catalyst_edge["edge_score"], compounder_edge["edge_score"])
            self.assertGreaterEqual(catalyst_rec.target_weight, compounder_rec.target_weight)

    def test_portfolio_constructor_smooths_small_weight_changes(self):
        constructor = PortfolioConstructor()
        previous_signal = SignalArtifact(
            ticker="STBL",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.18,
            catalyst_success_prob=0.68,
            confidence=0.78,
            crowding_risk=0.28,
            financing_risk=0.20,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        current_signal = SignalArtifact(
            ticker="STBL",
            as_of="2025-02-01T00:00:00+00:00",
            expected_return=0.17,
            catalyst_success_prob=0.65,
            confidence=0.75,
            crowding_risk=0.29,
            financing_risk=0.21,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        previous_recommendation = constructor.recommend(previous_signal)
        current_recommendation = constructor.recommend(
            current_signal,
            previous_recommendation=previous_recommendation,
            previous_signal=previous_signal,
        )

        self.assertGreater(previous_recommendation.target_weight, 0.0)
        self.assertLess(abs(current_recommendation.target_weight - previous_recommendation.target_weight), 1.0)

    def test_obesity_name_without_firm_event_is_treated_as_asymmetry_setup(self):
        raw = make_raw_company()
        raw["ticker"] = "VKTX"
        raw["company_name"] = "Viking Therapeutics"
        raw["finance"]["totalRevenue"] = 0.0
        raw["finance"]["description"] = "Obesity company with an oral and injectable metabolic pipeline."
        raw["trials"][0]["conditions"] = ["Obesity"]
        raw["trials"][0]["interventions"] = ["VK2735"]
        raw["trials"][1]["conditions"] = ["Obesity"]
        raw["trials"][1]["interventions"] = ["VK2809"]
        snapshot = build_company_snapshot(raw)
        signal = SignalArtifact(
            ticker="VKTX",
            as_of=snapshot.as_of,
            expected_return=0.18,
            catalyst_success_prob=0.60,
            confidence=0.70,
            crowding_risk=0.55,
            financing_risk=0.20,
            thesis_horizon="90d",
            primary_event_type="phase2_readout",
            primary_event_bucket="clinical",
            company_state="pre_commercial",
            rationale=[],
            supporting_evidence=[],
        )
        primary_event = snapshot.catalyst_events[0]
        lens = build_expectation_lens(snapshot, signal, primary_event, {"valuation_posture": "discounted"})

        self.assertEqual(lens["company_state"], "pre_commercial")
        self.assertEqual(lens["setup_type"], "asymmetry_without_near_term_catalyst")
        self.assertGreater(lens["competition_intensity"], 0.8)

    def test_walk_forward_leakage_audit_rejects_legacy_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            frame = pd.DataFrame(
                [
                    {
                        "entity_id": "A",
                        "ticker": "TEST",
                        "as_of": "2025-01-01T00:00:00",
                        "thesis_horizon": "90d",
                        "market_flow_return_6mo_legacy": 0.2,
                        "target_return_90d": 0.1,
                        "target_catalyst_success": 1,
                    }
                ]
            )
            frame.to_parquet(store.tables_dir / "feature_vectors.parquet", index=False)
            evaluator = WalkForwardEvaluator(store=store)
            self.assertFalse(evaluator.leakage_audit(frame))

    def test_universe_resolver_prefers_archived_snapshots_over_legacy_benchmark(self):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_snapshot(snapshot)
            resolver = UniverseResolver(store=store)
            universe = resolver.resolve_default_universe(prefer_archive=True)

            self.assertEqual(universe[0][0], "TEST")

    def test_universe_resolver_uses_raw_archive_inventory_when_tables_are_stale(self):
        snapshot = build_company_snapshot(make_raw_company() | {"ticker": "ONLY", "company_name": "Only Bio"})
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("snapshots", "ONLY_2026-03-28T16-45-12.853369+00-00", asdict(snapshot))
            resolver = UniverseResolver(store=store)

            universe = resolver.resolve_default_universe(prefer_archive=True)

            self.assertEqual(universe[0][0], "ONLY")

    def test_select_lead_trial_prefers_registrational_trial_over_first_inserted_trial(self):
        raw = make_raw_company()
        raw["trials"] = [
            {
                "nct_id": "NCT-LOW",
                "title": "Dose escalation safety study",
                "overall_status": "ACTIVE_NOT_RECRUITING",
                "phase": ["Phase 1"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["safety and tolerability"],
                "enrollment": 36,
            },
            {
                "nct_id": "NCT-HIGH",
                "title": "Pivotal RCC registrational trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 420,
            },
        ]
        snapshot = build_company_snapshot(raw)

        lead_trial = select_lead_trial(snapshot.programs[0])

        self.assertIsNotNone(lead_trial)
        self.assertEqual(lead_trial.title, "Pivotal RCC registrational trial")

    @patch("biopharma_agent.vnext.facade.AutoResearchAgent.generate_literature_review", return_value="Literature summary.")
    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    def test_facade_builds_legacy_report_shape(self, mock_fetch, _mock_lit):
        mock_fetch.return_value = make_raw_company()
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = TatetuckPlatform(store=LocalResearchStore(base_dir=tmpdir))
            report = platform.build_legacy_report("TEST", "Test Therapeutics")
            self.assertIn("valuation", report)
            self.assertIn("signal_artifact", report)
            self.assertIn("portfolio_recommendation", report)
            self.assertEqual(report["ticker"], "TEST")

    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    @patch("biopharma_agent.vnext.sources.CorporateCalendarClient.fetch_company_calendar")
    @patch("biopharma_agent.vnext.sources.SECXBRLClient.fetch_company_facts")
    def test_archive_universe_returns_store_summary(self, mock_sec, mock_calendar, mock_fetch):
        mock_fetch.return_value = make_raw_company()
        mock_sec.return_value = {
            "ticker": "TEST",
            "cik": "0000000123",
            "parsed": {
                "revenue_ttm": 120_000_000,
                "operating_cashflow": 25_000_000,
                "recent_filings": [],
                "recent_offering_forms": [],
            },
        }
        mock_calendar.return_value = {"ticker": "TEST", "events": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = TatetuckPlatform(store=LocalResearchStore(base_dir=tmpdir))
            analyses, summary = archive_universe(platform, [("TEST", "Test Therapeutics")])
            self.assertEqual(len(analyses), 1)
            self.assertEqual(summary.archived_companies, 1)
            self.assertEqual(summary.sec_enriched_companies, 1)
            self.assertGreaterEqual(summary.snapshot_rows, 1)
            self.assertGreaterEqual(summary.feature_rows, 1)
            self.assertGreaterEqual(summary.prediction_rows, 1)
            self.assertEqual(summary.top_ideas[0]["ticker"], "TEST")

    @patch("biopharma_agent.vnext.facade.IngestionService.ingest_company", side_effect=RuntimeError("offline"))
    def test_facade_falls_back_to_archived_snapshot(self, _mock_ingest):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("snapshots", "TEST_2025-01-01T00-00-00+00-00", asdict(snapshot))
            platform = TatetuckPlatform(store=store)
            analysis = platform.analyze_ticker("TEST", "Test Therapeutics", include_literature=False, fallback_to_archive=True)
            self.assertEqual(analysis.snapshot.ticker, "TEST")
            self.assertEqual(analysis.metadata["analysis_source"], "archive_fallback")
            self.assertIn("live_ingestion_error", analysis.snapshot.metadata)

    def test_facade_archive_read_only_does_not_persist(self):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("snapshots", "TEST_2025-01-01T00-00-00+00-00", asdict(snapshot))
            platform = TatetuckPlatform(store=store)
            before_rows = len(store.read_table("company_snapshots"))
            analysis = platform.analyze_ticker(
                "TEST",
                "Test Therapeutics",
                include_literature=False,
                prefer_archive=True,
                persist=False,
            )
            after_rows = len(store.read_table("company_snapshots"))
            self.assertEqual(analysis.metadata["analysis_source"], "archive")
            self.assertFalse(analysis.metadata["persisted"])
            self.assertIn("why_now", analysis.metadata)
            self.assertIn("valuation_summary", analysis.metadata)
            self.assertIn("kill_points", analysis.metadata)
            self.assertIn("company_state", analysis.metadata)
            self.assertIn("setup_type", analysis.metadata)
            self.assertIn("competitive_summary", analysis.metadata)
            self.assertIn("asymmetry_summary", analysis.metadata)
            self.assertEqual(before_rows, after_rows)

    def test_facade_primary_event_prefers_near_term_business_event_for_commercialized_name_without_exact_hard_catalyst(self):
        raw = make_raw_company()
        raw["ticker"] = "BMRN"
        raw["company_name"] = "BioMarin Pharmaceutical"
        raw["finance"]["totalRevenue"] = 3_200_000_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 3, tzinfo=timezone.utc))
        snapshot.catalyst_events = [
            CatalystEvent(
                event_id="BMRN:phase3",
                program_id="BMRN:1",
                event_type="phase3_readout",
                title="Legacy phase 3 readout",
                expected_date="2026-08-01",
                horizon_days=120,
                probability=0.7,
                importance=0.8,
                crowdedness=0.3,
                status="phase_timing_estimate",
            ),
            CatalystEvent(
                event_id="BMRN:commercial",
                program_id=None,
                event_type="commercial_update",
                title="BMRN estimated commercial update",
                expected_date="2026-05-18",
                horizon_days=45,
                probability=0.78,
                importance=0.6,
                crowdedness=0.45,
                status="estimated_from_revenue",
            ),
        ]

        primary = TatetuckPlatform._primary_event(snapshot)

        self.assertIsNotNone(primary)
        self.assertEqual(primary.event_type, "commercial_update")

    def test_facade_primary_event_prefers_guided_strategic_event_for_commercialized_name(self):
        raw = make_raw_company()
        raw["ticker"] = "BMRN"
        raw["company_name"] = "BioMarin Pharmaceutical"
        raw["finance"]["totalRevenue"] = 3_200_000_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 3, tzinfo=timezone.utc))
        snapshot.catalyst_events = [
            CatalystEvent(
                event_id="BMRN:phase3",
                program_id="BMRN:1",
                event_type="phase3_readout",
                title="Near-term phase 3 readout",
                expected_date="2026-05-01",
                horizon_days=28,
                probability=0.7,
                importance=0.8,
                crowdedness=0.3,
                status="exact_company_calendar",
            ),
            CatalystEvent(
                event_id="BMRN:deal",
                program_id=None,
                event_type="strategic_transaction",
                title="Acquisition expected to close in Q2 2026",
                expected_date="2026-06-30T00:00:00",
                horizon_days=88,
                probability=0.8,
                importance=0.9,
                crowdedness=0.3,
                status="guided_company_event",
            ),
        ]

        primary = TatetuckPlatform._primary_event(snapshot)

        self.assertIsNotNone(primary)
        self.assertEqual(primary.event_type, "strategic_transaction")

    def test_aggregate_company_signal_uses_selected_primary_event_not_modal_prediction_event(self):
        raw = make_raw_company()
        raw["ticker"] = "BMRN"
        raw["company_name"] = "BioMarin Pharmaceutical"
        raw["finance"]["totalRevenue"] = 3_200_000_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 3, tzinfo=timezone.utc))
        snapshot.catalyst_events = [
            CatalystEvent(
                event_id="BMRN:phase3",
                program_id="BMRN:1",
                event_type="phase3_readout",
                title="Legacy phase 3 readout",
                expected_date="2026-08-01",
                horizon_days=120,
                probability=0.7,
                importance=0.8,
                crowdedness=0.3,
                status="phase_timing_estimate",
            ),
            CatalystEvent(
                event_id="BMRN:commercial",
                program_id=None,
                event_type="commercial_update",
                title="BMRN estimated commercial update",
                expected_date="2026-05-18",
                horizon_days=45,
                probability=0.78,
                importance=0.6,
                crowdedness=0.45,
                status="estimated_from_revenue",
            ),
        ]
        platform = TatetuckPlatform(store=LocalResearchStore(base_dir=tempfile.mkdtemp()))
        predictions = [
            ModelPrediction(
                entity_id="BMRN:1",
                ticker="BMRN",
                as_of=snapshot.as_of,
                expected_return=0.2,
                catalyst_success_prob=0.7,
                confidence=0.8,
                crowding_risk=0.2,
                financing_risk=0.2,
                thesis_horizon="90d",
                model_name="test",
                model_version="v1",
                metadata={"event_type": "phase3_readout"},
            )
        ]

        signal = platform._aggregate_company_signal(snapshot, predictions, company_state="commercialized")

        self.assertEqual(signal.primary_event_type, "commercial_update")
        self.assertEqual(signal.primary_event_bucket, "commercial")

    def test_expectation_lens_uses_capital_allocation_setup_for_strategic_events(self):
        raw = make_raw_company()
        raw["ticker"] = "BMRN"
        raw["company_name"] = "BioMarin Pharmaceutical"
        raw["finance"]["totalRevenue"] = 3_200_000_000
        snapshot = build_company_snapshot(raw, as_of=datetime(2026, 4, 3, tzinfo=timezone.utc))
        primary_event = CatalystEvent(
            event_id="BMRN:deal",
            program_id=None,
            event_type="strategic_transaction",
            title="Acquisition expected to close in Q2 2026",
            expected_date="2026-06-30T00:00:00",
            horizon_days=88,
            probability=0.8,
            importance=0.9,
            crowdedness=0.3,
            status="guided_company_event",
        )
        signal = SignalArtifact(
            ticker="BMRN",
            as_of=snapshot.as_of,
            expected_return=0.16,
            catalyst_success_prob=0.62,
            confidence=0.74,
            crowding_risk=0.30,
            financing_risk=0.18,
            thesis_horizon="90d",
            primary_event_type="strategic_transaction",
            primary_event_bucket="strategic",
            company_state="commercialized",
            rationale=[],
            supporting_evidence=[],
        )

        lens = build_expectation_lens(snapshot, signal, primary_event, {"valuation_posture": "discounted"})

        self.assertEqual(lens["setup_type"], "capital_allocation")
        self.assertIn("strategic event window", lens["market_view"])

    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    @patch("biopharma_agent.vnext.sources.CorporateCalendarClient.fetch_company_calendar")
    @patch("biopharma_agent.vnext.sources.SECXBRLClient.fetch_company_facts")
    def test_ingestion_service_recovers_from_cached_payloads(self, mock_sec, mock_calendar, mock_fetch):
        sparse_raw = {
            "ticker": "TEST",
            "company_name": "Test Therapeutics",
            "finance": {"ticker": "TEST", "trailing_6mo_return": None, "momentum_3mo": None},
            "trials": [],
            "num_trials": 0,
            "num_total_trials": 0,
            "pubmed_papers": [],
            "num_papers": 0,
            "conditions": [],
        }
        mock_fetch.return_value = sparse_raw
        mock_sec.return_value = {"ticker": "TEST", "records": [], "source": "sec_xbrl", "status": "missing_cik"}
        mock_calendar.return_value = {"ticker": "TEST", "events": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("legacy_prepare", "TEST_2025-01-01T00-00-00+00-00", make_raw_company())
            store.write_raw_payload(
                "sec_xbrl",
                "TEST_2025-01-01T00-00-00+00-00",
                {
                    "ticker": "TEST",
                    "cik": "0000000123",
                    "parsed": {
                        "revenue_ttm": 120_000_000,
                        "operating_cashflow": 25_000_000,
                        "recent_filings": [],
                        "recent_offering_forms": [],
                    },
                },
            )
            store.write_raw_payload(
                "corp_calendar",
                "TEST_2025-01-01T00-00-00+00-00",
                {
                    "ticker": "TEST",
                    "events": [
                        {
                            "event_type": "earnings",
                            "title": "TEST estimated quarterly update",
                            "expected_date": "2025-05-01",
                        }
                    ],
                },
            )

            snapshot = IngestionService(store=store).ingest_company("TEST", "Test Therapeutics")
            self.assertTrue(snapshot.programs)
            self.assertEqual(snapshot.metadata["sec_cik"], "0000000123")
            self.assertEqual(
                snapshot.metadata["fallback_sources"],
                ["corp_calendar", "legacy_prepare", "sec_xbrl"],
            )

    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot", return_value=make_raw_company())
    @patch("biopharma_agent.vnext.sources.CorporateCalendarClient.fetch_company_calendar", return_value={"ticker": "TEST", "events": []})
    @patch(
        "biopharma_agent.vnext.sources.SECXBRLClient.fetch_company_facts",
        return_value={"ticker": "TEST", "records": [], "source": "sec_xbrl", "status": "missing_cik"},
    )
    @patch(
        "biopharma_agent.vnext.sources.EODHDEventTapeClient.fetch_event_payload",
        return_value={"ticker": "TEST", "events": [], "source": "eodhd_event_tape"},
    )
    def test_ingestion_service_skips_stale_cached_event_tape(
        self,
        _mock_events,
        _mock_sec,
        _mock_calendar,
        _mock_fetch,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "eodhd_event_tape",
                "TEST_2024-08-05T00-00-00+00-00",
                {
                    "ticker": "TEST",
                    "as_of": "2024-08-05T00:00:00+00:00",
                    "events": [
                        {
                            "event_id": "TEST.US:news:earnings:2024-08-05T20:03:00",
                            "event_type": "earnings",
                            "title": "Old earnings release",
                            "expected_date": "2024-08-05T20:03:00",
                            "status": "exact_press_release",
                            "importance": 0.58,
                            "crowdedness": 0.35,
                            "source": "eodhd_news",
                        }
                    ],
                    "source": "eodhd_event_tape",
                },
            )

            snapshot = IngestionService(store=store).ingest_company(
                "TEST",
                "Test Therapeutics",
                as_of=datetime(2026, 4, 2, tzinfo=timezone.utc),
                persist=False,
            )

            self.assertTrue(snapshot.metadata.get("stale_event_tape_skipped"))
            self.assertFalse(any(event.title == "Old earnings release" for event in snapshot.catalyst_events))


if __name__ == "__main__":
    unittest.main()
