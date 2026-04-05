import json
import tempfile
import unittest
from datetime import datetime, timezone

from biopharma_agent.vnext.dashboard import build_dashboard_payload
from biopharma_agent.vnext.storage import LocalResearchStore


class TestVNextDashboard(unittest.TestCase):
    def test_dashboard_falls_back_to_program_predictions_when_company_signals_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-01-06T00:00:00+00:00"

            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "BNTX",
                        "company_name": "BioNTech",
                        "as_of": as_of,
                        "market_cap": 10_000_000_000,
                        "revenue": 2_000_000_000,
                    }
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "BNTX_2026-01-06T00-00-00+00-00",
                {
                    "ticker": "BNTX",
                    "company_name": "BioNTech",
                    "as_of": as_of,
                    "market_cap": 10_000_000_000,
                    "enterprise_value": 9_500_000_000,
                    "revenue": 2_000_000_000,
                    "cash": 4_000_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.25,
                    "programs": [],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [
                        {
                            "source": "pubmed",
                            "source_id": "pmid:1",
                            "title": "BNTX evidence",
                            "excerpt": "Peer-reviewed evidence attached to the company snapshot.",
                        }
                    ],
                    "metadata": {},
                },
            )
            store.append_records(
                "predictions",
                [
                    {
                        "entity_id": "BNTX:1",
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "expected_return": 0.34,
                        "catalyst_success_prob": 0.77,
                        "confidence": 0.83,
                        "crowding_risk": 0.31,
                        "financing_risk": 0.22,
                        "thesis_horizon": "90d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps(
                            {
                                "program_name": "PD-L1 combo",
                                "event_type": "phase3_readout",
                                "event_status": "phase_timing_estimate",
                                "event_expected_date": "2026-05-20",
                                "primary_indication": "NSCLC",
                                "phase": "PHASE3",
                                "modality": "antibody",
                                "company_state": "pre_commercial",
                                "event_bucket": "clinical",
                            }
                        ),
                    },
                    {
                        "entity_id": "BNTX:2",
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "expected_return": -0.26,
                        "catalyst_success_prob": 0.58,
                        "confidence": 0.79,
                        "crowding_risk": 0.49,
                        "financing_risk": 0.41,
                        "thesis_horizon": "180d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps(
                            {
                                "program_name": "ADC follow-up",
                                "event_type": "phase2_readout",
                                "event_status": "phase_timing_estimate",
                                "event_expected_date": "2026-07-10",
                                "primary_indication": "Breast cancer",
                                "phase": "PHASE2",
                                "modality": "adc",
                                "company_state": "pre_commercial",
                                "event_bucket": "clinical",
                            }
                        ),
                    },
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["research_book"]["company_idea_count"], 0)
            self.assertEqual(payload["research_book"]["program_idea_count"], 2)
            self.assertEqual(payload["research_book"]["longs"][0]["idea_level"], "program")
            self.assertEqual(payload["research_book"]["longs"][0]["program_name"], "PD-L1 combo")
            self.assertEqual(payload["research_book"]["shorts"][0]["program_name"], "ADC follow-up")
            self.assertFalse(payload["research_book"]["longs"][0]["deployable"])
            self.assertEqual(payload["research_book"]["shorts"][0]["deployment_status"], "research_only_program_short")
            self.assertIn("archived research snapshot", payload["research_book"]["notice"])
            self.assertIn("program-level predictions", payload["research_book"]["notice"])
            self.assertEqual(payload["summary"]["idea_count"], 2)
            self.assertEqual(payload["research_book"]["longs"][0]["evidence_count"], 1)

    def test_dashboard_tops_up_sparse_company_book_and_marks_current_plan_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-01-06T00:00:00+00:00"
            planned_at = "2026-04-04T13:30:00+00:00"

            store.append_records(
                "signal_artifacts",
                [
                    {
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "expected_return": -0.18,
                        "catalyst_success_prob": 0.62,
                        "confidence": 0.81,
                        "crowding_risk": 0.45,
                        "financing_risk": 0.21,
                        "thesis_horizon": "90d",
                        "primary_event_type": "commercial_update",
                        "primary_event_bucket": "commercial",
                        "primary_event_status": "exact_sec_filing",
                        "primary_event_date": "2026-01-06T11:02:00+00:00",
                        "primary_event_exact": True,
                        "company_state": "pre_commercial",
                        "setup_type": "watchful",
                        "rationale": json.dumps(["Street setup is asymmetric to downside."]),
                        "supporting_evidence": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "portfolio_recommendations",
                [
                    {
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "stance": "avoid",
                        "target_weight": 0.0,
                        "max_weight": 0.0,
                        "confidence": 0.81,
                        "scenario": "watchlist only",
                        "risk_flags": json.dumps(["commercial revision risk"]),
                    }
                ],
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "BNTX",
                        "company_name": "BioNTech",
                        "as_of": as_of,
                        "market_cap": 10_000_000_000,
                        "revenue": 2_000_000_000,
                    }
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "BNTX_2026-01-06T00-00-00+00-00",
                {
                    "ticker": "BNTX",
                    "company_name": "BioNTech",
                    "as_of": as_of,
                    "market_cap": 10_000_000_000,
                    "enterprise_value": 9_500_000_000,
                    "revenue": 2_000_000_000,
                    "cash": 4_000_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.25,
                    "programs": [],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [
                        {
                            "source": "pubmed",
                            "source_id": "pmid:1",
                            "title": "BNTX evidence",
                            "excerpt": "Peer-reviewed evidence attached to the company snapshot.",
                        }
                    ],
                    "metadata": {},
                },
            )
            store.append_records(
                "predictions",
                [
                    {
                        "entity_id": "BNTX:1",
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "expected_return": 0.29,
                        "catalyst_success_prob": 0.74,
                        "confidence": 0.85,
                        "crowding_risk": 0.28,
                        "financing_risk": 0.19,
                        "thesis_horizon": "90d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps(
                            {
                                "program_name": "Oncology combo",
                                "event_type": "phase3_readout",
                                "event_status": "phase_timing_estimate",
                                "event_expected_date": "2026-05-20",
                                "primary_indication": "NSCLC",
                                "phase": "PHASE3",
                                "modality": "antibody",
                                "company_state": "pre_commercial",
                                "event_bucket": "clinical",
                            }
                        ),
                    },
                    {
                        "entity_id": "BNTX:2",
                        "ticker": "BNTX",
                        "as_of": as_of,
                        "expected_return": -0.22,
                        "catalyst_success_prob": 0.56,
                        "confidence": 0.77,
                        "crowding_risk": 0.51,
                        "financing_risk": 0.37,
                        "thesis_horizon": "180d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps(
                            {
                                "program_name": "Vaccine follow-up",
                                "event_type": "phase2_readout",
                                "event_status": "phase_timing_estimate",
                                "event_expected_date": "2026-07-10",
                                "primary_indication": "Influenza",
                                "phase": "PHASE2",
                                "modality": "mrna",
                                "company_state": "pre_commercial",
                                "event_bucket": "clinical",
                            }
                        ),
                    },
                ],
            )
            store.append_records(
                "pipeline_runs",
                [
                    {
                        "job_name": "trade_vnext",
                        "status": "completed",
                        "started_at": "2026-04-04T13:20:00+00:00",
                        "finished_at": "2026-04-04T13:35:00+00:00",
                        "duration_seconds": 900,
                    }
                ],
            )
            store.append_records(
                "order_plans",
                [
                    {
                        "symbol": "BNTX",
                        "company_name": "BioNTech",
                        "action": "buy",
                        "scenario": "pre-catalyst long",
                        "confidence": 0.74,
                        "target_weight": 3.0,
                        "scaled_target_weight": 3.0,
                        "target_notional": 30_000.0,
                        "delta_notional": 30_000.0,
                        "requested_notional": 30_000.0,
                        "planned_at": planned_at,
                        "as_of": "2026-04-04T13:35:00+00:00",
                        "execution_profile": "standard",
                        "setup_type": "hard_catalyst",
                    }
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, 14, 0, tzinfo=timezone.utc))

            self.assertEqual(payload["summary"]["company_idea_count"], 1)
            self.assertEqual(payload["summary"]["program_idea_count"], 2)
            self.assertTrue(payload["current_plan"]["trade_rows"][0]["idea_key"].startswith("trade:BNTX:"))
            self.assertEqual(payload["research_book"]["company_ideas"][0]["direction"], "watch")
            self.assertEqual(payload["research_book"]["shorts"][0]["idea_level"], "program")
            self.assertTrue(any(item["idea_level"] == "program" and item["in_current_plan"] for item in payload["research_book"]["longs"]))
            self.assertIn("topped up with program-level predictions", payload["research_book"]["notice"])
            self.assertTrue(all(item["event_date"] >= "2026-05-20" for item in payload["research_book"]["catalyst_calendar"]))
            self.assertEqual(payload["research_book"]["company_ideas"][0]["evidence_count"], 1)
            self.assertEqual(payload["research_book"]["company_ideas"][0]["evidence_items"][0]["source"], "pubmed")

    def test_failed_run_does_not_inherit_stale_orders_but_blotter_keeps_recent_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.append_records(
                "pipeline_runs",
                [
                    {
                        "job_name": "trade_vnext",
                        "status": "failed",
                        "started_at": "2026-04-04T13:20:00+00:00",
                        "finished_at": "2026-04-04T13:35:00+00:00",
                        "duration_seconds": 900,
                        "metrics": json.dumps({"actionable_instructions": 0}),
                        "notes": "ConnectionError: broker unreachable",
                    }
                ],
            )
            store.append_records(
                "order_plans",
                [
                    {
                        "symbol": "REGN",
                        "company_name": "Regeneron",
                        "action": "buy",
                        "scenario": "pairs candidate",
                        "confidence": 0.88,
                        "target_weight": 1.75,
                        "scaled_target_weight": 1.75,
                        "target_notional": 17_500.0,
                        "planned_at": "2026-03-29T12:18:55+00:00",
                        "as_of": "2026-03-29T12:18:55+00:00",
                    },
                    {
                        "symbol": "REGN",
                        "company_name": "Regeneron",
                        "action": "buy",
                        "scenario": "pairs candidate",
                        "confidence": 0.88,
                        "target_weight": 1.75,
                        "scaled_target_weight": 1.75,
                        "target_notional": 17_500.0,
                        "planned_at": "2026-03-29T12:23:58+00:00",
                        "as_of": "2026-03-29T12:23:58+00:00",
                    },
                ],
            )
            store.append_records(
                "order_ledger",
                [
                    {
                        "symbol": "REGN",
                        "company_name": "Regeneron",
                        "action": "buy",
                        "scenario": "pairs candidate",
                        "confidence": 0.88,
                        "target_weight": 1.75,
                        "scaled_target_weight": 1.75,
                        "target_notional": 17_500.0,
                        "planned_at": "2026-03-29T12:18:55+00:00",
                        "as_of": "2026-03-29T12:18:55+00:00",
                    },
                    {
                        "symbol": "REGN",
                        "company_name": "Regeneron",
                        "action": "buy",
                        "scenario": "pairs candidate",
                        "confidence": 0.88,
                        "target_weight": 1.75,
                        "scaled_target_weight": 1.75,
                        "target_notional": 17_500.0,
                        "planned_at": "2026-03-29T12:23:58+00:00",
                        "as_of": "2026-03-29T12:23:58+00:00",
                    },
                ],
            )
            store.append_records(
                "order_submissions",
                [
                    {
                        "symbol": "REGN",
                        "submitted_at": "2026-03-29T12:23:58.267755+00:00",
                        "status": "submitted",
                        "action": "buy_notional",
                        "order_id": "shared-order",
                        "client_order_id": "regn-buy-1",
                    }
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, 14, 0, tzinfo=timezone.utc))

            self.assertEqual(payload["current_plan"]["trade_rows"], [])
            self.assertEqual(payload["summary"]["buy_orders"], 0)
            self.assertEqual(payload["summary"]["gross_target_weight"], 0.0)
            self.assertEqual(payload["current_plan"]["trade_rows_source"], "latest_activity")
            self.assertEqual(len(payload["current_plan"]["recent_trade_rows"]), 1)
            self.assertEqual(payload["current_plan"]["recent_trade_rows"][0]["submission"]["order_id"], "shared-order")

    def test_dashboard_uses_latest_signal_per_ticker_not_global_snapshot_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "AAA",
                        "company_name": "Alpha Bio",
                        "as_of": "2026-01-01T00:00:00+00:00",
                        "market_cap": 1_000_000_000,
                        "revenue": 100_000_000,
                    },
                    {
                        "ticker": "BBB",
                        "company_name": "Beta Bio",
                        "as_of": "2024-08-01T00:00:00+00:00",
                        "market_cap": 2_000_000_000,
                        "revenue": 50_000_000,
                    },
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "AAA_2026-01-01T00-00-00+00-00",
                {
                    "ticker": "AAA",
                    "company_name": "Alpha Bio",
                    "as_of": "2026-01-01T00:00:00+00:00",
                    "market_cap": 1_000_000_000,
                    "enterprise_value": 1_000_000_000,
                    "revenue": 100_000_000,
                    "cash": 200_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.2,
                    "programs": [],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [{"source": "sec", "source_id": "1", "title": "AAA filing", "excerpt": "AAA evidence"}],
                    "metadata": {},
                },
            )
            store.write_raw_payload(
                "snapshots",
                "BBB_2024-08-01T00-00-00+00-00",
                {
                    "ticker": "BBB",
                    "company_name": "Beta Bio",
                    "as_of": "2024-08-01T00:00:00+00:00",
                    "market_cap": 2_000_000_000,
                    "enterprise_value": 2_000_000_000,
                    "revenue": 50_000_000,
                    "cash": 250_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.2,
                    "programs": [],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [{"source": "press_release", "source_id": "2", "title": "BBB PR", "excerpt": "BBB evidence"}],
                    "metadata": {},
                },
            )
            store.append_records(
                "signal_artifacts",
                [
                    {
                        "ticker": "AAA",
                        "as_of": "2026-01-01T00:00:00+00:00",
                        "expected_return": 0.21,
                        "catalyst_success_prob": 0.68,
                        "confidence": 0.77,
                        "crowding_risk": 0.2,
                        "financing_risk": 0.2,
                        "thesis_horizon": "90d",
                        "rationale": json.dumps(["AAA thesis"]),
                        "supporting_evidence": json.dumps([]),
                    },
                    {
                        "ticker": "BBB",
                        "as_of": "2024-08-01T00:00:00+00:00",
                        "expected_return": 0.18,
                        "catalyst_success_prob": 0.61,
                        "confidence": 0.71,
                        "crowding_risk": 0.3,
                        "financing_risk": 0.2,
                        "thesis_horizon": "90d",
                        "rationale": json.dumps(["BBB thesis"]),
                        "supporting_evidence": json.dumps([]),
                    },
                ],
            )
            store.append_records(
                "portfolio_recommendations",
                [
                    {
                        "ticker": "AAA",
                        "as_of": "2026-01-01T00:00:00+00:00",
                        "stance": "long",
                        "target_weight": 3.0,
                        "max_weight": 4.0,
                        "confidence": 0.77,
                        "scenario": "alpha",
                        "risk_flags": json.dumps([]),
                    },
                    {
                        "ticker": "BBB",
                        "as_of": "2024-08-01T00:00:00+00:00",
                        "stance": "long",
                        "target_weight": 2.0,
                        "max_weight": 4.0,
                        "confidence": 0.71,
                        "scenario": "beta",
                        "risk_flags": json.dumps([]),
                    },
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["summary"]["company_idea_count"], 2)
            self.assertTrue(payload["research_book"]["mixed_vintage"])
            self.assertIn("mixed-vintage research deck", payload["research_book"]["notice"])
            tickers = {item["symbol"] for item in payload["research_book"]["company_ideas"]}
            self.assertEqual(tickers, {"AAA", "BBB"})
            bbb = next(item for item in payload["research_book"]["company_ideas"] if item["symbol"] == "BBB")
            self.assertEqual(bbb["evidence_items"][0]["source"], "press_release")

    def test_dashboard_program_context_matches_by_entity_id_and_selects_best_trial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-03-28T16:45:12.853369+00:00"
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "AAA",
                        "company_name": "Alias Bio",
                        "as_of": as_of,
                        "market_cap": 1_500_000_000,
                        "revenue": 0,
                    }
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "AAA_2026-03-28T16-45-12.853369+00-00",
                {
                    "ticker": "AAA",
                    "company_name": "Alias Bio",
                    "as_of": as_of,
                    "market_cap": 1_500_000_000,
                    "enterprise_value": 1_300_000_000,
                    "revenue": 0,
                    "cash": 400_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.3,
                    "programs": [
                        {
                            "program_id": "AAA:1",
                            "name": "TTX-101",
                            "modality": "antibody",
                            "phase": "PHASE3",
                            "conditions": ["Metastatic CRC"],
                            "trials": [
                                {
                                    "trial_id": "NCT-LOW",
                                    "title": "Dose escalation safety study",
                                    "phase": "PHASE1",
                                    "status": "ACTIVE_NOT_RECRUITING",
                                    "conditions": ["Metastatic CRC"],
                                    "interventions": ["TTX-101"],
                                    "enrollment": 36,
                                    "primary_outcomes": ["safety and tolerability"],
                                },
                                {
                                    "trial_id": "NCT-HIGH",
                                    "title": "Pivotal CRC registrational study",
                                    "phase": "PHASE3",
                                    "status": "RECRUITING",
                                    "conditions": ["Metastatic CRC"],
                                    "interventions": ["TTX-101"],
                                    "enrollment": 420,
                                    "primary_outcomes": ["overall survival"],
                                },
                            ],
                            "pos_prior": 0.62,
                            "tam_estimate": 5_000_000_000,
                            "catalyst_events": [],
                            "evidence": [
                                {
                                    "source": "pubmed",
                                    "source_id": "pmid:99",
                                    "title": "TTX-101 translational paper",
                                    "excerpt": "Mechanistic evidence for TTX-101.",
                                }
                            ],
                        }
                    ],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [],
                    "metadata": {},
                },
            )
            store.append_records(
                "predictions",
                [
                    {
                        "entity_id": "AAA:1",
                        "ticker": "AAA",
                        "as_of": as_of,
                        "expected_return": 0.27,
                        "catalyst_success_prob": 0.71,
                        "confidence": 0.81,
                        "crowding_risk": 0.2,
                        "financing_risk": 0.2,
                        "thesis_horizon": "90d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps(
                            {
                                "program_name": "Alias registrational asset",
                                "event_type": "phase3_readout",
                                "event_status": "phase_timing_estimate",
                                "event_expected_date": "2026-06-01",
                                "primary_indication": "Metastatic CRC",
                                "phase": "PHASE3",
                                "modality": "antibody",
                                "company_state": "pre_commercial",
                                "event_bucket": "clinical",
                            }
                        ),
                    }
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            program = payload["research_book"]["program_ideas"][0]
            self.assertEqual(program["lead_trial_title"], "Pivotal CRC registrational study")
            self.assertEqual(program["lead_trial_phase"], "PHASE3")
            self.assertEqual(program["evidence_items"][0]["source"], "pubmed")

    def test_dashboard_filters_program_predictions_to_latest_snapshot_per_ticker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            latest_as_of = "2026-03-28T16:45:12.853369+00:00"
            stale_as_of = "2024-08-01T00:00:00+00:00"
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "AAA",
                        "company_name": "Alpha Bio",
                        "as_of": latest_as_of,
                        "market_cap": 1_000_000_000,
                        "revenue": 0,
                    }
                ],
            )
            store.append_records(
                "predictions",
                [
                    {
                        "entity_id": "AAA:1",
                        "ticker": "AAA",
                        "as_of": latest_as_of,
                        "expected_return": 0.22,
                        "catalyst_success_prob": 0.7,
                        "confidence": 0.8,
                        "crowding_risk": 0.2,
                        "financing_risk": 0.2,
                        "thesis_horizon": "90d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps({"program_name": "Current Program", "phase": "PHASE3"}),
                    },
                    {
                        "entity_id": "AAA:9",
                        "ticker": "AAA",
                        "as_of": stale_as_of,
                        "expected_return": -0.31,
                        "catalyst_success_prob": 0.4,
                        "confidence": 0.6,
                        "crowding_risk": 0.5,
                        "financing_risk": 0.4,
                        "thesis_horizon": "90d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps({"program_name": "Stale Program", "phase": "PHASE1"}),
                    },
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["research_book"]["program_idea_count"], 1)
            self.assertEqual(payload["research_book"]["program_ideas"][0]["program_name"], "Current Program")
            self.assertNotIn("mixed-vintage research deck", payload["research_book"]["notice"])

    def test_dashboard_filters_delisted_zero_cap_company_from_research_book(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-04-04T20:09:55.014793+00:00"
            store.append_records(
                "signal_artifacts",
                [
                    {
                        "ticker": "BPMC",
                        "as_of": as_of,
                        "expected_return": 1.27,
                        "catalyst_success_prob": 0.8,
                        "confidence": 0.89,
                        "crowding_risk": 0.3,
                        "financing_risk": 0.1,
                        "thesis_horizon": "90d",
                        "primary_event_type": "phase3_readout",
                        "primary_event_bucket": "clinical",
                        "primary_event_status": "phase_timing_estimate",
                        "primary_event_date": "2026-08-02T00:00:00+00:00",
                        "primary_event_exact": False,
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                        "rationale": json.dumps(["Legacy bullish thesis."]),
                        "supporting_evidence": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "portfolio_recommendations",
                [
                    {
                        "ticker": "BPMC",
                        "as_of": as_of,
                        "stance": "long",
                        "target_weight": 4.0,
                        "max_weight": 4.0,
                        "confidence": 0.89,
                        "scenario": "pre-catalyst long",
                        "risk_flags": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "BPMC",
                        "company_name": "Blueprint Medicines",
                        "as_of": as_of,
                        "market_cap": 0.0,
                        "revenue": 0.0,
                    }
                ],
            )
            store.append_records(
                "universe_membership",
                [
                    {
                        "ticker": "BPMC",
                        "company_name": "Blueprint Medicines Corp",
                        "as_of": "2026-03-29T18:26:05.623753+00:00",
                        "membership_source": "eodhd_exchange_symbols",
                        "is_delisted": True,
                        "exchange": "NASDAQ",
                        "security_type": "Common Stock",
                        "listing_symbol": "BPMC",
                    },
                    {
                        "ticker": "BPMC",
                        "company_name": "BPMC",
                        "as_of": as_of,
                        "membership_source": "prepare_compatibility_layer",
                        "is_delisted": False,
                    },
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "BPMC_2026-04-04T20-09-55.014793+00-00",
                {
                    "ticker": "BPMC",
                    "company_name": "BPMC",
                    "as_of": as_of,
                    "market_cap": 0.0,
                    "enterprise_value": 0.0,
                    "revenue": 0.0,
                    "cash": 0.0,
                    "debt": 0.0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.3,
                    "programs": [],
                    "approved_products": [],
                    "catalyst_events": [],
                    "financing_events": [],
                    "evidence": [
                        {
                            "source": "pubmed",
                            "source_id": "pmid:11",
                            "title": "Irrelevant anesthesia study",
                            "excerpt": "No relation to Blueprint Medicines.",
                        }
                    ],
                    "metadata": {},
                },
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["research_book"]["company_idea_count"], 0)
            self.assertEqual(payload["research_book"]["ideas"], [])

    def test_dashboard_prefers_snapshot_primary_event_and_filters_low_signal_program_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-04-04T20:09:34.781245+00:00"
            store.append_records(
                "signal_artifacts",
                [
                    {
                        "ticker": "RCKT",
                        "as_of": as_of,
                        "expected_return": -0.04,
                        "catalyst_success_prob": 0.5,
                        "confidence": 0.72,
                        "crowding_risk": 0.28,
                        "financing_risk": 0.22,
                        "thesis_horizon": "180d",
                        "primary_event_type": "phase2_readout",
                        "primary_event_bucket": "clinical",
                        "primary_event_status": "phase_timing_estimate",
                        "primary_event_date": "2026-10-01T00:00:00+00:00",
                        "primary_event_exact": False,
                        "company_state": "pre_commercial",
                        "setup_type": "watchful",
                        "rationale": json.dumps(["Legacy synthetic catalyst."]),
                        "supporting_evidence": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "portfolio_recommendations",
                [
                    {
                        "ticker": "RCKT",
                        "as_of": as_of,
                        "stance": "avoid",
                        "target_weight": 0.0,
                        "max_weight": 0.0,
                        "confidence": 0.72,
                        "scenario": "watchlist only",
                        "risk_flags": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "RCKT",
                        "company_name": "Rocket Pharmaceuticals",
                        "as_of": as_of,
                        "market_cap": 400_000_000,
                        "revenue": 0.0,
                    }
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "RCKT_2026-04-04T20-09-34.781245+00-00",
                {
                    "ticker": "RCKT",
                    "company_name": "Rocket Pharmaceuticals",
                    "as_of": as_of,
                    "market_cap": 400_000_000,
                    "enterprise_value": 380_000_000,
                    "revenue": 0.0,
                    "cash": 180_000_000,
                    "debt": 0.0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.45,
                    "programs": [
                        {
                            "program_id": "RCKT:1",
                            "name": "RP-L102",
                            "modality": "gene therapy",
                            "phase": "PHASE2",
                            "conditions": ["Fanconi Anemia Complementation Group A"],
                            "trials": [
                                {
                                    "trial_id": "NCT-RPL102",
                                    "title": "Lentiviral-mediated Gene Therapy for Pediatric Patients With Fanconi Anemia Subtype A",
                                    "phase": "PHASE2",
                                    "status": "ACTIVE_NOT_RECRUITING",
                                    "conditions": ["Fanconi Anemia Complementation Group A"],
                                    "interventions": ["RP-L102"],
                                    "enrollment": 12,
                                    "primary_outcomes": ["HSCT-free survival"],
                                    "locations": [],
                                }
                            ],
                            "pos_prior": 0.35,
                            "tam_estimate": 1_000_000_000,
                            "catalyst_events": [
                                {
                                    "event_id": "RCKT:1:phase2",
                                    "program_id": "RCKT:1",
                                    "event_type": "phase2_readout",
                                    "title": "RP-L102 phase 2 readout in Fanconi Anemia Complementation Group A",
                                    "expected_date": "2026-10-01",
                                    "horizon_days": 180,
                                    "probability": 0.6,
                                    "importance": 0.7,
                                    "crowdedness": 0.3,
                                    "status": "phase_timing_estimate",
                                }
                            ],
                            "evidence": [],
                        },
                        {
                            "program_id": "RCKT:2",
                            "name": "No intervention",
                            "modality": "platform",
                            "phase": "PHASE1",
                            "conditions": ["Danon Disease"],
                            "trials": [
                                {
                                    "trial_id": "NCT-NOHIT",
                                    "title": "Danon Disease Natural History Study",
                                    "phase": "PHASE1",
                                    "status": "RECRUITING",
                                    "conditions": ["Danon Disease"],
                                    "interventions": ["No intervention"],
                                    "enrollment": 60,
                                    "primary_outcomes": ["Left Ventricular Mass Index (LVMI) by echocardiogram"],
                                    "locations": [],
                                }
                            ],
                            "pos_prior": 0.1,
                            "tam_estimate": 500_000_000,
                            "catalyst_events": [
                                {
                                    "event_id": "RCKT:2:phase1",
                                    "program_id": "RCKT:2",
                                    "event_type": "phase1_readout",
                                    "title": "No intervention phase 1 update in Danon Disease",
                                    "expected_date": "2026-12-30",
                                    "horizon_days": 270,
                                    "probability": 0.4,
                                    "importance": 0.4,
                                    "crowdedness": 0.3,
                                    "status": "phase_timing_estimate",
                                }
                            ],
                            "evidence": [],
                        },
                    ],
                    "approved_products": [],
                    "catalyst_events": [
                        {
                            "event_id": "RCKT:fda",
                            "program_id": None,
                            "event_type": "clinical_readout",
                            "title": "Rocket Pharma Wins FDA Approval For KRESLADI",
                            "expected_date": "2026-03-27T11:04:31",
                            "horizon_days": 0,
                            "probability": 0.95,
                            "importance": 0.9,
                            "crowdedness": 0.3,
                            "status": "exact_press_release",
                        }
                    ],
                    "financing_events": [],
                    "evidence": [],
                    "metadata": {"company_state": "pre_commercial", "price_now": 7.25},
                },
            )
            store.append_records(
                "predictions",
                [
                    {
                        "entity_id": "RCKT:1",
                        "ticker": "RCKT",
                        "as_of": as_of,
                        "expected_return": 0.04,
                        "catalyst_success_prob": 0.52,
                        "confidence": 0.77,
                        "crowding_risk": 0.2,
                        "financing_risk": 0.2,
                        "thesis_horizon": "180d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps({"program_name": "RP-L102", "event_type": "phase2_readout", "phase": "PHASE2"}),
                    },
                    {
                        "entity_id": "RCKT:2",
                        "ticker": "RCKT",
                        "as_of": as_of,
                        "expected_return": -0.03,
                        "catalyst_success_prob": 0.44,
                        "confidence": 0.74,
                        "crowding_risk": 0.3,
                        "financing_risk": 0.2,
                        "thesis_horizon": "180d",
                        "model_name": "event_driven_ensemble",
                        "model_version": "v3",
                        "metadata": json.dumps({"program_name": "No intervention", "event_type": "phase1_readout", "phase": "PHASE1"}),
                    },
                ],
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["research_book"]["company_ideas"][0]["primary_event_type"], "regulatory_update")
            self.assertEqual(payload["research_book"]["program_idea_count"], 1)
            self.assertEqual(payload["research_book"]["program_ideas"][0]["program_name"], "RP-L102")

    def test_dashboard_validation_panel_uses_generated_at_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": "2026-04-03T18:30:00+00:00",
                    "rows": 120,
                    "windows": 12,
                    "rank_ic": 0.18,
                    "strict_rank_ic": 0.16,
                },
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            self.assertEqual(payload["validation"]["as_of"], "2026-04-03T18:30:00")
            self.assertEqual(payload["validation"]["freshness_days"], 1)

    def test_dashboard_merges_snapshot_risk_flags_and_dedupes_catalyst_radar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            as_of = "2026-04-04T20:08:38.019563+00:00"
            store.append_records(
                "signal_artifacts",
                [
                    {
                        "ticker": "ARVN",
                        "as_of": as_of,
                        "expected_return": 0.31,
                        "catalyst_success_prob": 0.66,
                        "confidence": 0.79,
                        "crowding_risk": 0.45,
                        "financing_risk": 0.21,
                        "thesis_horizon": "90d",
                        "primary_event_type": "pdufa",
                        "primary_event_bucket": "regulatory",
                        "primary_event_status": "exact_company_calendar",
                        "primary_event_date": "2026-06-05",
                        "primary_event_exact": True,
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                        "rationale": json.dumps(["PDUFA setup with narrowed market opportunity."]),
                        "supporting_evidence": json.dumps([]),
                    }
                ],
            )
            store.append_records(
                "portfolio_recommendations",
                [
                    {
                        "ticker": "ARVN",
                        "as_of": as_of,
                        "stance": "long",
                        "target_weight": 2.0,
                        "max_weight": 4.0,
                        "confidence": 0.79,
                        "scenario": "pre-catalyst long",
                        "risk_flags": json.dumps(["crowded catalyst"]),
                    }
                ],
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "ARVN",
                        "company_name": "Arvinas",
                        "as_of": as_of,
                        "market_cap": 900_000_000,
                        "revenue": 262_600_000,
                    }
                ],
            )
            store.write_raw_payload(
                "snapshots",
                "ARVN_2026-04-04T20-08-38.019563+00-00",
                {
                    "ticker": "ARVN",
                    "company_name": "Arvinas",
                    "as_of": as_of,
                    "market_cap": 900_000_000,
                    "enterprise_value": 850_000_000,
                    "revenue": 262_600_000,
                    "cash": 1_100_000_000,
                    "debt": 0,
                    "momentum_3mo": 0.0,
                    "trailing_6mo_return": 0.0,
                    "volatility": 0.35,
                    "programs": [
                        {
                            "program_id": "ARVN:1",
                            "name": "vepdegestrant",
                            "modality": "small molecule",
                            "phase": "PHASE3",
                            "conditions": ["ER+/HER2- advanced breast cancer with ESR1 mutation"],
                            "pos_prior": 0.62,
                            "tam_estimate": 2500000000,
                            "trials": [
                                {
                                    "trial_id": "NCT-ARV-471",
                                    "title": "VERITAC-2",
                                    "phase": "PHASE3",
                                    "status": "ACTIVE_NOT_RECRUITING",
                                    "conditions": ["ER+/HER2- advanced breast cancer with ESR1 mutation"],
                                    "interventions": ["vepdegestrant"],
                                    "enrollment": 420,
                                    "primary_outcomes": ["progression-free survival"],
                                    "locations": [],
                                }
                            ],
                            "catalyst_events": [
                                {
                                    "event_id": "ARVN:1:pdufa",
                                    "program_id": "ARVN:1",
                                    "event_type": "pdufa",
                                    "title": "Vepdegestrant PDUFA decision",
                                    "expected_date": "2026-06-05",
                                    "horizon_days": 62,
                                    "probability": 0.88,
                                    "importance": 0.98,
                                    "crowdedness": 0.42,
                                    "status": "exact_company_calendar",
                                }
                            ],
                            "evidence": [],
                        }
                    ],
                    "approved_products": [],
                    "catalyst_events": [
                        {
                            "event_id": "ARVN:company:pdufa",
                            "program_id": None,
                            "event_type": "pdufa",
                            "title": "Vepdegestrant PDUFA decision",
                            "expected_date": "2026-06-05",
                            "horizon_days": 62,
                            "probability": 0.88,
                            "importance": 0.98,
                            "crowdedness": 0.42,
                            "status": "exact_company_calendar",
                        }
                    ],
                    "financing_events": [],
                    "evidence": [],
                    "metadata": {
                        "company_state": "pre_commercial",
                        "price_now": 7.5,
                        "special_situation": "partner_search_overhang",
                        "special_situation_label": "partner search overhang",
                        "special_situation_reason": "Commercial partner has not yet been identified.",
                        "bear_case_flags": [
                            "seeking third-party commercialization partner for vepdegestrant",
                            "VERITAC-2 missed PFS significance in the overall intent-to-treat population",
                        ],
                    },
                },
            )

            payload = build_dashboard_payload(store=store, now=datetime(2026, 4, 4, tzinfo=timezone.utc))

            company_idea = payload["research_book"]["company_ideas"][0]
            self.assertEqual(company_idea["special_situation_label"], "partner search overhang")
            self.assertIn(
                "seeking third-party commercialization partner for vepdegestrant",
                company_idea["risk_flags"],
            )
            self.assertEqual(len(payload["research_book"]["catalyst_calendar"]), 1)
            catalyst = payload["research_book"]["catalyst_calendar"][0]
            self.assertEqual(catalyst["idea_key"], "company:ARVN")
            self.assertFalse(catalyst["in_current_plan"])
            self.assertEqual(catalyst["as_of"], "2026-04-04T20:08:38.019563")


if __name__ == "__main__":
    unittest.main()
