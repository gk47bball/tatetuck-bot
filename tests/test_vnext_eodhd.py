import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from biopharma_agent.vnext.eodhd import EODHDEventTapeClient, EODHDUniverseClient
from biopharma_agent.vnext.sources import _classify_sec_filing_event
from biopharma_agent.vnext.storage import LocalResearchStore


class TestVNextEODHD(unittest.TestCase):
    def test_universe_client_syncs_active_and_delisted_rows(self):
        active_response = Mock()
        active_response.raise_for_status.return_value = None
        active_response.json.return_value = [
            {"Code": "TEST.US", "Name": "Test Bio", "Exchange": "NASDAQ", "Type": "Common Stock", "IPODate": "2020-01-01"}
        ]
        delisted_response = Mock()
        delisted_response.raise_for_status.return_value = None
        delisted_response.json.return_value = [
            {
                "Code": "OLD.US",
                "Name": "Old Bio",
                "Exchange": "NASDAQ",
                "Type": "Common Stock",
                "IPODate": "2018-01-01",
                "DelistedAt": "2024-09-15",
            }
        ]
        session = Mock()
        session.get.side_effect = [active_response, delisted_response]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            client = EODHDUniverseClient(store=store, api_key="test-key", session=session)
            summary = client.sync_universe_membership(exchanges=("NASDAQ",))
            membership = store.read_table("universe_membership")

            self.assertEqual(summary.active_rows, 1)
            self.assertEqual(summary.delisted_rows, 1)
            self.assertEqual(len(membership), 2)
            self.assertTrue(membership["is_delisted"].fillna(False).astype(bool).any())

    def test_event_tape_client_builds_exact_earnings_and_press_release_events(self):
        earnings_response = Mock()
        earnings_response.raise_for_status.return_value = None
        earnings_response.json.return_value = {
            "earnings": [
                {
                    "code": "TEST.US",
                    "report_date": "2025-11-05",
                    "before_after_market": "AfterMarket",
                    "estimate": 0.12,
                    "actual": 0.15,
                    "percent": 25.0,
                }
            ]
        }
        news_response = Mock()
        news_response.raise_for_status.return_value = None
        news_response.json.return_value = [
            {
                "date": "2025-11-04T12:03:00+00:00",
                "title": "Test Bio announces positive Phase 2 data",
                "content": "The company announces topline data from a phase 2 study.",
                "link": "https://example.com/press-release",
                "tags": ["press releases"],
            }
        ]
        session = Mock()
        session.get.side_effect = [earnings_response, news_response]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            client = EODHDEventTapeClient(store=store, api_key="test-key", session=session)
            payload = client.fetch_event_payload("TEST", as_of=__import__("datetime").datetime(2025, 11, 6))

            event_types = {item["event_type"] for item in payload["events"]}
            statuses = {item["status"] for item in payload["events"]}
            self.assertIn("earnings", event_types)
            self.assertIn("phase2_readout", event_types)
            self.assertIn("exact_company_calendar", statuses)
            self.assertIn("exact_press_release", statuses)

    def test_event_tape_client_ignores_news_rows_without_requested_symbol(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2025-11-04T12:03:00+00:00",
            "title": "Other Bio announces positive Phase 2 data",
            "content": "The company announces topline data from a phase 2 study.",
            "symbols": ["OTHR.US"],
            "link": "https://example.com/press-release",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "TEST.US")

        self.assertIsNone(classified)

    def test_event_tape_client_extracts_guided_strategic_close_dates(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2026-04-03T12:03:00+00:00",
            "title": "BioMarin announces acquisition of Amicus",
            "content": "The company announced a definitive agreement to acquire Amicus and expects the transaction to close in Q2 2026.",
            "symbols": ["BMRN.US"],
            "link": "https://example.com/amicus-deal",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "BMRN.US")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "strategic_transaction")
        self.assertEqual(classified["status"], "guided_company_event")
        self.assertTrue(classified["expected_date"].startswith("2026-06-30"))

    def test_event_tape_client_classifies_portfolio_repositioning_news(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2026-02-25T12:03:00+00:00",
            "title": "BioMarin announces voluntary withdrawal of ROCTAVIAN",
            "content": "The company will discontinue commercialization of ROCTAVIAN following a portfolio review.",
            "symbols": ["BMRN.US"],
            "link": "https://example.com/roctavian",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "BMRN.US")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "portfolio_repositioning")
        self.assertEqual(classified["status"], "exact_press_release")
        self.assertTrue(classified["expected_date"].startswith("2026-02-25"))

    def test_event_tape_client_classifies_guided_label_expansion_news(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2026-01-15T12:03:00+00:00",
            "title": "BioMarin announces supplemental NDA plan for VOXZOGO",
            "content": "The company expects a supplemental NDA submission and label expansion decision in 1H 2026.",
            "symbols": ["BMRN.US"],
            "link": "https://example.com/voxzogo-label",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "BMRN.US")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "label_expansion")
        self.assertEqual(classified["status"], "guided_company_event")
        self.assertTrue(classified["expected_date"].startswith("2026-06-30"))

    def test_event_tape_client_distinguishes_regulatory_progress_from_pdufa(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2026-03-02T12:03:00+00:00",
            "title": "Test Bio announces FDA acceptance and priority review for TB-101 NDA",
            "content": "The FDA accepted the NDA for review and granted priority review status.",
            "symbols": ["TEST.US"],
            "link": "https://example.com/fda-acceptance",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "TEST.US")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "regulatory_update")
        self.assertEqual(classified["status"], "exact_press_release")

    def test_event_tape_client_classifies_adcom_news_separately(self):
        client = EODHDEventTapeClient(api_key="test-key", session=Mock())
        row = {
            "date": "2026-03-20T18:00:00+00:00",
            "title": "FDA advisory committee votes in favor of TB-101 approval",
            "content": "The advisory committee voted 10-2 in favor of approval.",
            "symbols": ["TEST.US"],
            "link": "https://example.com/adcom",
            "tags": ["press releases"],
        }

        classified = client._classify_news_item(row, "TEST.US")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "adcom")

    def test_sec_classifier_treats_acceptance_as_regulatory_update_not_pdufa(self):
        filing = {
            "form": "8-K",
            "filing_date": "2026-03-02",
            "acceptance_datetime": "2026-03-02T21:05:00Z",
            "primary_doc_description": "Company announces FDA acceptance and priority review for TB-101 NDA",
            "primary_document": "acceptance.htm",
            "report_date": "2026-03-02",
            "url": "https://example.com/8k-acceptance",
        }

        classified = _classify_sec_filing_event("TEST", filing)

        self.assertIsNotNone(classified)
        self.assertEqual(classified["event_type"], "regulatory_update")

    def test_event_tape_client_recovers_exact_events_from_cached_news_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "eodhd_news",
                "APLS_2026-03-27_2026-04-03",
                [
                    {
                        "date": "2026-04-01T14:07:00+00:00",
                        "title": "Biogen to Acquire Apellis for $5.6B to Strengthen Immunology Portfolio",
                        "content": "Biogen announced it will acquire Apellis in an all-cash transaction.",
                        "symbols": ["APLS.US", "BIIB.US"],
                        "link": "https://example.com/apls-deal",
                        "tags": ["press releases"],
                    }
                ],
            )
            client = EODHDEventTapeClient(store=store, api_key="", session=Mock())

            payload = client.fetch_event_payload(
                "APLS",
                as_of=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(len(payload["events"]), 1)
            self.assertEqual(payload["events"][0]["event_type"], "strategic_transaction")
            self.assertEqual(payload["events"][0]["status"], "exact_press_release")


if __name__ == "__main__":
    unittest.main()
