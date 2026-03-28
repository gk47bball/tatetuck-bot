import tempfile
import unittest
from unittest.mock import Mock

from biopharma_agent.vnext.eodhd import EODHDEventTapeClient, EODHDUniverseClient
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
            self.assertIn("clinical_readout", event_types)
            self.assertIn("exact_company_calendar", statuses)
            self.assertIn("exact_press_release", statuses)


if __name__ == "__main__":
    unittest.main()
