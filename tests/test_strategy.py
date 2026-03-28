import unittest

from strategy import score_company


def make_company(
    *,
    market_cap: float,
    revenue: float,
    cash: float,
    debt: float,
    momentum_3mo: float,
    volatility: float,
    best_phase: str,
    conditions: list[str],
    trials: list[dict],
    total_enrollment: int,
    max_single_enrollment: int,
    num_papers: int,
    fda_serious_events: int = 0,
    fda_serious_ratio: float = 0.0,
) -> dict:
    phase_trial_counts: dict[str, int] = {}
    for trial in trials:
        phase_key = trial.get("phase_key", best_phase)
        phase_trial_counts[phase_key] = phase_trial_counts.get(phase_key, 0) + 1

    return {
        "ticker": "TEST",
        "company_name": "Test Co",
        "finance": {
            "marketCap": market_cap,
            "enterpriseValue": market_cap - cash + debt,
            "totalRevenue": revenue,
            "cash": cash,
            "debt": debt,
            "momentum_3mo": momentum_3mo,
            "volatility": volatility,
        },
        "trials": [
            {
                "title": trial["title"],
                "phase": trial["phase"],
                "conditions": trial["conditions"],
                "interventions": trial["interventions"],
                "enrollment": trial["enrollment"],
            }
            for trial in trials
        ],
        "num_trials": len(trials),
        "best_phase": best_phase,
        "phase_trial_counts": phase_trial_counts,
        "conditions": conditions,
        "total_enrollment": total_enrollment,
        "max_single_enrollment": max_single_enrollment,
        "num_papers": num_papers,
        "fda_serious_events": fda_serious_events,
        "fda_serious_ratio": fda_serious_ratio,
    }


class TestStrategy(unittest.TestCase):
    def test_archetypes_score_differ_meaningfully(self):
        phase3_oncology = make_company(
            market_cap=3_500_000_000,
            revenue=0,
            cash=900_000_000,
            debt=100_000_000,
            momentum_3mo=0.18,
            volatility=0.045,
            best_phase="PHASE3",
            conditions=["Clear Cell Renal Cell Carcinoma"],
            trials=[
                {
                    "title": "Lead RCC study",
                    "phase": ["Phase 3"],
                    "phase_key": "PHASE3",
                    "conditions": ["Clear Cell Renal Cell Carcinoma"],
                    "interventions": ["TKI-101"],
                    "enrollment": 620,
                },
                {
                    "title": "Expansion cohort",
                    "phase": ["Phase 2"],
                    "phase_key": "PHASE2",
                    "conditions": ["Clear Cell Renal Cell Carcinoma"],
                    "interventions": ["TKI-101"],
                    "enrollment": 140,
                },
            ],
            total_enrollment=760,
            max_single_enrollment=620,
            num_papers=18,
        )

        prerevenue_phase1 = make_company(
            market_cap=2_000_000_000,
            revenue=0,
            cash=350_000_000,
            debt=50_000_000,
            momentum_3mo=-0.08,
            volatility=0.07,
            best_phase="PHASE1",
            conditions=["Alpha-1 Antitrypsin Deficiency"],
            trials=[
                {
                    "title": "AATD dose escalation",
                    "phase": ["Phase 1"],
                    "phase_key": "PHASE1",
                    "conditions": ["Alpha-1 Antitrypsin Deficiency"],
                    "interventions": ["EDIT-001"],
                    "enrollment": 42,
                }
            ],
            total_enrollment=42,
            max_single_enrollment=42,
            num_papers=2,
        )

        commercial_stage = make_company(
            market_cap=6_000_000_000,
            revenue=500_000_000,
            cash=1_100_000_000,
            debt=150_000_000,
            momentum_3mo=0.12,
            volatility=0.03,
            best_phase="PHASE3",
            conditions=["Type 1 Diabetes"],
            trials=[
                {
                    "title": "Adjunctive T1D study",
                    "phase": ["Phase 3"],
                    "phase_key": "PHASE3",
                    "conditions": ["Type 1 Diabetes"],
                    "interventions": ["COM-101"],
                    "enrollment": 480,
                }
            ],
            total_enrollment=480,
            max_single_enrollment=480,
            num_papers=24,
        )

        phase3_result = score_company(phase3_oncology)
        prerevenue_result = score_company(prerevenue_phase1)
        commercial_result = score_company(commercial_stage)

        self.assertGreater(phase3_result["signal"], prerevenue_result["signal"])
        self.assertGreater(commercial_result["signal"], prerevenue_result["signal"])
        self.assertGreater(commercial_result["rnpv"], prerevenue_result["rnpv"])

    def test_crsp_like_and_beam_like_rnpv_diverge(self):
        crsp_like = make_company(
            market_cap=5_000_000_000,
            revenue=65_000_000,
            cash=1_800_000_000,
            debt=150_000_000,
            momentum_3mo=0.10,
            volatility=0.05,
            best_phase="PHASE3",
            conditions=["Sickle Cell Disease", "Transfusion-Dependent Beta-Thalassemia"],
            trials=[
                {
                    "title": "CASGEVY SCD",
                    "phase": ["Phase 3"],
                    "phase_key": "PHASE3",
                    "conditions": ["Sickle Cell Disease"],
                    "interventions": ["exa-cel"],
                    "enrollment": 120,
                },
                {
                    "title": "CASGEVY TDT",
                    "phase": ["Phase 3"],
                    "phase_key": "PHASE3",
                    "conditions": ["Transfusion-Dependent Beta-Thalassemia"],
                    "interventions": ["exa-cel"],
                    "enrollment": 95,
                },
                {
                    "title": "In vivo follow-on",
                    "phase": ["Phase 1"],
                    "phase_key": "PHASE1",
                    "conditions": ["Type 1 Diabetes"],
                    "interventions": ["CTX-001B"],
                    "enrollment": 28,
                },
            ],
            total_enrollment=243,
            max_single_enrollment=120,
            num_papers=30,
        )

        beam_like = make_company(
            market_cap=1_700_000_000,
            revenue=0,
            cash=1_100_000_000,
            debt=0,
            momentum_3mo=-0.02,
            volatility=0.06,
            best_phase="PHASE1",
            conditions=["Alpha-1 Antitrypsin Deficiency", "Glycogen Storage Disease Type Ia"],
            trials=[
                {
                    "title": "AATD base editing",
                    "phase": ["Phase 1"],
                    "phase_key": "PHASE1",
                    "conditions": ["Alpha-1 Antitrypsin Deficiency"],
                    "interventions": ["BEAM-302"],
                    "enrollment": 54,
                },
                {
                    "title": "GSD follow-on",
                    "phase": ["Phase 1"],
                    "phase_key": "PHASE1",
                    "conditions": ["Glycogen Storage Disease Type Ia"],
                    "interventions": ["BEAM-301"],
                    "enrollment": 36,
                },
            ],
            total_enrollment=90,
            max_single_enrollment=54,
            num_papers=11,
        )

        crsp_result = score_company(crsp_like)
        beam_result = score_company(beam_like)

        self.assertNotEqual(crsp_result["rnpv"], beam_result["rnpv"])
        self.assertGreater(crsp_result["rnpv"], beam_result["rnpv"])

    def test_signal_is_bounded(self):
        high_momentum = make_company(
            market_cap=900_000_000,
            revenue=20_000_000,
            cash=400_000_000,
            debt=20_000_000,
            momentum_3mo=1.5,
            volatility=0.01,
            best_phase="PHASE2",
            conditions=["Non-Hodgkin Lymphoma"],
            trials=[
                {
                    "title": "NHL registrational study",
                    "phase": ["Phase 2"],
                    "phase_key": "PHASE2",
                    "conditions": ["Non-Hodgkin Lymphoma"],
                    "interventions": ["ONC-200"],
                    "enrollment": 210,
                }
            ],
            total_enrollment=210,
            max_single_enrollment=210,
            num_papers=8,
            fda_serious_events=1,
            fda_serious_ratio=0.05,
        )

        result = score_company(high_momentum)
        self.assertGreaterEqual(result["signal"], -1.0)
        self.assertLessEqual(result["signal"], 1.0)

    def test_missing_market_cap_returns_zero_signal(self):
        missing_market_cap = make_company(
            market_cap=0,
            revenue=0,
            cash=100_000_000,
            debt=0,
            momentum_3mo=0.0,
            volatility=0.05,
            best_phase="PHASE1",
            conditions=["Muscular Dystrophy"],
            trials=[],
            total_enrollment=0,
            max_single_enrollment=0,
            num_papers=0,
        )

        result = score_company(missing_market_cap)
        self.assertEqual(result["signal"], 0.0)
        self.assertEqual(result["rnpv"], 0.0)
        self.assertEqual(result["conviction_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
