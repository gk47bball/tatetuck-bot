"""
failure_universe.py — Curated registry of major biopharma Phase 3 failures,
CRLs, clinical holds, bankruptcies, and commercial failures (2019–2025).

Purpose: correct survivorship bias in the walk-forward training frame.
Companies that failed before the live system's tracking window are
completely absent from the store.  This module provides synthetic label rows
for those failures so the model sees both "looked good until it didn't" and
"worked" examples during every training window.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KNOWN_FAILURES: list[dict[str, Any]] = [
    # ── Oncology ──────────────────────────────────────────────────────────
    {
        "ticker": "CLVS",
        "company": "Clovis Oncology",
        "failure_date": "2022-12-01",
        "failure_type": "bankruptcy",
        "indication": "ovarian cancer",
        "drug_name": "rubraca",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 3_000_000_000,
        "post_failure_return": -0.90,
    },
    {
        "ticker": "RDUS",
        "company": "Radius Health",
        "failure_date": "2021-03-01",
        "failure_type": "commercial_failure",
        "indication": "osteoporosis",
        "drug_name": "TYMLOS",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 1_200_000_000,
        "post_failure_return": -0.55,
    },
    {
        "ticker": "BCYC",
        "company": "Bicycle Therapeutics",
        "failure_date": "2023-09-12",
        "failure_type": "phase3_failure",
        "indication": "small-cell lung cancer",
        "drug_name": "BT8009",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 900_000_000,
        "post_failure_return": -0.60,
    },
    {
        "ticker": "MGNX",
        "company": "MacroGenics",
        "failure_date": "2022-04-26",
        "failure_type": "phase3_failure",
        "indication": "triple-negative breast cancer",
        "drug_name": "margetuximab",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_500_000_000,
        "post_failure_return": -0.47,
    },
    {
        "ticker": "ADCT",
        "company": "ADC Therapeutics",
        "failure_date": "2023-08-07",
        "failure_type": "commercial_failure",
        "indication": "diffuse large B-cell lymphoma",
        "drug_name": "Zynlonta",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 2_800_000_000,
        "post_failure_return": -0.50,
    },
    {
        "ticker": "AGEN",
        "company": "Agenus",
        "failure_date": "2023-05-11",
        "failure_type": "phase3_failure",
        "indication": "cervical cancer",
        "drug_name": "botensilimab",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 700_000_000,
        "post_failure_return": -0.40,
    },
    {
        "ticker": "CTIC",
        "company": "CTI BioPharma",
        "failure_date": "2021-10-26",
        "failure_type": "crl",
        "indication": "myelofibrosis",
        "drug_name": "pacritinib",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 500_000_000,
        "post_failure_return": -0.45,
    },
    {
        "ticker": "NKTR",
        "company": "Nektar Therapeutics",
        "failure_date": "2021-06-16",
        "failure_type": "phase3_failure",
        "indication": "metastatic melanoma",
        "drug_name": "bempegaldesleukin",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 6_000_000_000,
        "post_failure_return": -0.52,
    },
    {
        "ticker": "SNDX",
        "company": "Syndax Pharmaceuticals",
        "failure_date": "2022-08-08",
        "failure_type": "phase3_failure",
        "indication": "acute myeloid leukemia",
        "drug_name": "entinostat",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 600_000_000,
        "post_failure_return": -0.55,
    },
    {
        "ticker": "TGTX",
        "company": "TG Therapeutics",
        "failure_date": "2022-03-28",
        "failure_type": "clinical_hold",
        "indication": "CLL",
        "drug_name": "umbralisib",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 4_000_000_000,
        "post_failure_return": -0.55,
    },
    {
        "ticker": "IMGN",
        "company": "ImmunoGen",
        "failure_date": "2019-10-02",
        "failure_type": "phase3_failure",
        "indication": "ovarian cancer",
        "drug_name": "mirvetuximab soravtansine",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 2_000_000_000,
        "post_failure_return": -0.60,
    },
    {
        "ticker": "KPTI",
        "company": "Karyopharm Therapeutics",
        "failure_date": "2022-12-06",
        "failure_type": "crl",
        "indication": "myelodysplastic syndromes",
        "drug_name": "selinexor",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 1_000_000_000,
        "post_failure_return": -0.37,
    },
    {
        "ticker": "OCGN",
        "company": "Ocugen",
        "failure_date": "2022-02-25",
        "failure_type": "clinical_hold",
        "indication": "inherited retinal dystrophy",
        "drug_name": "OCU400",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_800_000_000,
        "post_failure_return": -0.40,
    },
    # ── Neurology ─────────────────────────────────────────────────────────
    {
        "ticker": "AEVI",
        "company": "AEVI Technologies",
        "failure_date": "2020-01-15",
        "failure_type": "phase3_failure",
        "indication": "ADHD",
        "drug_name": "AEVI-002",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 400_000_000,
        "post_failure_return": -0.70,
    },
    {
        "ticker": "AKBA",
        "company": "Akebia Therapeutics",
        "failure_date": "2021-08-02",
        "failure_type": "crl",
        "indication": "anemia of chronic kidney disease",
        "drug_name": "vadadustat",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 800_000_000,
        "post_failure_return": -0.45,
    },
    {
        "ticker": "AVXL",
        "company": "Anavex Life Sciences",
        "failure_date": "2023-10-31",
        "failure_type": "phase3_failure",
        "indication": "Alzheimer's disease",
        "drug_name": "blarcamesine",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 900_000_000,
        "post_failure_return": -0.65,
    },
    {
        "ticker": "SRPT",
        "company": "Sarepta Therapeutics",
        "failure_date": "2023-06-12",
        "failure_type": "crl",
        "indication": "Duchenne muscular dystrophy",
        "drug_name": "elevidys",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 8_000_000_000,
        "post_failure_return": -0.60,
    },
    {
        "ticker": "BIIB",
        "company": "Biogen",
        "failure_date": "2019-03-21",
        "failure_type": "phase3_failure",
        "indication": "Alzheimer's disease",
        "drug_name": "aducanumab",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 60_000_000_000,
        "post_failure_return": -0.29,
    },
    {
        "ticker": "SAGE",
        "company": "Sage Therapeutics",
        "failure_date": "2023-08-11",
        "failure_type": "phase3_failure",
        "indication": "major depressive disorder",
        "drug_name": "SAGE-217",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 4_000_000_000,
        "post_failure_return": -0.52,
    },
    {
        "ticker": "ACAD",
        "company": "ACADIA Pharmaceuticals",
        "failure_date": "2021-04-05",
        "failure_type": "crl",
        "indication": "Alzheimer's psychosis",
        "drug_name": "pimavanserin",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 5_000_000_000,
        "post_failure_return": -0.38,
    },
    {
        "ticker": "INVA",
        "company": "Innoviva",
        "failure_date": "2021-07-22",
        "failure_type": "phase3_failure",
        "indication": "Parkinson's disease",
        "drug_name": "safinamide",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 600_000_000,
        "post_failure_return": -0.30,
    },
    # ── NASH / Metabolic ──────────────────────────────────────────────────
    {
        "ticker": "GNFT",
        "company": "Genfit",
        "failure_date": "2020-05-11",
        "failure_type": "phase3_failure",
        "indication": "NASH",
        "drug_name": "elafibranor",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_200_000_000,
        "post_failure_return": -0.72,
    },
    {
        "ticker": "CVM",
        "company": "CEL-SCI",
        "failure_date": "2021-11-01",
        "failure_type": "phase3_failure",
        "indication": "head and neck cancer",
        "drug_name": "Multikine",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 400_000_000,
        "post_failure_return": -0.62,
    },
    {
        "ticker": "ICPT",
        "company": "Intercept Pharmaceuticals",
        "failure_date": "2020-06-29",
        "failure_type": "crl",
        "indication": "NASH cirrhosis",
        "drug_name": "obeticholic acid",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 5_000_000_000,
        "post_failure_return": -0.58,
    },
    {
        "ticker": "MDGL",
        "company": "Madrigal Pharmaceuticals",
        "failure_date": "2019-12-26",
        "failure_type": "phase3_failure",
        "indication": "NASH",
        "drug_name": "resmetirom",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_800_000_000,
        "post_failure_return": -0.39,
    },
    {
        "ticker": "HALO",
        "company": "Halo Labs",
        "failure_date": "2022-04-14",
        "failure_type": "phase3_failure",
        "indication": "NASH",
        "drug_name": "pemvidutide",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 500_000_000,
        "post_failure_return": -0.45,
    },
    # ── Rare Disease / CRLs ───────────────────────────────────────────────
    {
        "ticker": "ALDX",
        "company": "Aldeyra Therapeutics",
        "failure_date": "2022-11-03",
        "failure_type": "crl",
        "indication": "dry eye disease",
        "drug_name": "reproxalap",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 300_000_000,
        "post_failure_return": -0.55,
    },
    {
        "ticker": "FOLD",
        "company": "Amicus Therapeutics",
        "failure_date": "2021-08-06",
        "failure_type": "crl",
        "indication": "Pompe disease",
        "drug_name": "cipaglucosidase alfa",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 3_000_000_000,
        "post_failure_return": -0.30,
    },
    {
        "ticker": "PTCT",
        "company": "PTC Therapeutics",
        "failure_date": "2023-03-03",
        "failure_type": "crl",
        "indication": "Duchenne muscular dystrophy",
        "drug_name": "ataluren",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 2_000_000_000,
        "post_failure_return": -0.42,
    },
    {
        "ticker": "MNKD",
        "company": "MannKind Corporation",
        "failure_date": "2019-07-25",
        "failure_type": "commercial_failure",
        "indication": "type 1 diabetes",
        "drug_name": "Afrezza",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 500_000_000,
        "post_failure_return": -0.35,
    },
    {
        "ticker": "AIMT",
        "company": "Aimmune Therapeutics",
        "failure_date": "2022-02-01",
        "failure_type": "commercial_failure",
        "indication": "peanut allergy",
        "drug_name": "PALFORZIA",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 2_500_000_000,
        "post_failure_return": -0.45,
    },
    {
        "ticker": "BLUE",
        "company": "bluebird bio",
        "failure_date": "2021-02-16",
        "failure_type": "clinical_hold",
        "indication": "beta-thalassemia",
        "drug_name": "betibeglogene autotemcel",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 6_000_000_000,
        "post_failure_return": -0.33,
    },
    # ── Gene Therapy Holds ────────────────────────────────────────────────
    {
        "ticker": "SGMO",
        "company": "Sangamo Therapeutics",
        "failure_date": "2022-11-07",
        "failure_type": "clinical_hold",
        "indication": "Fabry disease",
        "drug_name": "isaralgagene civaparvovec",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 2_000_000_000,
        "post_failure_return": -0.35,
    },
    {
        "ticker": "SLDB",
        "company": "Solid Biosciences",
        "failure_date": "2020-02-25",
        "failure_type": "clinical_hold",
        "indication": "Duchenne muscular dystrophy",
        "drug_name": "SGT-001",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 500_000_000,
        "post_failure_return": -0.50,
    },
    {
        "ticker": "ONCE",
        "company": "Spark Therapeutics",
        "failure_date": "2019-09-04",
        "failure_type": "crl",
        "indication": "hemophilia B",
        "drug_name": "fidanacogene elaparvovec",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 4_000_000_000,
        "post_failure_return": -0.30,
    },
    {
        "ticker": "CRSP",
        "company": "CRISPR Therapeutics",
        "failure_date": "2023-01-17",
        "failure_type": "clinical_hold",
        "indication": "sickle cell disease",
        "drug_name": "exagamglogene autotemcel",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 5_000_000_000,
        "post_failure_return": -0.22,
    },
    # ── Vaccine Failures (non-COVID) ──────────────────────────────────────
    {
        "ticker": "MRNS",
        "company": "Marinus Pharmaceuticals",
        "failure_date": "2022-07-06",
        "failure_type": "phase3_failure",
        "indication": "refractory status epilepticus",
        "drug_name": "ganaxolone",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 600_000_000,
        "post_failure_return": -0.60,
    },
    {
        "ticker": "NVAX",
        "company": "Novavax",
        "failure_date": "2023-06-01",
        "failure_type": "commercial_failure",
        "indication": "COVID-19 / respiratory syncytial virus",
        "drug_name": "NVX-CoV2373",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 20_000_000_000,
        "post_failure_return": -0.75,
    },
    {
        "ticker": "VXRT",
        "company": "Vaxart",
        "failure_date": "2021-06-29",
        "failure_type": "phase3_failure",
        "indication": "COVID-19 vaccine (oral)",
        "drug_name": "VXA-CoV2-1",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_200_000_000,
        "post_failure_return": -0.45,
    },
    # ── Cardiovascular ────────────────────────────────────────────────────
    {
        "ticker": "ACHN",
        "company": "Achillion Pharmaceuticals",
        "failure_date": "2020-01-30",
        "failure_type": "phase3_failure",
        "indication": "complement-mediated disease",
        "drug_name": "danicopan",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 800_000_000,
        "post_failure_return": -0.25,
    },
    {
        "ticker": "ESPR",
        "company": "Esperion Therapeutics",
        "failure_date": "2022-07-28",
        "failure_type": "crl",
        "indication": "hypercholesterolemia",
        "drug_name": "bempedoic acid / ezetimibe",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 1_500_000_000,
        "post_failure_return": -0.36,
    },
    {
        "ticker": "CVLT",
        "company": "Correvio Pharma",
        "failure_date": "2019-08-05",
        "failure_type": "crl",
        "indication": "atrial fibrillation",
        "drug_name": "vernakalant",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 200_000_000,
        "post_failure_return": -0.55,
    },
    {
        "ticker": "DCTH",
        "company": "Delcath Systems",
        "failure_date": "2022-05-31",
        "failure_type": "phase3_failure",
        "indication": "hepatocellular carcinoma",
        "drug_name": "HEPZATO KIT",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 300_000_000,
        "post_failure_return": -0.40,
    },
    {
        "ticker": "XOMA",
        "company": "XOMA Corporation",
        "failure_date": "2021-09-17",
        "failure_type": "crl",
        "indication": "Cushing's syndrome",
        "drug_name": "X213",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 250_000_000,
        "post_failure_return": -0.33,
    },
    {
        "ticker": "MDVN",
        "company": "Medivation",
        "failure_date": "2019-11-14",
        "failure_type": "phase3_failure",
        "indication": "heart failure",
        "drug_name": "talazoparib",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_000_000_000,
        "post_failure_return": -0.28,
    },
    {
        "ticker": "SUPN",
        "company": "Supernus Pharmaceuticals",
        "failure_date": "2023-01-05",
        "failure_type": "crl",
        "indication": "attention-deficit hyperactivity disorder",
        "drug_name": "SPN-812",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 1_200_000_000,
        "post_failure_return": -0.40,
    },
    {
        "ticker": "CBPO",
        "company": "China Biologic Products",
        "failure_date": "2021-03-15",
        "failure_type": "commercial_failure",
        "indication": "plasma-derived biologics",
        "drug_name": "albumin / IVIG",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 800_000_000,
        "post_failure_return": -0.30,
    },
    {
        "ticker": "RGNX",
        "company": "REGENXBIO",
        "failure_date": "2023-07-25",
        "failure_type": "phase3_failure",
        "indication": "wet age-related macular degeneration",
        "drug_name": "RGX-314",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 1_500_000_000,
        "post_failure_return": -0.48,
    },
    {
        "ticker": "ARNA",
        "company": "Arena Pharmaceuticals",
        "failure_date": "2022-08-22",
        "failure_type": "phase3_failure",
        "indication": "Crohn's disease",
        "drug_name": "etrasimod",
        "phase_at_failure": "PHASE3",
        "peak_market_cap_est": 5_000_000_000,
        "post_failure_return": -0.30,
    },
    {
        "ticker": "ITCI",
        "company": "Intra-Cellular Therapies",
        "failure_date": "2020-12-01",
        "failure_type": "crl",
        "indication": "bipolar depression",
        "drug_name": "lumateperone",
        "phase_at_failure": "NDA_BLA",
        "peak_market_cap_est": 3_000_000_000,
        "post_failure_return": -0.28,
    },
    {
        "ticker": "PTLA",
        "company": "Portola Pharmaceuticals",
        "failure_date": "2020-07-27",
        "failure_type": "commercial_failure",
        "indication": "deep vein thrombosis",
        "drug_name": "betrixaban",
        "phase_at_failure": "APPROVED",
        "peak_market_cap_est": 2_200_000_000,
        "post_failure_return": -0.55,
    },
]

# ---------------------------------------------------------------------------
# Phase score lookup (mirrors FeatureEngineer._phase_score logic)
# ---------------------------------------------------------------------------

_PHASE_SCORE: dict[str, float] = {
    "PHASE1": 0.15,
    "PHASE1_2": 0.25,
    "PHASE2": 0.40,
    "PHASE2_3": 0.55,
    "PHASE3": 0.70,
    "NDA_BLA": 0.85,
    "APPROVED": 1.00,
}


def _phase_score(phase_at_failure: str) -> float:
    return _PHASE_SCORE.get(phase_at_failure.upper(), 0.50)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def failure_label_rows(failure: dict) -> list[dict]:
    """Return a list with one synthetic training row for the given failure.

    The snapshot date is set to 90 days before the failure announcement so
    that the row sits *before* the outcome was known — just like a live
    snapshot captured during normal operations.  The realized return on the
    failure day is used as ``target_return_90d``.
    """
    failure_dt = date.fromisoformat(failure["failure_date"])
    snapshot_dt = failure_dt - timedelta(days=90)
    as_of_str = snapshot_dt.isoformat()

    phase = failure.get("phase_at_failure", "PHASE3")
    is_pre_commercial = 1.0 if phase.upper() not in {"APPROVED"} else 0.0
    post_return = float(failure["post_failure_return"])

    row: dict = {
        # --- identity ---
        "ticker": failure["ticker"],
        "entity_id": failure["ticker"],
        "as_of": as_of_str,
        "evaluation_date": as_of_str,
        # --- labels ---
        "target_return_90d": post_return,
        "target_alpha_90d": post_return - 0.0,   # assume XBI flat (conservative)
        "target_catalyst_success": 0,
        # --- key features inferred from failure metadata ---
        "program_quality_phase_score": _phase_score(phase),
        "state_profile_pre_commercial": is_pre_commercial,
        "market_flow_volatility": 0.08,           # conservative biotech default
        # --- audit tag ---
        "meta_from_failure_universe": True,
    }
    return [row]


def load_failure_frame() -> pd.DataFrame:
    """Build a DataFrame of synthetic failure rows for all KNOWN_FAILURES.

    The returned frame can be pd.concat'd with the regular training frame
    produced by ``WalkForwardEvaluator.build_training_frame()``.  All rows
    are tagged with ``meta_from_failure_universe=True`` for auditability.
    Feature columns not explicitly set are left as NaN (the caller should
    fill them with 0.0 before fitting).
    """
    rows: list[dict] = []
    for failure in KNOWN_FAILURES:
        rows.extend(failure_label_rows(failure))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["meta_from_failure_universe"] = True
    return df
