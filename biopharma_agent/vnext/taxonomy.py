from __future__ import annotations

PROGRAM_PHASE_EVENT_TYPE = {
    "APPROVED": "commercial_update",
    "NDA_BLA": "pdufa",
    "PHASE3": "phase3_readout",
    "PHASE2": "phase2_readout",
    "PHASE1": "phase1_readout",
    "EARLY_PHASE1": "phase1_readout",
}

EVENT_TYPE_PRIORITY = {
    "pdufa": 6,
    "adcom": 5,
    "phase3_readout": 5,
    "strategic_transaction": 4,
    "phase2_readout": 4,
    "label_expansion": 4,
    "regulatory_update": 4,
    "portfolio_repositioning": 3,
    "capital_allocation": 3,
    "phase1_readout": 3,
    "clinical_readout": 3,
    "commercial_update": 2,
    "earnings": 1,
    "expected_financing": 0,
    "recent_offering_filing": 0,
}

EVENT_TYPE_BUCKET = {
    "pdufa": "regulatory",
    "adcom": "regulatory",
    "phase3_readout": "clinical",
    "strategic_transaction": "strategic",
    "phase2_readout": "clinical",
    "label_expansion": "strategic",
    "regulatory_update": "regulatory",
    "portfolio_repositioning": "strategic",
    "capital_allocation": "strategic",
    "phase1_readout": "clinical",
    "clinical_readout": "clinical",
    "commercial_update": "commercial",
    "earnings": "earnings",
    "expected_financing": "financing",
    "recent_offering_filing": "financing",
}

CLINICAL_EVENT_TYPES = {
    "pdufa",
    "adcom",
    "phase3_readout",
    "phase2_readout",
    "phase1_readout",
    "clinical_readout",
}

SYNTHETIC_EVENT_STATUSES = {
    "phase_timing_estimate",
    "calendar_estimate",
    "estimated_from_revenue",
    # A synthetic event that has been flagged as stale because a recent SEC
    # 8-K filing suggests the trial already read out.  Still treated as
    # synthetic (not exact) so it does not inflate exact_primary_event_rate.
    "stale_synthetic",
}

EXACT_EVENT_STATUSES = {
    "exact_sec_filing",
    "exact_company_calendar",
    "exact_press_release",
}


def program_event_type_for_phase(phase: str) -> str:
    return PROGRAM_PHASE_EVENT_TYPE.get(phase, "clinical_readout")


def event_type_priority(event_type: str | None) -> int:
    if not event_type:
        return -1
    return EVENT_TYPE_PRIORITY.get(event_type, 0)


def _prefer_explicit_event_type(explicit_event_type: str | None, inferred_event_type: str) -> str:
    if not explicit_event_type:
        return inferred_event_type
    if event_type_priority(inferred_event_type) > event_type_priority(explicit_event_type):
        return inferred_event_type
    return explicit_event_type


def normalized_event_type(event_type: str | None, title: str | None = None, details: str | None = None) -> str | None:
    explicit_normalized: str | None = None
    if event_type:
        normalized = str(event_type).strip().lower().replace(" ", "_")
        if normalized in EVENT_TYPE_PRIORITY:
            explicit_normalized = normalized
    text = f"{title or ''} {details or ''}".lower()
    if any(
        marker in text
        for marker in (
            "label expansion",
            "expanded indication",
            "supplemental nda",
            "snda",
            "new indication",
        )
    ):
        return _prefer_explicit_event_type(explicit_normalized, "label_expansion")
    if any(
        marker in text
        for marker in (
            "acquisition",
            "acquire ",
            "acquires",
            "acquiring",
            "merger",
            "merge ",
            "merges",
            "buyout",
            "takeover",
            "definitive agreement",
            "m&a",
        )
    ):
        return _prefer_explicit_event_type(explicit_normalized, "strategic_transaction")
    if any(
        marker in text
        for marker in (
            "withdraw",
            "withdrawal",
            "withdraws",
            "discontinue",
            "discontinued",
            "discontinuation",
            "terminated",
            "termination",
            "divest",
            "divestiture",
            "strategic review",
            "out-license",
            "outlicense",
            "reprioritize",
            "deprioritize",
        )
    ):
        return _prefer_explicit_event_type(explicit_normalized, "portfolio_repositioning")
    if any(
        marker in text
        for marker in (
            "share repurchase",
            "buyback",
            "repurchase",
            "capital allocation",
            "deleveraging",
            "special dividend",
            "dividend increase",
        )
    ):
        return _prefer_explicit_event_type(explicit_normalized, "capital_allocation")
    if "adcom" in text or "advisory committee" in text:
        return _prefer_explicit_event_type(explicit_normalized, "adcom")
    if "pdufa" in text:
        return _prefer_explicit_event_type(explicit_normalized, "pdufa")
    if "target action date" in text or "action date" in text:
        return _prefer_explicit_event_type(explicit_normalized, "pdufa")
    if any(
        marker in text
        for marker in (
            "regulatory",
            "approval",
            "approved",
            "fda",
            "complete response",
            "crl",
            "priority review",
            "accepted for review",
            "acceptance for review",
            "filing accepted",
            "refuse to file",
            "rtf",
            "nda",
            "bla",
        )
    ):
        return _prefer_explicit_event_type(explicit_normalized, "regulatory_update")
    if "phase 3" in text or "phase iii" in text or "registrational" in text or "pivotal" in text:
        return _prefer_explicit_event_type(explicit_normalized, "phase3_readout")
    if "phase 2" in text or "phase ii" in text:
        return _prefer_explicit_event_type(explicit_normalized, "phase2_readout")
    if "phase 1" in text or "phase i" in text:
        return _prefer_explicit_event_type(explicit_normalized, "phase1_readout")
    if any(marker in text for marker in ("topline", "top-line", "readout", "clinical data", "interim data")):
        return _prefer_explicit_event_type(explicit_normalized, "clinical_readout")
    if "commercial" in text or "launch" in text or "revenue" in text:
        return _prefer_explicit_event_type(explicit_normalized, "commercial_update")
    if "earnings" in text or "quarter" in text:
        return _prefer_explicit_event_type(explicit_normalized, "earnings")
    return explicit_normalized


def event_type_bucket(event_type: str | None, title: str | None = None, details: str | None = None) -> str:
    normalized = normalized_event_type(event_type, title, details)
    if not normalized:
        return "none"
    return EVENT_TYPE_BUCKET.get(normalized, "other")


def is_clinical_event_type(event_type: str | None) -> bool:
    return bool(event_type in CLINICAL_EVENT_TYPES)


def is_synthetic_event(status: str | None, title: str | None = None) -> bool:
    status_lc = str(status or "").lower()
    title_lc = str(title or "").lower()
    if status_lc in SYNTHETIC_EVENT_STATUSES:
        return True
    return any(
        marker in title_lc
        for marker in (
            "next milestone",
            "estimated quarterly update",
            "quarterly commercial update",
            "reported commercial update",
        )
    )


def is_exact_timing_event(status: str | None, expected_date: str | None, title: str | None = None) -> bool:
    status_lc = str(status or "").lower()
    if status_lc in EXACT_EVENT_STATUSES:
        return bool(expected_date)
    return bool(expected_date) and not is_synthetic_event(status, title)


def event_timing_priority(status: str | None, expected_date: str | None, title: str | None = None) -> int:
    if is_exact_timing_event(status, expected_date, title):
        return 2
    if expected_date:
        return 1 if not is_synthetic_event(status, title) else 0
    return -1


def event_pm_priority(event_type: str | None, status: str | None, expected_date: str | None, title: str | None = None) -> int:
    return (event_type_priority(event_type) * 10) + max(event_timing_priority(status, expected_date, title), 0)
