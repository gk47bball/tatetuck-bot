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
    "phase3_readout": 5,
    "phase2_readout": 4,
    "phase1_readout": 3,
    "clinical_readout": 3,
    "commercial_update": 2,
    "earnings": 1,
    "expected_financing": 0,
    "recent_offering_filing": 0,
}

EVENT_TYPE_BUCKET = {
    "pdufa": "regulatory",
    "phase3_readout": "clinical",
    "phase2_readout": "clinical",
    "phase1_readout": "clinical",
    "clinical_readout": "clinical",
    "commercial_update": "commercial",
    "earnings": "earnings",
    "expected_financing": "financing",
    "recent_offering_filing": "financing",
}

CLINICAL_EVENT_TYPES = {
    "pdufa",
    "phase3_readout",
    "phase2_readout",
    "phase1_readout",
    "clinical_readout",
}

SYNTHETIC_EVENT_STATUSES = {
    "phase_timing_estimate",
    "calendar_estimate",
    "estimated_from_revenue",
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


def event_type_bucket(event_type: str | None) -> str:
    if not event_type:
        return "none"
    return EVENT_TYPE_BUCKET.get(event_type, "other")


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
