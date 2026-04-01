"""
kol_proxy.py — Academic Medical Center (AMC) site quality scoring.

Replaces the need for proprietary KOL relationships by scoring trial
credibility from three public signals:
1. AMC site presence: Major academic hospitals have more rigorous trial
   conduct, stricter endpoint adjudication, and more credibility with FDA.
2. Multi-site diversity: Single-site trials have higher risk of
   investigator bias; 10+ sites indicates scalable recruitment.
3. International reach: FDA values multi-regional trials for broad
   label coverage.

Not a replacement for true KOL relationships, but the best available
public proxy for trial execution quality.
"""
from __future__ import annotations

# Tier 1: Major US academic medical centers with highest FDA credibility
# and most-cited clinical research (NCI-designated cancer centers, top NIH
# recipients). Presence of even one Tier 1 site is a meaningful signal.
TIER1_AMC_KEYWORDS = frozenset([
    "johns hopkins", "harvard", "massachusetts general", "mgh",
    "memorial sloan kettering", "msk", "md anderson",
    "mayo clinic", "ucsf", "university of california san francisco",
    "stanford", "university of pennsylvania", "upenn",
    "columbia university", "yale", "duke university",
    "university of michigan", "vanderbilt", "brigham and women",
    "cleveland clinic", "dana-farber", "dana farber",
    "university of chicago", "northwestern university",
    "university of colorado", "washington university",
    "university of washington", "emory university",
    "national institutes of health", "nih clinical center",
    "university of texas", "ut southwestern",
])

# Tier 2: Strong regional AMCs and international equivalents
TIER2_AMC_KEYWORDS = frozenset([
    "university hospital", "academic medical", "children's hospital",
    "royal college", "oxford university", "cambridge university",
    "karolinska", "charité", "charite",
    "university medical center", "cancer center",
    "research institute", "medical college",
    "cedars-sinai", "cedars sinai",
    "mount sinai", "houston methodist",
])


def score_trial_sites(locations: list[str]) -> dict[str, float]:
    """
    Score a trial's site quality from its list of location strings.

    Args:
        locations: List of location strings from ClinicalTrials.gov
                   (e.g., ["Massachusetts General Hospital, Boston, MA",
                            "Community Medical Center, Anytown, TX"])

    Returns dict with:
        amc_site_score: 0.0-1.0, presence and tier of AMC sites
        site_count_score: 0.0-1.0, scaled by number of sites (10+ = max)
        international_score: 0.0-1.0, presence of non-US sites
    """
    if not locations:
        return {"amc_site_score": 0.0, "site_count_score": 0.0, "international_score": 0.0}

    locs_lower = [loc.lower() for loc in locations]
    n_sites = len(locations)

    tier1_count = sum(
        1 for loc in locs_lower
        if any(kw in loc for kw in TIER1_AMC_KEYWORDS)
    )
    tier2_count = sum(
        1 for loc in locs_lower
        if not any(kw in loc for kw in TIER1_AMC_KEYWORDS)
        and any(kw in loc for kw in TIER2_AMC_KEYWORDS)
    )

    # Score: tier1 sites worth 2x tier2; diminishing returns above 3 sites
    raw_amc = min(tier1_count * 2 + tier2_count, 8)
    amc_site_score = min(raw_amc / 8.0, 1.0)

    # Site count: 1 site=0.1, 5 sites=0.5, 10+ sites=1.0
    site_count_score = min(n_sites / 10.0, 1.0)

    # International: look for non-US country names in locations
    us_indicators = ("united states", ", al", ", ak", ", az", ", ca", ", co",
                     ", fl", ", ga", ", il", ", ma", ", md", ", mi", ", mn",
                     ", mo", ", nc", ", nj", ", ny", ", oh", ", pa", ", tx",
                     ", va", ", wa", " usa")
    intl_count = sum(
        1 for loc in locs_lower
        if not any(ind in loc for ind in us_indicators)
    )
    international_score = min(intl_count / max(n_sites, 1), 1.0)

    return {
        "amc_site_score": float(amc_site_score),
        "site_count_score": float(site_count_score),
        "international_score": float(international_score),
    }


def program_site_quality(trials: list) -> dict[str, float]:
    """
    Aggregate site quality scores across all trials for a program.
    Trials with more enrollment get higher weight.
    """
    if not trials:
        return {"amc_site_score": 0.0, "site_count_score": 0.0, "international_score": 0.0}

    weighted: dict[str, float] = {"amc_site_score": 0.0, "site_count_score": 0.0, "international_score": 0.0}
    total_weight = 0.0
    for trial in trials:
        locations = getattr(trial, "locations", []) or []
        enrollment = max(float(getattr(trial, "enrollment", 0) or 0), 1.0)
        scores = score_trial_sites(locations)
        for key in weighted:
            weighted[key] += scores[key] * enrollment
        total_weight += enrollment

    if total_weight > 0:
        for key in weighted:
            weighted[key] /= total_weight
    return weighted
