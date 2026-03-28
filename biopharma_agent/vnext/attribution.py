from __future__ import annotations

from collections.abc import Iterable


FACTOR_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "program_quality": ("program_quality_",),
    "catalyst_timing": ("catalyst_timing_",),
    "commercial_execution": ("commercial_execution_",),
    "balance_sheet": ("balance_sheet_",),
    "market_flow": ("market_flow_",),
    "state_profile": ("state_profile_",),
}

MOMENTUM_SENSITIVE_COLUMNS = {
    "market_flow_momentum_3mo",
    "commercial_execution_growth_signal",
}


def family_columns(columns: Iterable[str], family: str) -> list[str]:
    prefixes = FACTOR_FAMILY_PREFIXES.get(family, ())
    return [
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]


def momentum_sensitive_columns(columns: Iterable[str]) -> list[str]:
    return sorted(
        [
            column
            for column in columns
            if column in MOMENTUM_SENSITIVE_COLUMNS or column.startswith("market_flow_")
        ]
    )


def ablate_feature_family(features: dict[str, float], family: str) -> dict[str, float]:
    ablated = dict(features)
    for column in family_columns(features.keys(), family):
        ablated[column] = 0.0
    return ablated


def ablate_momentum(features: dict[str, float]) -> dict[str, float]:
    ablated = dict(features)
    for column in momentum_sensitive_columns(features.keys()):
        ablated[column] = 0.0
    return ablated
