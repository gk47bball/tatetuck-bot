from __future__ import annotations

from dataclasses import asdict, dataclass
import os

from dotenv import load_dotenv


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class VNextSettings:
    store_dir: str = ".tatetuck_store"
    eodhd_api_key: str | None = None
    sec_user_agent: str = "TatetuckBot/1.0 support@tatetuck.local"
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    alpaca_api_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_paper_account_id: str | None = None
    include_literature: bool = False
    min_snapshot_dates: int = 10
    min_matured_return_rows: int = 100
    min_walkforward_windows: int = 3
    max_snapshot_age_hours: int = 36
    min_archive_runs: int = 1
    max_gross_exposure_pct: float = 18.0
    max_single_position_pct: float = 4.0
    min_execution_weight_pct: float = 1.0
    min_execution_confidence: float = 0.60
    min_order_notional: float = 150.0
    max_new_positions: int = 6
    allow_blocked_paper_trading: bool = False
    simulated_paper_equity: float = 100000.0

    @classmethod
    def from_env(cls) -> "VNextSettings":
        load_dotenv()
        return cls(
            store_dir=os.environ.get("TATETUCK_STORE_DIR", ".tatetuck_store"),
            eodhd_api_key=os.environ.get("EODHD_API_KEY"),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", "TatetuckBot/1.0 support@tatetuck.local"),
            alpaca_api_key_id=os.environ.get("APCA_API_KEY_ID"),
            alpaca_api_secret_key=os.environ.get("APCA_API_SECRET_KEY"),
            alpaca_api_base_url=os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
            alpaca_paper_account_id=os.environ.get("APCA_PAPER_ACCOUNT_ID"),
            include_literature=_env_flag("TATETUCK_INCLUDE_LITERATURE", False),
            min_snapshot_dates=_env_int("TATETUCK_MIN_SNAPSHOT_DATES", 10),
            min_matured_return_rows=_env_int("TATETUCK_MIN_MATURED_RETURN_ROWS", 100),
            min_walkforward_windows=_env_int("TATETUCK_MIN_WALKFORWARD_WINDOWS", 3),
            max_snapshot_age_hours=_env_int("TATETUCK_MAX_SNAPSHOT_AGE_HOURS", 36),
            min_archive_runs=_env_int("TATETUCK_MIN_ARCHIVE_RUNS", 1),
            max_gross_exposure_pct=float(os.environ.get("TATETUCK_MAX_GROSS_EXPOSURE_PCT", "18.0")),
            max_single_position_pct=float(os.environ.get("TATETUCK_MAX_SINGLE_POSITION_PCT", "4.0")),
            min_execution_weight_pct=float(os.environ.get("TATETUCK_MIN_EXECUTION_WEIGHT_PCT", "1.0")),
            min_execution_confidence=float(os.environ.get("TATETUCK_MIN_EXECUTION_CONFIDENCE", "0.60")),
            min_order_notional=float(os.environ.get("TATETUCK_MIN_ORDER_NOTIONAL", "150.0")),
            max_new_positions=_env_int("TATETUCK_MAX_NEW_POSITIONS", 6),
            allow_blocked_paper_trading=_env_flag("TATETUCK_ALLOW_BLOCKED_PAPER_TRADING", False),
            simulated_paper_equity=float(os.environ.get("TATETUCK_SIMULATED_PAPER_EQUITY", "100000.0")),
        )

    def public_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["eodhd_api_key"] = bool(self.eodhd_api_key)
        payload["alpaca_api_key_id"] = bool(self.alpaca_api_key_id)
        payload["alpaca_api_secret_key"] = bool(self.alpaca_api_secret_key)
        return payload
