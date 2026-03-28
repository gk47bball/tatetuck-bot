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
    store_dir: str
    eodhd_api_key: str | None
    sec_user_agent: str
    include_literature: bool
    min_snapshot_dates: int
    min_matured_return_rows: int
    min_walkforward_windows: int
    max_snapshot_age_hours: int
    min_archive_runs: int

    @classmethod
    def from_env(cls) -> "VNextSettings":
        load_dotenv()
        return cls(
            store_dir=os.environ.get("TATETUCK_STORE_DIR", ".tatetuck_store"),
            eodhd_api_key=os.environ.get("EODHD_API_KEY"),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", "TatetuckBot/1.0 support@tatetuck.local"),
            include_literature=_env_flag("TATETUCK_INCLUDE_LITERATURE", False),
            min_snapshot_dates=_env_int("TATETUCK_MIN_SNAPSHOT_DATES", 10),
            min_matured_return_rows=_env_int("TATETUCK_MIN_MATURED_RETURN_ROWS", 100),
            min_walkforward_windows=_env_int("TATETUCK_MIN_WALKFORWARD_WINDOWS", 3),
            max_snapshot_age_hours=_env_int("TATETUCK_MAX_SNAPSHOT_AGE_HOURS", 36),
            min_archive_runs=_env_int("TATETUCK_MIN_ARCHIVE_RUNS", 1),
        )

    def public_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["eodhd_api_key"] = bool(self.eodhd_api_key)
        return payload
