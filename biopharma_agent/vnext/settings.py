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
    biopharmcatalyst_api_url: str | None = None
    biopharmcatalyst_api_key: str | None = None
    biopharmcatalyst_calendar_path: str | None = None
    discord_token: str | None = None
    discord_channel_id: str | None = None
    discord_trade_log_channel_id: str | None = None
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
    execution_hold_weight_pct: float = 0.75
    execution_hold_confidence: float = 0.50
    execution_rebalance_band_pct: float = 0.75
    execution_min_hard_catalyst_confidence: float = 0.62
    execution_min_soft_catalyst_confidence: float = 0.68
    execution_min_launch_confidence: float = 0.58
    execution_min_franchise_confidence: float = 0.56
    execution_min_internal_upside_pct: float = 0.08
    execution_min_floor_support_pct: float = 0.10
    execution_max_hard_catalyst_weight_pct: float = 4.0
    execution_max_soft_catalyst_weight_pct: float = 2.5
    execution_max_launch_weight_pct: float = 3.0
    execution_max_franchise_weight_pct: float = 3.0
    execution_max_floor_weight_pct: float = 1.5
    execution_validation_min_windows: int = 4
    execution_validation_min_rows: int = 12
    execution_validation_min_hit_rate: float = 0.45
    execution_validation_block_spread: float = 0.0
    execution_validation_caution_spread: float = 0.05
    execution_validation_caution_weight_scale: float = 0.5
    validation_max_age_days: int = 7
    evaluation_rebalance_spacing_days: int = 21
    evaluation_min_names_per_window: int = 3
    evaluation_turnover_book_weight_floor: float = 1.0
    evaluation_max_snapshot_staleness_days: int = 120
    rolling_validation_windows: int = 8
    catalyst_pre_event_min_rows: int = 80
    catalyst_pre_event_min_windows: int = 8
    catalyst_pre_event_min_rank_ic: float = 0.0
    catalyst_pre_event_min_net_spread: float = 0.0
    catalyst_pre_event_min_hit_rate: float = 0.5
    catalyst_pre_event_min_non_negative_trailing_windows: int = 2
    catalyst_post_event_min_rows: int = 80
    catalyst_post_event_min_windows: int = 8
    catalyst_post_event_min_rank_ic: float = 0.0
    catalyst_post_event_min_net_spread: float = 0.0
    catalyst_short_min_rows: int = 40
    catalyst_short_min_windows: int = 6
    catalyst_short_min_rank_ic: float = 0.0
    catalyst_short_min_net_spread: float = 0.0
    catalyst_family_min_rows: int = 20
    catalyst_family_min_windows: int = 6
    catalyst_exact_timing_rate_min: float = 0.6
    catalyst_exact_outcome_rate_min: float = 0.6
    allow_blocked_paper_trading: bool = False
    simulated_paper_equity: float = 100000.0
    execution_adv_pct_cap: float = 0.05  # max 5% of 20-day dollar ADV per order
    monitor_loop_interval_seconds: int = 300
    monitor_event_trigger_lookback_hours: int = 24
    monitor_event_trigger_forward_hours: int = 12
    monitor_snapshot_stale_hours: int = 18
    monitor_recent_decision_lookback_hours: int = 72
    monitor_max_symbols_per_cycle: int = 8
    monitor_event_watchlist_limit: int = 24

    @classmethod
    def from_env(cls) -> "VNextSettings":
        load_dotenv()
        return cls(
            store_dir=os.environ.get("TATETUCK_STORE_DIR", ".tatetuck_store"),
            eodhd_api_key=os.environ.get("EODHD_API_KEY"),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", "TatetuckBot/1.0 support@tatetuck.local"),
            biopharmcatalyst_api_url=os.environ.get("BIOPHARMCATALYST_API_URL"),
            biopharmcatalyst_api_key=os.environ.get("BIOPHARMCATALYST_API_KEY"),
            biopharmcatalyst_calendar_path=os.environ.get("BIOPHARMCATALYST_CALENDAR_PATH"),
            discord_token=os.environ.get("DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN"),
            discord_channel_id=os.environ.get("DISCORD_CHANNEL_ID") or os.environ.get("TATETUCK_DISCORD_CHANNEL_ID"),
            discord_trade_log_channel_id=os.environ.get("DISCORD_TRADE_LOG_CHANNEL_ID"),
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
            execution_hold_weight_pct=float(os.environ.get("TATETUCK_EXECUTION_HOLD_WEIGHT_PCT", "0.75")),
            execution_hold_confidence=float(os.environ.get("TATETUCK_EXECUTION_HOLD_CONFIDENCE", "0.50")),
            execution_rebalance_band_pct=float(os.environ.get("TATETUCK_EXECUTION_REBALANCE_BAND_PCT", "0.75")),
            execution_min_hard_catalyst_confidence=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_HARD_CATALYST_CONFIDENCE", "0.62")
            ),
            execution_min_soft_catalyst_confidence=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_SOFT_CATALYST_CONFIDENCE", "0.68")
            ),
            execution_min_launch_confidence=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_LAUNCH_CONFIDENCE", "0.58")
            ),
            execution_min_franchise_confidence=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_FRANCHISE_CONFIDENCE", "0.56")
            ),
            execution_min_internal_upside_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_INTERNAL_UPSIDE_PCT", "0.08")
            ),
            execution_min_floor_support_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MIN_FLOOR_SUPPORT_PCT", "0.10")
            ),
            execution_max_hard_catalyst_weight_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MAX_HARD_CATALYST_WEIGHT_PCT", "4.0")
            ),
            execution_max_soft_catalyst_weight_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MAX_SOFT_CATALYST_WEIGHT_PCT", "2.5")
            ),
            execution_max_launch_weight_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MAX_LAUNCH_WEIGHT_PCT", "3.0")
            ),
            execution_max_franchise_weight_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MAX_FRANCHISE_WEIGHT_PCT", "3.0")
            ),
            execution_max_floor_weight_pct=float(
                os.environ.get("TATETUCK_EXECUTION_MAX_FLOOR_WEIGHT_PCT", "1.5")
            ),
            execution_validation_min_windows=_env_int("TATETUCK_EXECUTION_VALIDATION_MIN_WINDOWS", 4),
            execution_validation_min_rows=_env_int("TATETUCK_EXECUTION_VALIDATION_MIN_ROWS", 12),
            execution_validation_min_hit_rate=float(
                os.environ.get("TATETUCK_EXECUTION_VALIDATION_MIN_HIT_RATE", "0.45")
            ),
            execution_validation_block_spread=float(
                os.environ.get("TATETUCK_EXECUTION_VALIDATION_BLOCK_SPREAD", "0.0")
            ),
            execution_validation_caution_spread=float(
                os.environ.get("TATETUCK_EXECUTION_VALIDATION_CAUTION_SPREAD", "0.05")
            ),
            execution_validation_caution_weight_scale=float(
                os.environ.get("TATETUCK_EXECUTION_VALIDATION_CAUTION_WEIGHT_SCALE", "0.5")
            ),
            validation_max_age_days=_env_int("TATETUCK_VALIDATION_MAX_AGE_DAYS", 7),
            evaluation_rebalance_spacing_days=_env_int("TATETUCK_EVAL_REBALANCE_SPACING_DAYS", 21),
            evaluation_min_names_per_window=_env_int("TATETUCK_EVAL_MIN_NAMES_PER_WINDOW", 3),
            evaluation_turnover_book_weight_floor=float(os.environ.get("TATETUCK_EVAL_TURNOVER_BOOK_WEIGHT_FLOOR", "1.0")),
            evaluation_max_snapshot_staleness_days=_env_int("TATETUCK_EVAL_MAX_SNAPSHOT_STALENESS_DAYS", 120),
            rolling_validation_windows=_env_int("TATETUCK_ROLLING_VALIDATION_WINDOWS", 8),
            catalyst_pre_event_min_rows=_env_int("TATETUCK_CATALYST_PRE_EVENT_MIN_ROWS", 80),
            catalyst_pre_event_min_windows=_env_int("TATETUCK_CATALYST_PRE_EVENT_MIN_WINDOWS", 8),
            catalyst_pre_event_min_rank_ic=float(os.environ.get("TATETUCK_CATALYST_PRE_EVENT_MIN_RANK_IC", "0.0")),
            catalyst_pre_event_min_net_spread=float(os.environ.get("TATETUCK_CATALYST_PRE_EVENT_MIN_NET_SPREAD", "0.0")),
            catalyst_pre_event_min_hit_rate=float(os.environ.get("TATETUCK_CATALYST_PRE_EVENT_MIN_HIT_RATE", "0.5")),
            catalyst_pre_event_min_non_negative_trailing_windows=_env_int(
                "TATETUCK_CATALYST_PRE_EVENT_MIN_NON_NEGATIVE_TRAILING_WINDOWS",
                2,
            ),
            catalyst_post_event_min_rows=_env_int("TATETUCK_CATALYST_POST_EVENT_MIN_ROWS", 80),
            catalyst_post_event_min_windows=_env_int("TATETUCK_CATALYST_POST_EVENT_MIN_WINDOWS", 8),
            catalyst_post_event_min_rank_ic=float(os.environ.get("TATETUCK_CATALYST_POST_EVENT_MIN_RANK_IC", "0.0")),
            catalyst_post_event_min_net_spread=float(os.environ.get("TATETUCK_CATALYST_POST_EVENT_MIN_NET_SPREAD", "0.0")),
            catalyst_short_min_rows=_env_int("TATETUCK_CATALYST_SHORT_MIN_ROWS", 40),
            catalyst_short_min_windows=_env_int("TATETUCK_CATALYST_SHORT_MIN_WINDOWS", 6),
            catalyst_short_min_rank_ic=float(os.environ.get("TATETUCK_CATALYST_SHORT_MIN_RANK_IC", "0.0")),
            catalyst_short_min_net_spread=float(os.environ.get("TATETUCK_CATALYST_SHORT_MIN_NET_SPREAD", "0.0")),
            catalyst_family_min_rows=_env_int("TATETUCK_CATALYST_FAMILY_MIN_ROWS", 20),
            catalyst_family_min_windows=_env_int("TATETUCK_CATALYST_FAMILY_MIN_WINDOWS", 6),
            catalyst_exact_timing_rate_min=float(os.environ.get("TATETUCK_CATALYST_EXACT_TIMING_RATE_MIN", "0.6")),
            catalyst_exact_outcome_rate_min=float(os.environ.get("TATETUCK_CATALYST_EXACT_OUTCOME_RATE_MIN", "0.6")),
            allow_blocked_paper_trading=_env_flag("TATETUCK_ALLOW_BLOCKED_PAPER_TRADING", False),
            simulated_paper_equity=float(os.environ.get("TATETUCK_SIMULATED_PAPER_EQUITY", "100000.0")),
            execution_adv_pct_cap=float(os.environ.get("TATETUCK_EXECUTION_ADV_PCT_CAP", "0.05")),
            monitor_loop_interval_seconds=_env_int("TATETUCK_MONITOR_LOOP_INTERVAL_SECONDS", 300),
            monitor_event_trigger_lookback_hours=_env_int("TATETUCK_MONITOR_EVENT_TRIGGER_LOOKBACK_HOURS", 24),
            monitor_event_trigger_forward_hours=_env_int("TATETUCK_MONITOR_EVENT_TRIGGER_FORWARD_HOURS", 12),
            monitor_snapshot_stale_hours=_env_int("TATETUCK_MONITOR_SNAPSHOT_STALE_HOURS", 18),
            monitor_recent_decision_lookback_hours=_env_int("TATETUCK_MONITOR_RECENT_DECISION_LOOKBACK_HOURS", 72),
            monitor_max_symbols_per_cycle=_env_int("TATETUCK_MONITOR_MAX_SYMBOLS_PER_CYCLE", 8),
            monitor_event_watchlist_limit=_env_int("TATETUCK_MONITOR_EVENT_WATCHLIST_LIMIT", 24),
        )

    def public_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["eodhd_api_key"] = bool(self.eodhd_api_key)
        payload["discord_token"] = bool(self.discord_token)
        payload["discord_channel_id"] = bool(self.discord_channel_id)
        payload["discord_trade_log_channel_id"] = bool(self.discord_trade_log_channel_id)
        payload["alpaca_api_key_id"] = bool(self.alpaca_api_key_id)
        payload["alpaca_api_secret_key"] = bool(self.alpaca_api_secret_key)
        return payload
