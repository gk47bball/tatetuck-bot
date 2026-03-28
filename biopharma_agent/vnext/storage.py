from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .entities import CompanySnapshot, ExperimentRecord, FeatureVector, ModelPrediction

try:
    import duckdb
except Exception:  # pragma: no cover - optional dependency at runtime
    duckdb = None


DEFAULT_STORE_DIR = ".tatetuck_store"


class LocalResearchStore:
    """Local-first research store using Parquet tables and raw JSON payloads."""

    def __init__(self, base_dir: str | os.PathLike[str] | None = None):
        self.base_dir = Path(base_dir or DEFAULT_STORE_DIR)
        self.raw_dir = self.base_dir / "raw"
        self.tables_dir = self.base_dir / "tables"
        self.models_dir = self.base_dir / "models"
        self.experiments_dir = self.base_dir / "experiments"
        for path in (self.raw_dir, self.tables_dir, self.models_dir, self.experiments_dir):
            path.mkdir(parents=True, exist_ok=True)

    def write_raw_payload(self, namespace: str, key: str, payload: Any) -> Path:
        target = self.raw_dir / namespace
        target.mkdir(parents=True, exist_ok=True)
        safe_key = key.replace("/", "_")
        path = target / f"{safe_key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def read_latest_raw_payload(self, namespace: str, key_prefix: str) -> Any | None:
        target = self.raw_dir / namespace
        if not target.exists():
            return None
        safe_prefix = key_prefix.replace("/", "_")
        candidates = sorted(
            target.glob(f"{safe_prefix}*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def list_raw_payload_paths(self, namespace: str, key_prefix: str | None = None) -> list[Path]:
        target = self.raw_dir / namespace
        if not target.exists():
            return []
        pattern = "*.json" if not key_prefix else f"{key_prefix.replace('/', '_')}*.json"
        return sorted(target.glob(pattern))

    def append_records(self, table_name: str, rows: Iterable[dict[str, Any]]) -> Path:
        rows = list(rows)
        path = self.tables_dir / f"{table_name}.parquet"
        if not rows:
            if not path.exists():
                pd.DataFrame().to_parquet(path, index=False)
            return path
        frame = self._normalize_frame(pd.DataFrame(rows))
        if path.exists():
            existing = self._normalize_frame(pd.read_parquet(path))
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates()
        frame.to_parquet(path, index=False)
        return path

    def read_table(self, table_name: str) -> pd.DataFrame:
        path = self.tables_dir / f"{table_name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def replace_table(self, table_name: str, rows: Iterable[dict[str, Any]]) -> Path:
        rows = list(rows)
        path = self.tables_dir / f"{table_name}.parquet"
        if not rows:
            pd.DataFrame().to_parquet(path, index=False)
            return path
        frame = self._normalize_frame(pd.DataFrame(rows))
        frame = frame.drop_duplicates()
        frame.to_parquet(path, index=False)
        return path

    def write_snapshot(self, snapshot: CompanySnapshot) -> None:
        snapshot_id = f"{snapshot.ticker}_{snapshot.as_of.replace(':', '-')}"
        self.write_raw_payload("snapshots", snapshot_id, asdict(snapshot))
        self.append_records("company_snapshots", [snapshot.to_record()])

        program_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        catalyst_rows: list[dict[str, Any]] = []
        financing_rows: list[dict[str, Any]] = []

        for program in snapshot.programs:
            program_rows.append(
                {
                    "ticker": snapshot.ticker,
                    "as_of": snapshot.as_of,
                    "program_id": program.program_id,
                    "name": program.name,
                    "modality": program.modality,
                    "phase": program.phase,
                    "conditions": json.dumps(program.conditions),
                    "pos_prior": program.pos_prior,
                    "tam_estimate": program.tam_estimate,
                }
            )
            for trial in program.trials:
                trial_rows.append(
                    {
                        "ticker": snapshot.ticker,
                        "as_of": snapshot.as_of,
                        "program_id": program.program_id,
                        "trial_id": trial.trial_id,
                        "title": trial.title,
                        "phase": trial.phase,
                        "status": trial.status,
                        "conditions": json.dumps(trial.conditions),
                        "interventions": json.dumps(trial.interventions),
                        "enrollment": trial.enrollment,
                        "primary_outcomes": json.dumps(trial.primary_outcomes),
                    }
                )
            for catalyst in program.catalyst_events:
                catalyst_rows.append(
                    {
                        "ticker": snapshot.ticker,
                        "as_of": snapshot.as_of,
                        "program_id": catalyst.program_id,
                        "event_id": catalyst.event_id,
                        "event_type": catalyst.event_type,
                        "title": catalyst.title,
                        "expected_date": catalyst.expected_date,
                        "horizon_days": catalyst.horizon_days,
                        "probability": catalyst.probability,
                        "importance": catalyst.importance,
                        "crowdedness": catalyst.crowdedness,
                        "status": catalyst.status,
                    }
                )

        for approved_product in snapshot.approved_products:
            program_rows.append(
                {
                    "ticker": snapshot.ticker,
                    "as_of": snapshot.as_of,
                    "program_id": approved_product.product_id,
                    "name": approved_product.name,
                    "modality": "commercial",
                    "phase": "APPROVED",
                    "conditions": json.dumps([approved_product.indication]),
                    "pos_prior": 1.0,
                    "tam_estimate": approved_product.annual_revenue,
                }
            )

        for financing in snapshot.financing_events:
            financing_rows.append(
                {
                    "ticker": snapshot.ticker,
                    "as_of": snapshot.as_of,
                    "event_id": financing.event_id,
                    "event_type": financing.event_type,
                    "probability": financing.probability,
                    "horizon_days": financing.horizon_days,
                    "expected_dilution_pct": financing.expected_dilution_pct,
                    "summary": financing.summary,
                }
            )

        self.append_records("programs", program_rows)
        self.append_records("trials", trial_rows)
        self.append_records("catalysts", catalyst_rows)
        self.append_records("financing_events", financing_rows)

    def write_feature_vectors(self, feature_vectors: list[FeatureVector]) -> None:
        self.append_records("feature_vectors", [item.to_row() for item in feature_vectors])

    def write_predictions(self, predictions: list[ModelPrediction]) -> None:
        self.append_records("predictions", [item.to_record() for item in predictions])

    def write_labels(self, rows: Iterable[dict[str, Any]]) -> Path:
        return self.replace_table("labels", rows)

    def write_event_labels(self, rows: Iterable[dict[str, Any]]) -> Path:
        return self.replace_table("event_labels", rows)

    def latest_snapshot_for(self, ticker: str) -> dict[str, Any] | None:
        table = self.read_table("company_snapshots")
        if table.empty:
            return None
        subset = table[table["ticker"] == ticker]
        if subset.empty:
            return None
        latest = subset.sort_values("as_of").iloc[-1]
        return latest.to_dict()

    def register_experiment(self, record: ExperimentRecord) -> Path:
        registry_path = self.experiments_dir / "registry.jsonl"
        with open(registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return registry_path

    def model_path(self, model_name: str, model_version: str) -> Path:
        return self.models_dir / f"{model_name}_{model_version}.pkl"

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        for column in frame.columns:
            if frame[column].dtype == "object":
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
        return frame

    def duckdb_connection(self):
        if duckdb is None:
            raise RuntimeError("duckdb is not installed. Install requirements.txt to enable DuckDB queries.")
        db_path = self.base_dir / "tatetuck.duckdb"
        conn = duckdb.connect(str(db_path))
        for parquet_path in self.tables_dir.glob("*.parquet"):
            table_name = parquet_path.stem
            conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
            )
        return conn
