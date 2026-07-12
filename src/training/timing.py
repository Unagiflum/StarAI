"""Training timing diagnostics and CSV export helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
import threading
import time
from typing import Any, Iterator, Mapping


TIMING_FIELDS = (
    "opponent_load_seconds",
    "opponent_schedule_seconds",
    "observation_seconds",
    "trainee_inference_seconds",
    "opponent_observation_seconds",
    "opponent_inference_seconds",
    "simple_opponent_seconds",
    "decision_snapshot_seconds",
    "simulation_seconds",
    "outcome_seconds",
    "reward_seconds",
    "replay_insert_seconds",
    "progress_callback_seconds",
    "display_throttle_seconds",
    "optimization_seconds",
    "save_seconds",
)

BATCH_TIMING_CSV_HEADER = (
    "Timestamp",
    "Run ID",
    "Instance ID",
    "Ship",
    "Slot",
    "Batch",
    "Match Count",
    "Completed Rounds",
    "Frames",
    "Started At",
    "Ended At",
    "Total Seconds",
    "Running Instances Start",
    "Running Instances End",
    "Replay Size",
    "Average Loss",
    "Wins",
    "Losses",
    "Draws",
    *TIMING_FIELDS,
)

MULTI_INSTANCE_EVENT_CSV_HEADER = (
    "Timestamp",
    "Event",
    "Run ID",
    "Instance ID",
    "Ship",
    "Slot",
    "Batch",
    "Running Instances",
    "Total Seconds",
    "Details",
)

_CSV_LOCK = threading.Lock()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TrainingTimingAccumulator:
    """Collect elapsed seconds by named training timing bucket."""

    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, field: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.add(field, time.perf_counter() - started_at)

    def add(self, field: str, seconds: float) -> None:
        self.values[field] = self.values.get(field, 0.0) + max(0.0, float(seconds))

    def snapshot(self) -> dict[str, float]:
        return {field: float(self.values.get(field, 0.0)) for field in TIMING_FIELDS}


class NullTrainingTimingAccumulator:
    """No-op timing collector used when diagnostics are not requested."""

    @contextmanager
    def measure(self, field: str) -> Iterator[None]:
        yield

    def add(self, field: str, seconds: float) -> None:
        return None

    def snapshot(self) -> dict[str, float]:
        return {}


def append_batch_timing_csv(path: Path, row: Mapping[str, Any]) -> None:
    _append_csv_row(path, BATCH_TIMING_CSV_HEADER, row)


def append_multi_instance_event_csv(path: Path, row: Mapping[str, Any]) -> None:
    _append_csv_row(path, MULTI_INSTANCE_EVENT_CSV_HEADER, row)


def _append_csv_row(path: Path, header: tuple[str, ...], row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(header)
            writer.writerow([row.get(column, "") for column in header])
