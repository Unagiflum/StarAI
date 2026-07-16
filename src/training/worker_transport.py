"""Opt-in endpoint timing for coordinated worker pipe traffic."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
import time
from typing import Any


@dataclass(frozen=True)
class TransferMeasurement:
    serialization_seconds: float = 0.0
    transfer_seconds: float = 0.0
    byte_count: int = 0


class TransportTimingAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.send_serialize_seconds = 0.0
        self.send_transfer_seconds = 0.0
        self.recv_transfer_seconds = 0.0
        self.recv_deserialize_seconds = 0.0
        self.sent_bytes = 0
        self.received_bytes = 0
        self.sent_messages = 0
        self.received_messages = 0

    def record_send(self, measurement: TransferMeasurement) -> None:
        self.send_serialize_seconds += float(measurement.serialization_seconds)
        self.send_transfer_seconds += float(measurement.transfer_seconds)
        self.sent_bytes += int(measurement.byte_count)
        self.sent_messages += 1

    def record_receive(self, measurement: TransferMeasurement) -> None:
        self.recv_transfer_seconds += float(measurement.transfer_seconds)
        self.recv_deserialize_seconds += float(measurement.serialization_seconds)
        self.received_bytes += int(measurement.byte_count)
        self.received_messages += 1

    def snapshot(self) -> dict[str, float | int]:
        return {
            "send_serialize_seconds": self.send_serialize_seconds,
            "send_transfer_seconds": self.send_transfer_seconds,
            "recv_transfer_seconds": self.recv_transfer_seconds,
            "recv_deserialize_seconds": self.recv_deserialize_seconds,
            "sent_bytes": self.sent_bytes,
            "received_bytes": self.received_bytes,
            "sent_messages": self.sent_messages,
            "received_messages": self.received_messages,
        }


def send_timed(connection: Any, value: Any) -> TransferMeasurement:
    started_at = time.perf_counter()
    payload = ForkingPickler.dumps(value)
    serialized_at = time.perf_counter()
    connection.send_bytes(payload)
    finished_at = time.perf_counter()
    return TransferMeasurement(
        serialization_seconds=max(0.0, serialized_at - started_at),
        transfer_seconds=max(0.0, finished_at - serialized_at),
        byte_count=len(payload),
    )


def receive_timed(connection: Any) -> tuple[Any, TransferMeasurement]:
    started_at = time.perf_counter()
    payload = connection.recv_bytes()
    received_at = time.perf_counter()
    value = ForkingPickler.loads(payload)
    finished_at = time.perf_counter()
    return value, TransferMeasurement(
        serialization_seconds=max(0.0, finished_at - received_at),
        transfer_seconds=max(0.0, received_at - started_at),
        byte_count=len(payload),
    )
