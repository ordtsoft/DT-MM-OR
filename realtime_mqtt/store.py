"""Thread-safe validation, health accounting, and atomic frame persistence."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .protocol import BundleError, FrameBundle, REQUIRED_EXTENSIONS, decode_frame


@dataclass(frozen=True)
class FrameEvent:
    bundle: FrameBundle
    received_at_ns: int
    latency_ms: float
    paths: dict[str, Path]
    out_of_order: bool


@dataclass(frozen=True)
class HealthSnapshot:
    received: int
    accepted: int
    corrupt: int
    duplicates: int
    out_of_order: int
    missing: int
    queue_overflows: int
    current_latency_ms: float
    average_latency_ms: float
    maximum_latency_ms: float
    last_error: str


class FrameStore:
    def __init__(self, directory: Path, queue_size: int = 100) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events: queue.Queue[FrameEvent] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self._minimum_sequence: int | None = None
        self._maximum_sequence: int | None = None
        self._received = 0
        self._accepted = 0
        self._corrupt = 0
        self._duplicates = 0
        self._out_of_order = 0
        self._queue_overflows = 0
        self._latency_total = 0.0
        self._current_latency = 0.0
        self._maximum_latency = 0.0
        self._last_error = ""

    def ingest(self, payload: bytes, received_at_ns: int | None = None) -> FrameEvent | None:
        received_at_ns = time.time_ns() if received_at_ns is None else received_at_ns
        with self._lock:
            self._received += 1
        try:
            bundle = decode_frame(payload)
        except BundleError as exc:
            with self._lock:
                self._corrupt += 1
                self._last_error = str(exc)
            return None

        latency_ms = max((received_at_ns - bundle.captured_at_ns) / 1_000_000.0, 0.0)
        with self._lock:
            if bundle.sequence in self._seen:
                self._duplicates += 1
                return None
            out_of_order = (
                self._maximum_sequence is not None
                and bundle.sequence < self._maximum_sequence
            )
            if out_of_order:
                self._out_of_order += 1
            self._seen.add(bundle.sequence)
            self._minimum_sequence = (
                bundle.sequence
                if self._minimum_sequence is None
                else min(self._minimum_sequence, bundle.sequence)
            )
            self._maximum_sequence = (
                bundle.sequence
                if self._maximum_sequence is None
                else max(self._maximum_sequence, bundle.sequence)
            )
            self._accepted += 1
            self._current_latency = latency_ms
            self._latency_total += latency_ms
            self._maximum_latency = max(self._maximum_latency, latency_ms)

        paths = self._write_atomically(bundle)
        event = FrameEvent(bundle, received_at_ns, latency_ms, paths, out_of_order)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)
            with self._lock:
                self._queue_overflows += 1
        return event

    def _write_atomically(self, bundle: FrameBundle) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        stem = f"frame_{bundle.sequence:09d}"
        for extension in REQUIRED_EXTENSIONS:
            target = self.directory / f"{stem}{extension}"
            temporary = self.directory / f".{stem}{extension}.part"
            temporary.write_bytes(bundle.components[extension])
            os.replace(temporary, target)
            paths[extension] = target
        return paths

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            missing = 0
            if self._maximum_sequence is not None:
                expected = self._maximum_sequence + 1
                missing = expected - len(self._seen)
            average = self._latency_total / self._accepted if self._accepted else 0.0
            return HealthSnapshot(
                received=self._received,
                accepted=self._accepted,
                corrupt=self._corrupt,
                duplicates=self._duplicates,
                out_of_order=self._out_of_order,
                missing=missing,
                queue_overflows=self._queue_overflows,
                current_latency_ms=self._current_latency,
                average_latency_ms=average,
                maximum_latency_ms=self._maximum_latency,
                last_error=self._last_error,
            )
