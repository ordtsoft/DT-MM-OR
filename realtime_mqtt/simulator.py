"""Dataset replay with controllable transport and sensor-like faults."""

from __future__ import annotations

import heapq
import io
import random
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .protocol import encode_frame


PayloadSink = Callable[[bytes], None]
StatusSink = Callable[[str], None]


def sequence_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class FaultProfile:
    interval_ms: float = 100.0
    latency_ms: float = 120.0
    jitter_ms: float = 60.0
    drop_probability: float = 0.03
    corrupt_probability: float = 0.01
    duplicate_probability: float = 0.02
    reorder_probability: float = 0.04
    reorder_extra_ms: float = 350.0
    seed: int = 7


@dataclass(order=True)
class ScheduledPayload:
    due_at: float
    order: int
    payload: bytes = field(compare=False)
    description: str = field(compare=False)


def corrupt_payload(payload: bytes, rng: random.Random) -> bytes:
    """Damage a component without updating its manifest checksum."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        target = "frame.shp"
        damaged = bytearray(entries[target])
        if not damaged:
            damaged.append(0)
        else:
            damaged[rng.randrange(len(damaged))] ^= 0xFF
        entries[target] = bytes(damaged)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        return output.getvalue()
    except (KeyError, zipfile.BadZipFile):
        return payload + b"\x00"


def run_dataset_simulation(
    directory: Path,
    emit: PayloadSink,
    profile: FaultProfile,
    stop_event: threading.Event,
    *,
    pattern: str = "camera01_centroids_*.shp",
    loop: bool = False,
    maximum_frames: int | None = None,
    status: StatusSink | None = None,
) -> None:
    files = sorted(directory.glob(pattern), key=sequence_number)
    if not files:
        raise FileNotFoundError(f"No shapefiles match {pattern!r} in {directory}")
    rng = random.Random(profile.seed)
    scheduled: list[ScheduledPayload] = []
    schedule_order = 0
    emitted_source_frames = 0
    index = 0
    next_capture = time.monotonic()

    def report(message: str) -> None:
        if status:
            status(message)

    while not stop_event.is_set():
        now = time.monotonic()
        source_complete = maximum_frames is not None and emitted_source_frames >= maximum_frames
        dataset_complete = index >= len(files) and not loop

        if not source_complete and not dataset_complete and now >= next_capture:
            if index >= len(files):
                index = 0
            shp_path = files[index]
            sequence = emitted_source_frames
            captured_at_ns = time.time_ns()
            payload = encode_frame(
                shp_path,
                sequence=sequence,
                frame_id=sequence_number(shp_path),
                captured_at_ns=captured_at_ns,
                source=shp_path.stem,
            )
            emitted_source_frames += 1
            index += 1
            next_capture += max(profile.interval_ms, 1.0) / 1000.0

            if rng.random() < profile.drop_probability:
                report(f"dropped sequence {sequence}")
            else:
                description = f"sequence {sequence}"
                if rng.random() < profile.corrupt_probability:
                    payload = corrupt_payload(payload, rng)
                    description += " (corrupt)"
                delay_ms = max(0.0, rng.gauss(profile.latency_ms, profile.jitter_ms))
                if rng.random() < profile.reorder_probability:
                    delay_ms += profile.reorder_extra_ms
                    description += " (delayed/reordered)"
                schedule_order += 1
                heapq.heappush(
                    scheduled,
                    ScheduledPayload(now + delay_ms / 1000.0, schedule_order, payload, description),
                )
                if rng.random() < profile.duplicate_probability:
                    schedule_order += 1
                    heapq.heappush(
                        scheduled,
                        ScheduledPayload(
                            now + (delay_ms + 10.0) / 1000.0,
                            schedule_order,
                            payload,
                            description + " (duplicate)",
                        ),
                    )

        now = time.monotonic()
        while scheduled and scheduled[0].due_at <= now:
            item = heapq.heappop(scheduled)
            emit(item.payload)
            report(f"emitted {item.description}")

        if (source_complete or dataset_complete) and not scheduled:
            return
        wake_times = [next_capture]
        if scheduled:
            wake_times.append(scheduled[0].due_at)
        stop_event.wait(max(0.001, min(max(min(wake_times) - time.monotonic(), 0.0), 0.02)))
