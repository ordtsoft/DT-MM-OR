import random
import threading
import time
from pathlib import Path

import pytest
import shapefile

from realtime_mqtt.protocol import BundleError, decode_frame, encode_frame
from realtime_mqtt.simulator import (
    FaultProfile,
    corrupt_payload,
    run_dataset_simulation,
    sequence_number,
)
from realtime_mqtt.store import FrameStore


def make_shapefile(directory: Path, name: str = "camera01_centroids_1") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with shapefile.Writer(str(path), shapeType=shapefile.POINT) as writer:
        writer.field("FRAME_ID", "N", size=10)
        writer.field("R", "N", size=3)
        writer.field("G", "N", size=3)
        writer.field("B", "N", size=3)
        writer.point(1.25, 2.5)
        writer.record(17, 80, 0, 0)
    return path.with_suffix(".shp")


def test_frame_bundle_round_trip(tmp_path: Path) -> None:
    shp_path = make_shapefile(tmp_path)

    payload = encode_frame(
        shp_path,
        sequence=4,
        frame_id=17,
        captured_at_ns=123456789,
        source="sensor-a",
    )
    bundle = decode_frame(payload)

    assert bundle.sequence == 4
    assert bundle.frame_id == 17
    assert bundle.source == "sensor-a"
    assert set(bundle.components) == {".shp", ".shx", ".dbf"}
    assert bundle.components[".shp"] == shp_path.read_bytes()


def test_corrupt_bundle_is_rejected(tmp_path: Path) -> None:
    payload = encode_frame(
        make_shapefile(tmp_path),
        sequence=0,
        captured_at_ns=time.time_ns(),
    )

    with pytest.raises(BundleError):
        decode_frame(payload[: max(1, len(payload) // 2)])


def test_fault_corruption_changes_payload(tmp_path: Path) -> None:
    payload = encode_frame(
        make_shapefile(tmp_path),
        sequence=0,
        captured_at_ns=time.time_ns(),
    )

    corrupted = corrupt_payload(payload, random.Random(2))

    assert corrupted != payload
    with pytest.raises(BundleError, match="Checksum mismatch"):
        decode_frame(corrupted)


def test_store_tracks_gaps_duplicates_and_late_frames(tmp_path: Path) -> None:
    shp_path = make_shapefile(tmp_path / "source")
    store = FrameStore(tmp_path / "inbox")

    def payload(sequence: int) -> bytes:
        return encode_frame(
            shp_path,
            sequence=sequence,
            captured_at_ns=time.time_ns() - 20_000_000,
        )

    first = store.ingest(payload(0))
    second = store.ingest(payload(2))

    assert first is not None and second is not None
    assert store.snapshot().missing == 1

    late = store.ingest(payload(1))
    duplicate = store.ingest(payload(1))
    corrupt = store.ingest(b"not a zip frame")
    snapshot = store.snapshot()

    assert late is not None and late.out_of_order
    assert duplicate is None
    assert corrupt is None
    assert snapshot.received == 5
    assert snapshot.accepted == 3
    assert snapshot.missing == 0
    assert snapshot.out_of_order == 1
    assert snapshot.duplicates == 1
    assert snapshot.corrupt == 1
    assert first.paths[".shp"].is_file()
    assert snapshot.current_latency_ms >= 0


def test_store_counts_missing_sequences_before_first_arrival(tmp_path: Path) -> None:
    shp_path = make_shapefile(tmp_path / "source")
    store = FrameStore(tmp_path / "inbox")
    payload = encode_frame(
        shp_path,
        sequence=2,
        captured_at_ns=time.time_ns(),
    )

    store.ingest(payload)

    assert store.snapshot().missing == 2


def test_short_clean_simulation_emits_ordered_bundles(tmp_path: Path) -> None:
    make_shapefile(tmp_path, "camera01_centroids_2")
    make_shapefile(tmp_path, "camera01_centroids_10")
    emitted: list[bytes] = []
    profile = FaultProfile(
        interval_ms=1,
        latency_ms=0,
        jitter_ms=0,
        drop_probability=0,
        corrupt_probability=0,
        duplicate_probability=0,
        reorder_probability=0,
    )

    run_dataset_simulation(
        tmp_path,
        emitted.append,
        profile,
        threading.Event(),
        maximum_frames=2,
    )

    bundles = [decode_frame(payload) for payload in emitted]
    assert [bundle.sequence for bundle in bundles] == [0, 1]
    assert [bundle.frame_id for bundle in bundles] == [2, 10]


def test_simulator_sequence_number_uses_numeric_suffix() -> None:
    assert sequence_number(Path("frame_10.shp")) == 10
    assert sequence_number(Path("frame_2.shp")) == 2
