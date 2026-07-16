"""Replay the sample shapefiles through MQTT with configurable faults."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from .mqtt_transport import MqttFramePublisher, MqttSettings
from .simulator import FaultProfile, run_dataset_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("shp"))
    parser.add_argument("--pattern", default="camera01_centroids_*.shp")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dtmmor/shapefiles/frame")
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--latency-ms", type=float, default=120.0)
    parser.add_argument("--jitter-ms", type=float, default=60.0)
    parser.add_argument("--drop", type=float, default=0.03)
    parser.add_argument("--corrupt", type=float, default=0.01)
    parser.add_argument("--duplicate", type=float, default=0.02)
    parser.add_argument("--reorder", type=float, default=0.04)
    parser.add_argument("--reorder-extra-ms", type=float, default=350.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--tls", action="store_true")
    return parser.parse_args()


def probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def main() -> None:
    args = parse_args()
    settings = MqttSettings(
        args.host, args.port, args.topic, args.qos,
        args.username, args.password, args.tls,
    )
    profile = FaultProfile(
        interval_ms=max(args.interval_ms, 1.0),
        latency_ms=max(args.latency_ms, 0.0),
        jitter_ms=max(args.jitter_ms, 0.0),
        drop_probability=probability(args.drop),
        corrupt_probability=probability(args.corrupt),
        duplicate_probability=probability(args.duplicate),
        reorder_probability=probability(args.reorder),
        reorder_extra_ms=max(args.reorder_extra_ms, 0.0),
        seed=args.seed,
    )
    stop_event = threading.Event()
    publisher = MqttFramePublisher(settings)
    publisher.start()
    print(f"Publishing to mqtt://{args.host}:{args.port}/{args.topic}")
    try:
        run_dataset_simulation(
            args.directory,
            publisher.publish,
            profile,
            stop_event,
            pattern=args.pattern,
            loop=args.loop,
            maximum_frames=args.maximum_frames,
            status=print,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        publisher.stop()


if __name__ == "__main__":
    main()
