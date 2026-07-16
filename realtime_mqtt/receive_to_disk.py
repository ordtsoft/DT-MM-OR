"""Receive MQTT shapefile frames, validate them, and persist them headlessly."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .mqtt_transport import MqttFrameReceiver, MqttSettings
from .store import FrameStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("realtime_mqtt/inbox"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dtmmor/shapefiles/frame")
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--tls", action="store_true")
    args = parser.parse_args()

    store = FrameStore(args.output)
    receiver = MqttFrameReceiver(
        MqttSettings(
            args.host, args.port, args.topic, args.qos,
            args.username, args.password, args.tls,
        ),
        store,
    )
    receiver.start()
    print(f"Receiving mqtt://{args.host}:{args.port}/{args.topic} into {args.output}")
    try:
        while True:
            time.sleep(1)
            snapshot = store.snapshot()
            print(
                f"accepted={snapshot.accepted} missing={snapshot.missing} "
                f"corrupt={snapshot.corrupt} duplicate={snapshot.duplicates} "
                f"out_of_order={snapshot.out_of_order} "
                f"latency={snapshot.current_latency_ms:.1f} ms"
            )
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
