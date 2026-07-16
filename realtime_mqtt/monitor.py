"""Print MQTT frame traffic metadata without dumping binary payloads."""

from __future__ import annotations

import argparse
from datetime import datetime

from .mqtt_transport import MqttSettings, configured_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dtmmor/#")
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()

    settings = MqttSettings(args.host, args.port, args.topic, args.qos)
    client = configured_client("dtmmor-traffic-monitor", settings)

    def on_connect(connected_client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            print(f"Connection rejected: {reason_code}", flush=True)
            return
        connected_client.subscribe(args.topic, qos=args.qos)
        print(
            f"Monitoring mqtt://{args.host}:{args.port}/{args.topic}",
            flush=True,
        )

    def on_message(_client, _userdata, message) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        print(
            f"{timestamp}  topic={message.topic}  bytes={len(message.payload)}  "
            f"qos={message.qos}  retained={message.retain}",
            flush=True,
        )

    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(args.host, args.port, keepalive=30)
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
