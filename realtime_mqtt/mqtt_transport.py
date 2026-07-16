"""Small Paho MQTT adapters used by the live viewer and dataset publisher."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .store import FrameStore


def _mqtt_module():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError(
            "MQTT mode requires paho-mqtt; install realtime_mqtt/requirements.txt"
        ) from exc
    return mqtt


@dataclass(frozen=True)
class MqttSettings:
    host: str = "localhost"
    port: int = 1883
    topic: str = "dtmmor/shapefiles/frame"
    qos: int = 1
    username: str | None = None
    password: str | None = None
    tls: bool = False


def configured_client(client_id: str, settings: MqttSettings):
    mqtt = _mqtt_module()
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
    )
    if settings.username:
        client.username_pw_set(settings.username, settings.password)
    if settings.tls:
        client.tls_set()
    return client


class MqttFrameReceiver:
    def __init__(self, settings: MqttSettings, store: FrameStore) -> None:
        self.settings = settings
        self.store = store
        self.connected = threading.Event()
        self.last_connection_error = ""
        self.client = configured_client("dtmmor-live-viewer", settings)
        self.client.on_connect = self._on_connect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            self.last_connection_error = f"MQTT connection rejected: {reason_code}"
            return
        client.subscribe(self.settings.topic, qos=self.settings.qos)
        self.connected.set()
        self.last_connection_error = ""

    def _on_connect_fail(self, _client, _userdata) -> None:
        self.last_connection_error = "MQTT connection failed"

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        self.connected.clear()
        if reason_code.is_failure:
            self.last_connection_error = f"Unexpected MQTT disconnect: {reason_code}"

    def _on_message(self, _client, _userdata, message) -> None:
        self.store.ingest(bytes(message.payload))

    def start(self) -> None:
        self.client.connect_async(self.settings.host, self.settings.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()

class MqttFramePublisher:
    def __init__(self, settings: MqttSettings) -> None:
        self.settings = settings
        self.connected = threading.Event()
        self.client = configured_client("dtmmor-dataset-publisher", settings)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        if not reason_code.is_failure:
            self.connected.set()

    def _on_disconnect(self, _client, _userdata, _flags, _reason_code, _properties) -> None:
        self.connected.clear()

    def start(self, timeout: float = 10.0) -> None:
        self.client.connect(self.settings.host, self.settings.port, keepalive=30)
        self.client.loop_start()
        if not self.connected.wait(timeout):
            self.stop()
            raise TimeoutError(f"Could not connect to MQTT broker {self.settings.host}:{self.settings.port}")

    def publish(self, payload: bytes) -> None:
        result = self.client.publish(self.settings.topic, payload, qos=self.settings.qos)
        if result.rc != 0:
            raise RuntimeError(f"MQTT publish failed with result code {result.rc}")

    def stop(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()
