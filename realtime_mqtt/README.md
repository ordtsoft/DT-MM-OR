# Real-time MQTT shapefile prototype

This folder is isolated from the existing offline viewers. It can replay the
sample dataset locally with injected faults, or receive the same frame protocol
from an MQTT broker.

## Project assumption about `INTERP`

For this real-time prototype, every record in an incoming shapefile is treated
as a genuine sensor observation, including records where `INTERP=True`. The
historical meaning of that field in the source dataset is intentionally ignored.
A prediction is required only when a usable record does not arrive before its
deadline, or when the received frame is missing, corrupt, malformed, or late.

Each MQTT message is one ZIP payload containing:

- `manifest.json` with sequence, frame ID, capture time, source, sizes, and SHA-256 checksums
- `frame.shp`
- `frame.shx`
- `frame.dbf`

The receiver validates the complete bundle before atomically writing it to
`realtime_mqtt/inbox`. Partial or corrupt frames are rejected.

## Fastest test: no broker

From the repository root:

```powershell
python -m pip install -r realtime_mqtt/requirements.txt
python -m realtime_mqtt.live_viewer --mode loopback
```

The default simulation includes latency, jitter, dropped frames, corrupt
payloads, duplicates, and delayed frames that arrive out of order. The side
panel shows their effect in real time.

Make faults more obvious:

```powershell
python -m realtime_mqtt.live_viewer --mode loopback `
  --latency-ms 400 --jitter-ms 250 `
  --drop 0.12 --corrupt 0.05 --duplicate 0.08 --reorder 0.15
```

Set every fault probability to zero for a clean baseline:

```powershell
python -m realtime_mqtt.live_viewer --mode loopback `
  --latency-ms 0 --jitter-ms 0 --drop 0 --corrupt 0 --duplicate 0 --reorder 0
```

## MQTT test

Start an MQTT broker that accepts TCP connections on `localhost:1883`. Then use
two terminals.

Terminal 1, subscriber and live viewer:

```powershell
python -m realtime_mqtt.live_viewer --mode mqtt `
  --host localhost --port 1883 --topic dtmmor/shapefiles/frame
```

Terminal 2, fault-injecting dataset publisher:

```powershell
python -m realtime_mqtt.publish_dataset `
  --host localhost --port 1883 --topic dtmmor/shapefiles/frame --loop
```

For a headless receiver instead of the GUI:

```powershell
python -m realtime_mqtt.receive_to_disk --host localhost
```

All MQTT commands also accept `--username`, `--password`, and `--tls`.

## Important prototype decisions

- QoS 1 is the default, so duplicates are expected and explicitly detected.
- Sequence gaps are reported as missing and disappear if a delayed frame later arrives.
- Late frames are validated and saved, but the live viewer does not rewind.
- Latency is measured from simulated capture time to receiver ingestion time.
- The display predicts the selected entity causally: it scores the incoming
  frame before updating the velocity model with that frame.
- Large production frames may require raising the broker's maximum packet size.
  The prototype itself rejects payloads larger than 16 MiB.
