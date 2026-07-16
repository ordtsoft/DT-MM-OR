"""Binary MQTT payload format for complete, checksummed shapefile frames."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROTOCOL_VERSION = 1
REQUIRED_EXTENSIONS = (".shp", ".shx", ".dbf")
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class BundleError(ValueError):
    """Raised when an MQTT frame payload is incomplete or invalid."""


@dataclass(frozen=True)
class FrameBundle:
    sequence: int
    frame_id: int
    captured_at_ns: int
    source: str
    components: dict[str, bytes]


def component_paths(shp_path: Path) -> dict[str, Path]:
    base = shp_path.with_suffix("")
    paths = {extension: base.with_suffix(extension) for extension in REQUIRED_EXTENSIONS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shapefile components: {', '.join(missing)}")
    return paths


def encode_frame(
    shp_path: Path,
    *,
    sequence: int,
    frame_id: int | None = None,
    captured_at_ns: int | None = None,
    source: str | None = None,
) -> bytes:
    paths = component_paths(shp_path)
    components = {extension: path.read_bytes() for extension, path in paths.items()}
    return encode_components(
        components,
        sequence=sequence,
        frame_id=sequence if frame_id is None else frame_id,
        captured_at_ns=time.time_ns() if captured_at_ns is None else captured_at_ns,
        source=shp_path.stem if source is None else source,
    )


def encode_components(
    components: Mapping[str, bytes],
    *,
    sequence: int,
    frame_id: int,
    captured_at_ns: int,
    source: str,
) -> bytes:
    missing = [extension for extension in REQUIRED_EXTENSIONS if extension not in components]
    if missing:
        raise BundleError(f"Missing components: {', '.join(missing)}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "sequence": sequence,
        "frame_id": frame_id,
        "captured_at_ns": captured_at_ns,
        "source": source,
        "components": {
            extension: {
                "name": f"frame{extension}",
                "size": len(components[extension]),
                "sha256": hashlib.sha256(components[extension]).hexdigest(),
            }
            for extension in REQUIRED_EXTENSIONS
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        for extension in REQUIRED_EXTENSIONS:
            archive.writestr(f"frame{extension}", components[extension])
    payload = output.getvalue()
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise BundleError(f"Frame payload is {len(payload)} bytes; limit is {MAX_PAYLOAD_BYTES}")
    return payload


def decode_frame(payload: bytes) -> FrameBundle:
    if not payload:
        raise BundleError("Empty frame payload")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise BundleError(f"Frame payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if "manifest.json" not in names:
                raise BundleError("Payload has no manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("protocol_version") != PROTOCOL_VERSION:
                raise BundleError(f"Unsupported protocol version: {manifest.get('protocol_version')!r}")
            component_meta = manifest.get("components")
            if not isinstance(component_meta, dict):
                raise BundleError("Manifest components must be an object")
            components: dict[str, bytes] = {}
            for extension in REQUIRED_EXTENSIONS:
                metadata = component_meta.get(extension)
                if not isinstance(metadata, dict):
                    raise BundleError(f"Manifest is missing {extension}")
                name = metadata.get("name")
                if name != f"frame{extension}" or name not in names:
                    raise BundleError(f"Archive is missing frame{extension}")
                data = archive.read(name)
                if len(data) != metadata.get("size"):
                    raise BundleError(f"Size mismatch for {extension}")
                if hashlib.sha256(data).hexdigest() != metadata.get("sha256"):
                    raise BundleError(f"Checksum mismatch for {extension}")
                components[extension] = data
    except BundleError:
        raise
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BundleError(f"Invalid frame bundle: {exc}") from exc

    try:
        sequence = int(manifest["sequence"])
        frame_id = int(manifest["frame_id"])
        captured_at_ns = int(manifest["captured_at_ns"])
        source = str(manifest["source"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(f"Invalid manifest metadata: {exc}") from exc
    if sequence < 0 or captured_at_ns <= 0:
        raise BundleError("Sequence must be non-negative and capture time must be positive")
    return FrameBundle(sequence, frame_id, captured_at_ns, source, components)
