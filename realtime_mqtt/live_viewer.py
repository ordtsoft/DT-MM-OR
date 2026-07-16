"""Live shapefile viewer for MQTT or broker-free fault-simulation mode."""

from __future__ import annotations

import argparse
import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import shapefile

from viewer_comparison import VelocityPredictor
from viewer_prediction import AXIS, BACKGROUND, COLOR_CLASSES, GRID, NAME_TO_RGB, nice_step

from .mqtt_transport import MqttFrameReceiver, MqttSettings
from .simulator import FaultProfile, run_dataset_simulation
from .store import FrameEvent, FrameStore, HealthSnapshot


ACTUAL_COLOR = "#111827"
PREDICTION_COLOR = "#f97316"
HEALTHY = "#15803d"
WARNING = "#b45309"
FAILED = "#b91c1c"
Point = tuple[float, float]


def load_points(path: Path) -> tuple[list[dict], tuple[float, float, float, float]]:
    points: list[dict] = []
    with shapefile.Reader(str(path)) as reader:
        bounds = tuple(reader.bbox)
        for item in reader.iterShapeRecords():
            if not item.shape.points:
                continue
            attributes = item.record.as_dict()
            source_rgb = (attributes.get("R"), attributes.get("G"), attributes.get("B"))
            classification = COLOR_CLASSES.get(source_rgb)
            name, color = classification if classification else (
                attributes.get("RGB_HEX") or "unknown",
                attributes.get("RGB_HEX") or "#64748b",
            )
            points.append({
                "x": item.shape.points[0][0],
                "y": item.shape.points[0][1],
                "name": name,
                "color": color,
                "interpolated": bool(attributes.get("INTERP", False)),
            })
    return points, bounds


class LiveViewer(tk.Tk):
    def __init__(
        self,
        store: FrameStore,
        stop_event: threading.Event,
        *,
        entity: str,
        horizon: int,
        mode: str,
        source_state,
        on_close,
    ) -> None:
        super().__init__()
        self.store = store
        self.stop_event = stop_event
        self.entity = entity
        self.horizon = horizon
        self.mode = mode
        self.source_state = source_state
        self.on_close_callback = on_close
        self.current_event: FrameEvent | None = None
        self.current_points: list[dict] = []
        self.latest_sequence = -1
        self.bounds = [0.0, 0.0, 5.0, 5.0]
        self.predictor = VelocityPredictor()
        self.actual_track: list[Point | None] = []
        self.prediction_track: list[Point | None] = []
        self.scored_prediction: Point | None = None
        self.forecast: list[Point] = []
        self.current_error: float | None = None

        self.title(f"Real-time shapefile stream — {mode}")
        self.geometry("1120x720")
        self.minsize(780, 520)
        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

        controls = ttk.Frame(self, padding=(10, 7))
        controls.pack(fill="x")
        ttk.Label(controls, text=f"Mode: {mode}", font=("Segoe UI Semibold", 9)).pack(side="left")
        ttk.Label(controls, text=f"Predicting: {entity}").pack(side="left", padx=18)
        ttk.Label(
            controls,
            text="ring = scored prediction   diamond = future target",
        ).pack(side="left")
        self.status_label = ttk.Label(controls, anchor="e")
        self.status_label.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(25, self.poll)

    def close(self) -> None:
        self.stop_event.set()
        self.on_close_callback()
        self.destroy()

    def poll(self) -> None:
        newest: FrameEvent | None = None
        while True:
            try:
                event = self.store.events.get_nowait()
            except queue.Empty:
                break
            if (
                event.bundle.sequence > self.latest_sequence
                and (newest is None or event.bundle.sequence > newest.bundle.sequence)
            ):
                newest = event
        if newest is not None:
            self.accept_frame(newest)
        self.draw()
        if not self.stop_event.is_set():
            self.after(25, self.poll)

    def accept_frame(self, event: FrameEvent) -> None:
        try:
            points, bounds = load_points(event.paths[".shp"])
        except (OSError, shapefile.ShapefileException, ValueError):
            return
        self.current_event = event
        self.current_points = points
        previous_sequence = self.latest_sequence
        self.latest_sequence = event.bundle.sequence
        self.bounds[2] = max(self.bounds[2], bounds[2] * 1.08)
        self.bounds[3] = max(self.bounds[3], bounds[3] * 1.08)

        actual = next(
            ((point["x"], point["y"]) for point in points if point["name"] == self.entity),
            None,
        )
        sequence = event.bundle.sequence
        self.scored_prediction = self.predictor.predict(sequence)
        self.current_error = (
            math.dist(actual, self.scored_prediction)
            if actual is not None and self.scored_prediction is not None
            else None
        )
        missing_frames = max(sequence - previous_sequence - 1, 0)
        if previous_sequence >= 0 and missing_frames:
            self.actual_track.extend([None] * missing_frames)
            self.prediction_track.extend([None] * missing_frames)
        self.actual_track.append(actual)
        self.prediction_track.append(self.scored_prediction)
        if actual is not None:
            self.predictor.update(sequence, actual)
        self.forecast = [
            prediction
            for target in range(sequence + 1, sequence + self.horizon + 1)
            if (prediction := self.predictor.predict(target)) is not None
        ]

    def draw(self) -> None:
        if self.canvas.winfo_width() < 30 or self.canvas.winfo_height() < 30:
            return
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        left, right, top, bottom = 72.0, 300.0, 50.0, 55.0
        plot_width = max(width - left - right - 20, 1)
        plot_height = max(height - top - bottom, 1)
        x0, y0, x1, y1 = self.bounds

        def screen(point: Point) -> Point:
            return (
                left + (point[0] - x0) / max(x1 - x0, 1e-9) * plot_width,
                top + (y1 - point[1]) / max(y1 - y0, 1e-9) * plot_height,
            )

        x_step, y_step = nice_step(x1 - x0), nice_step(y1 - y0)
        tick = math.ceil(x0 / x_step) * x_step
        while tick <= x1:
            sx, _ = screen((tick, y0))
            self.canvas.create_line(sx, top, sx, top + plot_height, fill=GRID)
            self.canvas.create_text(sx, top + plot_height + 18, text=f"{tick:g}", fill=AXIS)
            tick += x_step
        tick = math.ceil(y0 / y_step) * y_step
        while tick <= y1:
            _, sy = screen((x0, tick))
            self.canvas.create_line(left, sy, left + plot_width, sy, fill=GRID)
            self.canvas.create_text(left - 10, sy, text=f"{tick:g}", anchor="e", fill=AXIS)
            tick += y_step
        self.canvas.create_rectangle(
            left, top, left + plot_width, top + plot_height, outline=AXIS,
        )
        self.canvas.create_text(
            left + plot_width / 2, height - 16, text="X (m)", fill=AXIS,
        )
        self.canvas.create_text(
            18, top + plot_height / 2, text="Y (m)", angle=90, fill=AXIS,
        )

        for point in self.current_points:
            sx, sy = screen((point["x"], point["y"]))
            radius = 6
            outline = "#ffffff" if not point["interpolated"] else "#64748b"
            self.canvas.create_oval(
                sx - radius, sy - radius, sx + radius, sy + radius,
                fill=point["color"], outline=outline, width=2,
            )
            self.canvas.create_text(
                sx + 9, sy, text=point["name"], anchor="w",
                fill=ACTUAL_COLOR, font=("Segoe UI Semibold", 8),
            )

        def draw_segments(
            track: list[Point | None], color: str, line_width: int, dash=(),
        ) -> None:
            segment: list[Point] = []

            def flush() -> None:
                if len(segment) > 1:
                    converted = [screen(point) for point in segment]
                    self.canvas.create_line(
                        *sum(converted, ()), fill=color, width=line_width,
                        dash=dash, smooth=True,
                    )
                segment.clear()

            for point in track[-80:]:
                if point is None:
                    flush()
                else:
                    segment.append(point)
            flush()

        draw_segments(self.actual_track, ACTUAL_COLOR, 3)
        draw_segments(self.prediction_track, "#fdba74", 2, (4, 4))

        if self.scored_prediction is not None:
            px, py = screen(self.scored_prediction)
            self.canvas.create_oval(
                px - 6, py - 6, px + 6, py + 6,
                fill=BACKGROUND, outline=PREDICTION_COLOR, width=2,
            )
        if self.forecast:
            future = [screen(point) for point in self.forecast]
            origin = None
            current_actual = self.actual_track[-1] if self.actual_track else None
            if current_actual is not None:
                origin = screen(current_actual)
            if origin:
                self.canvas.create_line(
                    *(origin + sum(future, ())),
                    fill=PREDICTION_COLOR, width=2, dash=(6, 4),
                )
            for px, py in future[::max(1, len(future) // 5)]:
                self.canvas.create_oval(
                    px - 2.5, py - 2.5, px + 2.5, py + 2.5,
                    fill=PREDICTION_COLOR, outline=BACKGROUND,
                )
            px, py = future[-1]
            self.canvas.create_polygon(
                px, py - 7, px + 7, py, px, py + 7, px - 7, py,
                fill=PREDICTION_COLOR, outline=BACKGROUND, width=2,
            )

        snapshot = self.store.snapshot()
        self.draw_health(left + plot_width + 20, top, snapshot)
        if self.current_event:
            error = (
                f"prediction error {self.current_error:.3f} m"
                if self.current_error is not None
                else "prediction warming up"
            )
            self.status_label.configure(
                text=f"sequence {self.latest_sequence} · {error}",
            )
        else:
            self.status_label.configure(text="waiting for first frame")

    def draw_health(self, x: float, y: float, snapshot: HealthSnapshot) -> None:
        latency_color = (
            HEALTHY if snapshot.current_latency_ms < 250
            else WARNING if snapshot.current_latency_ms < 600
            else FAILED
        )
        source_status = self.source_state()
        self.canvas.create_text(
            x, y, anchor="nw", text="Stream health",
            fill=ACTUAL_COLOR, font=("Segoe UI Semibold", 13),
        )
        self.canvas.create_text(
            x, y + 29, anchor="nw",
            text=f"{self.mode}: {source_status}",
            fill=HEALTHY if source_status == "connected" else WARNING,
            font=("Segoe UI Semibold", 9),
        )
        rows = [
            ("received", snapshot.received),
            ("accepted", snapshot.accepted),
            ("missing", snapshot.missing),
            ("corrupt", snapshot.corrupt),
            ("duplicates", snapshot.duplicates),
            ("out of order", snapshot.out_of_order),
            ("viewer queue drops", snapshot.queue_overflows),
        ]
        row_y = y + 62
        for label, value in rows:
            value_color = FAILED if value and label not in {"received", "accepted"} else ACTUAL_COLOR
            self.canvas.create_text(x, row_y, anchor="nw", text=label, fill=AXIS)
            self.canvas.create_text(
                x + 205, row_y, anchor="ne", text=str(value),
                fill=value_color, font=("Segoe UI Semibold", 9),
            )
            row_y += 24
        self.canvas.create_text(
            x, row_y + 8, anchor="nw",
            text=f"latency now   {snapshot.current_latency_ms:7.1f} ms",
            fill=latency_color, font=("Segoe UI Semibold", 10),
        )
        self.canvas.create_text(
            x, row_y + 34, anchor="nw",
            text=f"average       {snapshot.average_latency_ms:7.1f} ms\n"
                 f"maximum       {snapshot.maximum_latency_ms:7.1f} ms",
            fill=AXIS, font=("Consolas", 9),
        )
        if snapshot.last_error:
            self.canvas.create_text(
                x, row_y + 88, anchor="nw",
                text=f"Last rejected payload:\n{snapshot.last_error}",
                width=255, fill=FAILED, font=("Segoe UI", 8),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loopback", "mqtt"), default="loopback")
    parser.add_argument("--directory", type=Path, default=Path("shp"))
    parser.add_argument("--output", type=Path, default=Path("realtime_mqtt/inbox"))
    parser.add_argument("--pattern", default="camera01_centroids_*.shp")
    parser.add_argument("--entity", choices=sorted(NAME_TO_RGB), default="circulator")
    parser.add_argument("--horizon", type=int, default=10)
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
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--tls", action="store_true")
    return parser.parse_args()


def clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def main() -> None:
    args = parse_args()
    stop_event = threading.Event()
    store = FrameStore(args.output)
    receiver: MqttFrameReceiver | None = None

    if args.mode == "mqtt":
        settings = MqttSettings(
            args.host, args.port, args.topic, args.qos,
            args.username, args.password, args.tls,
        )
        receiver = MqttFrameReceiver(settings, store)
        receiver.start()

        def source_state() -> str:
            if receiver.connected.is_set():
                return "connected"
            return receiver.last_connection_error or "connecting"

        def stop_source() -> None:
            receiver.stop()
    else:
        profile = FaultProfile(
            interval_ms=max(args.interval_ms, 1.0),
            latency_ms=max(args.latency_ms, 0.0),
            jitter_ms=max(args.jitter_ms, 0.0),
            drop_probability=clamp_probability(args.drop),
            corrupt_probability=clamp_probability(args.corrupt),
            duplicate_probability=clamp_probability(args.duplicate),
            reorder_probability=clamp_probability(args.reorder),
        )
        thread = threading.Thread(
            target=run_dataset_simulation,
            args=(args.directory, store.ingest, profile, stop_event),
            kwargs={"pattern": args.pattern, "loop": True},
            daemon=True,
            name="shapefile-loopback",
        )
        thread.start()

        def source_state() -> str:
            return "connected" if thread.is_alive() else "stopped"

        def stop_source() -> None:
            stop_event.set()

    viewer = LiveViewer(
        store,
        stop_event,
        entity=args.entity,
        horizon=max(args.horizon, 1),
        mode=args.mode,
        source_state=source_state,
        on_close=stop_source,
    )
    viewer.mainloop()


if __name__ == "__main__":
    main()
