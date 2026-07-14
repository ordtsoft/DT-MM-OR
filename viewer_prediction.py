"""Cold-start, online trajectory prediction for one shapefile entity."""

from __future__ import annotations

import argparse
import math
import re
import tkinter as tk
from collections import deque
from functools import lru_cache
from pathlib import Path
from tkinter import messagebox, ttk

import shapefile


BACKGROUND = "#f8fafc"
GRID = "#dbe3ec"
AXIS = "#52606d"
ACTUAL_COLOR = "#2563eb"
PREDICTED_COLOR = "#f97316"

# Source RGB -> (display name, display color)
COLOR_CLASSES = {
    (10, 0, 0): ("instrument_table", "#FF3399"),
    (30, 0, 0): ("operating table", "#FFFF00"),
    (40, 0, 0): ("mps station", "#850085"),
    (50, 0, 0): ("patient", "#FF0000"),
    (60, 0, 0): ("drape", "#B75BFF"),
    (70, 0, 0): ("anesthesist", "#B1FF6E"),
    (80, 0, 0): ("circulator", "#FF8000"),
    (90, 0, 0): ("assistant surgeon", "#74A674"),
    (100, 0, 0): ("head surgeon", "#4CA1F5"),
    (110, 0, 0): ("robot technician (mps)", "#7D6419"),
    (120, 0, 0): ("nurse", "#80FF00"),
    (130, 0, 0): ("drill", "#00FF80"),
    (170, 0, 0): ("robot", "#3C4BFF"),
}
NAME_TO_RGB = {name: rgb for rgb, (name, _color) in COLOR_CLASSES.items()}


def sequence_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def nice_step(span: float, target_lines: int = 8) -> float:
    raw = max(span / target_lines, 1e-12)
    power = 10 ** math.floor(math.log10(raw))
    fraction = raw / power
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * power


class ShapeSequence:
    """Read only geometry and class information; no interaction data is loaded."""

    def __init__(self, directory: Path, pattern: str) -> None:
        self.files = sorted(directory.glob(pattern), key=sequence_number)
        if not self.files:
            raise FileNotFoundError(f"No shapefiles match {pattern!r} in {directory}")
        self.bounds = self._sampled_bounds()

    def _sampled_bounds(self) -> tuple[float, float, float, float]:
        last = len(self.files) - 1
        count = min(len(self.files), 60)
        indices = sorted({round(i * last / max(count - 1, 1)) for i in range(count)})
        boxes = []
        for index in indices:
            with shapefile.Reader(str(self.files[index])) as reader:
                if len(reader):
                    boxes.append(tuple(reader.bbox))
        if not boxes:
            return (0.0, 0.0, 1.0, 1.0)
        return (
            min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes),
        )

    @lru_cache(maxsize=80)
    def frame(self, index: int) -> tuple[list[dict], int | None]:
        points: list[dict] = []
        frame_id = None
        with shapefile.Reader(str(self.files[index])) as reader:
            for item in reader.iterShapeRecords():
                if not item.shape.points:
                    continue
                attributes = item.record.as_dict()
                source_rgb = (attributes.get("R"), attributes.get("G"), attributes.get("B"))
                classification = COLOR_CLASSES.get(source_rgb)
                if not classification:
                    continue
                name, color = classification
                frame_id = attributes.get("FRAME_ID", frame_id)
                points.append({
                    "x": item.shape.points[0][0], "y": item.shape.points[0][1],
                    "name": name, "color": color,
                    "interpolated": bool(attributes.get("INTERP", False)),
                })
        return points, frame_id

# Model tuple: x intercept, x velocity, y intercept, y velocity, sample count.
MotionModel = tuple[float, float, float, float, int]


def fit_motion_model(observations: deque[tuple[int, float, float]]) -> MotionModel | None:
    """Fit x(t) and y(t) with a small online, recent-history regression."""
    if not observations:
        return None
    if len(observations) == 1:
        t, x, y = observations[0]
        return (x, 0.0, y, 0.0, 1)
    mean_t = sum(value[0] for value in observations) / len(observations)
    mean_x = sum(value[1] for value in observations) / len(observations)
    mean_y = sum(value[2] for value in observations) / len(observations)
    denominator = sum((value[0] - mean_t) ** 2 for value in observations)
    slope_x = sum((t - mean_t) * (x - mean_x) for t, x, _ in observations) / denominator
    slope_y = sum((t - mean_t) * (y - mean_y) for t, _, y in observations) / denominator
    return (mean_x - slope_x * mean_t, slope_x, mean_y - slope_y * mean_t, slope_y, len(observations))


def forecast(model: MotionModel | None, frame: int) -> tuple[float, float] | None:
    if model is None:
        return None
    intercept_x, slope_x, intercept_y, slope_y, _ = model
    return intercept_x + slope_x * frame, intercept_y + slope_y * frame


def build_causal_predictions(
    track: list[tuple[float, float] | None], window: int,
) -> tuple[list[tuple[float, float] | None], list[MotionModel | None], list[float | None], list[int]]:
    """Predict each frame before adding that frame's observation to the model."""
    observations: deque[tuple[int, float, float]] = deque(maxlen=window)
    predictions: list[tuple[float, float] | None] = []
    models: list[MotionModel | None] = []
    errors: list[float | None] = []
    sample_counts: list[int] = []
    total_samples = 0
    for frame, actual in enumerate(track):
        predicted = forecast(fit_motion_model(observations), frame)
        predictions.append(predicted)
        errors.append(math.dist(actual, predicted) if actual and predicted else None)
        if actual:
            observations.append((frame, actual[0], actual[1]))
            total_samples += 1
        models.append(fit_motion_model(observations))
        sample_counts.append(total_samples)
    return predictions, models, errors, sample_counts


class Viewer(tk.Tk):
    def __init__(
        self, sequence: ShapeSequence, entity: str, interval_ms: int,
        window: int, trail: int, horizon: int,
    ) -> None:
        super().__init__()
        self.sequence = sequence
        self.entity = entity
        self.interval_ms = interval_ms
        self.trail = trail
        self.horizon = horizon
        self.playing = False
        self.after_id: str | None = None
        self.observations: deque[tuple[int, float, float]] = deque(maxlen=window)
        self.track: list[tuple[float, float] | None] = []
        self.predictions: list[tuple[float, float] | None] = []
        self.models: list[MotionModel | None] = []
        self.errors: list[float | None] = []
        self.sample_counts: list[int] = []
        self.total_samples = 0

        self.title(f"Online trajectory prediction — {entity}")
        self.geometry("1000x720")
        self.minsize(620, 440)
        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

        controls = ttk.Frame(self, padding=(10, 7))
        controls.pack(fill="x")
        ttk.Button(controls, text="◀", width=4, command=lambda: self.move(-1)).pack(side="left")
        self.play_button = ttk.Button(controls, text="Play", width=7, command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(controls, text="▶", width=4, command=lambda: self.move(1)).pack(side="left")
        self.frame_var = tk.IntVar(value=1)
        self.slider = ttk.Scale(controls, from_=1, to=len(sequence.files), variable=self.frame_var,
                                command=lambda _value: self.draw())
        self.slider.pack(side="left", fill="x", expand=True, padx=12)
        self.status = ttk.Label(controls, width=43, anchor="e")
        self.status.pack(side="right")

        self.bind("<space>", lambda _event: self.toggle_play())
        self.bind("<Left>", lambda _event: self.move(-1))
        self.bind("<Right>", lambda _event: self.move(1))
        self.bind("<Home>", lambda _event: self.set_frame(0))
        self.bind("<End>", lambda _event: self.set_frame(len(self.sequence.files) - 1))
        self.after_idle(self.draw)

    @property
    def index(self) -> int:
        return max(0, min(len(self.sequence.files) - 1, round(self.frame_var.get()) - 1))

    def set_frame(self, index: int) -> None:
        self.frame_var.set(index + 1)
        self.draw()

    def move(self, amount: int) -> None:
        self.set_frame((self.index + amount) % len(self.sequence.files))

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()
        elif self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _tick(self) -> None:
        if self.playing:
            self.move(1)
            self.after_id = self.after(self.interval_ms, self._tick)

    def ensure_processed(self, index: int) -> None:
        """Advance the online model only as far as the user has viewed."""
        while len(self.track) <= index:
            frame = len(self.track)
            points, _frame_id = self.sequence.frame(frame)
            actual = next(
                ((point["x"], point["y"]) for point in points if point["name"] == self.entity),
                None,
            )
            predicted = forecast(fit_motion_model(self.observations), frame)
            self.track.append(actual)
            self.predictions.append(predicted)
            self.errors.append(math.dist(actual, predicted) if actual and predicted else None)
            if actual:
                self.observations.append((frame, actual[0], actual[1]))
                self.total_samples += 1
            self.models.append(fit_motion_model(self.observations))
            self.sample_counts.append(self.total_samples)

    def draw(self) -> None:
        if self.canvas.winfo_width() < 20 or self.canvas.winfo_height() < 20:
            return
        try:
            self.ensure_processed(self.index)
            _points, frame_id = self.sequence.frame(self.index)
        except Exception as exc:
            self.playing = False
            self.play_button.configure(text="Play")
            messagebox.showerror("Could not read shapefile", str(exc))
            return

        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        left, right, top, bottom = 78, 24, 42, 62
        _, _, x1, y1 = self.sequence.bounds
        x0, y0 = 0.0, 0.0
        x1, y1 = max(x1, 0.1) * 1.06, max(y1, 0.1) * 1.06
        plot_w, plot_h = max(width - left - right, 1), max(height - top - bottom, 1)

        def screen(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return left + x / x1 * plot_w, top + (y1 - y) / y1 * plot_h

        x_step, y_step = nice_step(x1), nice_step(y1)
        tick = 0.0
        while tick <= x1 + x_step * 1e-6:
            sx, _ = screen((tick, 0.0))
            self.canvas.create_line(sx, top, sx, top + plot_h, fill=GRID)
            self.canvas.create_text(sx, top + plot_h + 18, text=f"{tick:g}", fill=AXIS, font=("Segoe UI", 9))
            tick += x_step
        tick = 0.0
        while tick <= y1 + y_step * 1e-6:
            _, sy = screen((0.0, tick))
            self.canvas.create_line(left, sy, left + plot_w, sy, fill=GRID)
            self.canvas.create_text(left - 12, sy, text=f"{tick:g}", anchor="e", fill=AXIS, font=("Segoe UI", 9))
            tick += y_step
        self.canvas.create_rectangle(left, top, left + plot_w, top + plot_h, outline=AXIS)
        self.canvas.create_text(left + plot_w / 2, height - 17, text="X (m)", fill=AXIS, font=("Segoe UI", 10))
        self.canvas.create_text(18, top + plot_h / 2, text="Y (m)", angle=90, fill=AXIS, font=("Segoe UI", 10))

        start = max(0, self.index - self.trail + 1)
        actual_screen = [screen(point) for point in self.track[start:self.index + 1] if point]
        predicted_screen = [screen(point) for point in self.predictions[start:self.index + 1] if point]
        if len(actual_screen) > 1:
            self.canvas.create_line(*sum(actual_screen, ()), fill=ACTUAL_COLOR, width=3, smooth=True)
        if len(predicted_screen) > 1:
            self.canvas.create_line(*sum(predicted_screen, ()), fill=PREDICTED_COLOR, width=2, dash=(6, 4), smooth=True)

        current_actual = self.track[self.index]
        current_prediction = self.predictions[self.index]
        if current_prediction:
            px, py = screen(current_prediction)
            self.canvas.create_line(px - 6, py - 6, px + 6, py + 6, fill=PREDICTED_COLOR, width=2)
            self.canvas.create_line(px - 6, py + 6, px + 6, py - 6, fill=PREDICTED_COLOR, width=2)
        if current_actual:
            ax, ay = screen(current_actual)
            radius = 7
            color = COLOR_CLASSES[NAME_TO_RGB[self.entity]][1]
            self.canvas.create_oval(ax - radius, ay - radius, ax + radius, ay + radius,
                                    fill=color, outline=ACTUAL_COLOR, width=3)
            self.canvas.create_text(ax + 12, ay, text=self.entity, anchor="w", fill="#17202a",
                                    font=("Segoe UI Semibold", 9))

        model = self.models[self.index]
        future = [forecast(model, frame) for frame in range(self.index + 1,
                  min(self.index + self.horizon + 1, len(self.sequence.files)))]
        future_screen = [screen(point) for point in future if point]
        origin = screen(current_actual) if current_actual else (predicted_screen[-1] if predicted_screen else None)
        if origin and future_screen:
            self.canvas.create_line(*(origin + sum(future_screen, ())), fill=PREDICTED_COLOR,
                                    width=3, dash=(3, 4), arrow="last")
            for px, py in future_screen[::max(1, len(future_screen) // 5)]:
                self.canvas.create_oval(px - 2, py - 2, px + 2, py + 2,
                                        fill=BACKGROUND, outline=PREDICTED_COLOR)

        self.canvas.create_line(left, 18, left + 24, 18, fill=ACTUAL_COLOR, width=3)
        self.canvas.create_text(left + 30, 18, text="actual", anchor="w", fill=ACTUAL_COLOR,
                                font=("Segoe UI Semibold", 9))
        self.canvas.create_line(left + 105, 18, left + 129, 18, fill=PREDICTED_COLOR, width=2, dash=(5, 3))
        self.canvas.create_text(left + 135, 18, text="causal prediction / forecast", anchor="w",
                                fill=PREDICTED_COLOR, font=("Segoe UI Semibold", 9))

        observed_errors = [error for error in self.errors[:self.index + 1] if error is not None]
        current_error = self.errors[self.index]
        error_text = "cold start" if current_error is None else f"error {current_error:.3f} m"
        mean_text = f" • mean {sum(observed_errors) / len(observed_errors):.3f} m" if observed_errors else ""
        source_text = f" • source frame {frame_id}" if frame_id is not None else ""
        self.status.configure(text=(f"{self.index + 1} / {len(self.sequence.files)} • "
                                    f"{self.sample_counts[self.index]} samples • {error_text}{mean_text}{source_text}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent,
                        help="directory containing the shapefiles")
    parser.add_argument("--pattern", default="camera01_centroids_*.shp", help="shapefile glob pattern")
    parser.add_argument("--entity", choices=sorted(NAME_TO_RGB), default="instrument_table",
                        help="single entity whose trajectory is modeled")
    parser.add_argument("--window", type=int, default=45, help="recent observations used by the online model")
    parser.add_argument("--trail", type=int, default=100, help="past frames drawn")
    parser.add_argument("--horizon", type=int, default=30, help="future frames forecast")
    parser.add_argument("--interval", type=int, default=100, help="animation interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    if args.directory == Path(__file__).parent and (directory / "shp").is_dir():
        directory = directory / "shp"
    try:
        sequence = ShapeSequence(directory, args.pattern)
        viewer = Viewer(sequence, args.entity, max(args.interval, 10),
                        max(args.window, 2), max(args.trail, 2), max(args.horizon, 1))
    except (FileNotFoundError, ValueError, shapefile.ShapefileException) as exc:
        raise SystemExit(str(exc)) from exc
    viewer.mainloop()


if __name__ == "__main__":
    main()
