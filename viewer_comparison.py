"""Compare causal cold-start trajectory predictors on one moving entity."""

from __future__ import annotations

import argparse
import math
import random
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import shapefile

from viewer_prediction import (
    AXIS,
    BACKGROUND,
    COLOR_CLASSES,
    GRID,
    NAME_TO_RGB,
    ShapeSequence,
    fit_motion_model,
    forecast,
    nice_step,
)


ACTUAL_COLOR = "#111827"
ALGORITHM_STYLES = {
    "Persistence": ("#64748b", (2, 4)),
    "Constant velocity": ("#f59e0b", (7, 4)),
    "Rolling regression": ("#7c3aed", (5, 3)),
    "Alpha-beta filter": ("#059669", (9, 3)),
    "Online neural net": ("#db2777", (4, 3)),
    "Memory k-NN": ("#0891b2", (2, 3)),
    "Adaptive ensemble": ("#dc2626", (8, 3)),
}

Point = tuple[float, float]


class Predictor:
    def predict(self, frame: int) -> Point | None:
        raise NotImplementedError

    def update(self, frame: int, actual: Point) -> None:
        raise NotImplementedError


class PersistencePredictor(Predictor):
    """Cold-start baseline: the entity stays at its last observed position."""

    def __init__(self) -> None:
        self.last: Point | None = None

    def predict(self, frame: int) -> Point | None:
        return self.last

    def update(self, frame: int, actual: Point) -> None:
        self.last = actual


class VelocityPredictor(Predictor):
    """Exponential smoothing of observed frame-to-frame velocity."""

    def __init__(self, smoothing: float = 0.35) -> None:
        self.smoothing = smoothing
        self.last_frame: int | None = None
        self.last: Point | None = None
        self.velocity: Point = (0.0, 0.0)

    def predict(self, frame: int) -> Point | None:
        if self.last is None or self.last_frame is None:
            return None
        elapsed = frame - self.last_frame
        return self.last[0] + self.velocity[0] * elapsed, self.last[1] + self.velocity[1] * elapsed

    def update(self, frame: int, actual: Point) -> None:
        if self.last is not None and self.last_frame is not None and frame > self.last_frame:
            elapsed = frame - self.last_frame
            measured = ((actual[0] - self.last[0]) / elapsed, (actual[1] - self.last[1]) / elapsed)
            weight = self.smoothing
            self.velocity = (
                (1 - weight) * self.velocity[0] + weight * measured[0],
                (1 - weight) * self.velocity[1] + weight * measured[1],
            )
        self.last, self.last_frame = actual, frame


class RegressionPredictor(Predictor):
    """Rolling least-squares X/Y velocity model."""

    def __init__(self, window: int) -> None:
        self.observations: deque[tuple[int, float, float]] = deque(maxlen=window)

    def predict(self, frame: int) -> Point | None:
        return forecast(fit_motion_model(self.observations), frame)

    def update(self, frame: int, actual: Point) -> None:
        self.observations.append((frame, actual[0], actual[1]))


class AlphaBetaPredictor(Predictor):
    """A lightweight tracking filter that corrects position and velocity."""

    def __init__(self, alpha: float = 0.65, beta: float = 0.18) -> None:
        self.alpha, self.beta = alpha, beta
        self.frame: int | None = None
        self.position: Point | None = None
        self.velocity: Point = (0.0, 0.0)

    def predict(self, frame: int) -> Point | None:
        if self.position is None or self.frame is None:
            return None
        elapsed = frame - self.frame
        return (self.position[0] + self.velocity[0] * elapsed,
                self.position[1] + self.velocity[1] * elapsed)

    def update(self, frame: int, actual: Point) -> None:
        if self.position is None or self.frame is None:
            self.position, self.frame = actual, frame
            return
        elapsed = max(frame - self.frame, 1)
        predicted = self.predict(frame)
        assert predicted is not None
        residual = actual[0] - predicted[0], actual[1] - predicted[1]
        self.position = (predicted[0] + self.alpha * residual[0],
                         predicted[1] + self.alpha * residual[1])
        self.velocity = (self.velocity[0] + self.beta * residual[0] / elapsed,
                         self.velocity[1] + self.beta * residual[1] / elapsed)
        self.frame = frame


class OnlineNeuralPredictor(Predictor):
    """Tiny MLP trained by SGD to predict the next per-frame displacement."""

    def __init__(self, hidden_size: int = 10, learning_rate: float = 0.04) -> None:
        rng = random.Random(1)  # Reproducible comparisons between runs.
        self.learning_rate = learning_rate
        self.input_size = 6
        self.hidden_size = hidden_size
        self.input_weights = [
            [rng.uniform(-0.12, 0.12) for _ in range(self.input_size + 1)]
            for _ in range(hidden_size)
        ]
        # Broad random output biases intentionally make the untrained model
        # visibly poor; SGD must earn its place in the comparison over time.
        self.output_weights = [
            [rng.uniform(-0.03, 0.03) for _ in range(hidden_size)] + [rng.uniform(-0.8, 0.8)]
            for _ in range(2)
        ]
        self.last_frame: int | None = None
        self.last_position: Point | None = None
        self.velocity: Point = (0.0, 0.0)
        self.previous_velocity: Point = (0.0, 0.0)
        self.training_samples = 0

    @staticmethod
    def _bounded(value: float) -> float:
        return math.tanh(value * 5.0)

    def _features(self) -> list[float]:
        acceleration = (self.velocity[0] - self.previous_velocity[0],
                        self.velocity[1] - self.previous_velocity[1])
        speed = math.hypot(*self.velocity)
        return [
            self._bounded(self.velocity[0]), self._bounded(self.velocity[1]),
            self._bounded(acceleration[0]), self._bounded(acceleration[1]),
            self._bounded(speed), min(self.training_samples / 100.0, 1.0),
        ]

    def _forward(self, features: list[float]) -> tuple[list[float], list[float]]:
        augmented = features + [1.0]
        hidden = [math.tanh(sum(weight * value for weight, value in zip(row, augmented)))
                  for row in self.input_weights]
        hidden_augmented = hidden + [1.0]
        output = [sum(weight * value for weight, value in zip(row, hidden_augmented))
                  for row in self.output_weights]
        return hidden, output

    def predict(self, frame: int) -> Point | None:
        if self.last_position is None or self.last_frame is None:
            return None
        _hidden, output = self._forward(self._features())
        # The network learns scaled velocity; convert back to metres/frame.
        elapsed = frame - self.last_frame
        return (self.last_position[0] + output[0] / 5.0 * elapsed,
                self.last_position[1] + output[1] / 5.0 * elapsed)

    def update(self, frame: int, actual: Point) -> None:
        if self.last_position is None or self.last_frame is None:
            self.last_position, self.last_frame = actual, frame
            return
        elapsed = max(frame - self.last_frame, 1)
        target_velocity = ((actual[0] - self.last_position[0]) / elapsed,
                           (actual[1] - self.last_position[1]) / elapsed)
        features = self._features()
        hidden, output = self._forward(features)
        targets = [max(-2.0, min(2.0, target_velocity[axis] * 5.0)) for axis in range(2)]
        errors = [max(-2.0, min(2.0, output[axis] - targets[axis])) for axis in range(2)]

        # Back-propagate before changing the output weights.
        hidden_gradients = []
        for hidden_index, hidden_value in enumerate(hidden):
            downstream = sum(errors[axis] * self.output_weights[axis][hidden_index] for axis in range(2))
            hidden_gradients.append(downstream * (1.0 - hidden_value * hidden_value))
        rate = self.learning_rate
        hidden_augmented = hidden + [1.0]
        for axis in range(2):
            for index, value in enumerate(hidden_augmented):
                self.output_weights[axis][index] -= rate * errors[axis] * value
        feature_augmented = features + [1.0]
        for hidden_index in range(self.hidden_size):
            for index, value in enumerate(feature_augmented):
                self.input_weights[hidden_index][index] -= rate * hidden_gradients[hidden_index] * value

        self.previous_velocity, self.velocity = self.velocity, target_velocity
        self.last_position, self.last_frame = actual, frame
        self.training_samples += 1


class KNearestMotionPredictor(Predictor):
    """Data-hungry instance learner over previously observed motion contexts."""

    def __init__(self, neighbors: int = 9, minimum_samples: int = 40, memory: int = 1200) -> None:
        self.neighbors = neighbors
        self.minimum_samples = minimum_samples
        self.examples: deque[tuple[tuple[float, ...], Point]] = deque(maxlen=memory)
        self.last_frame: int | None = None
        self.last_position: Point | None = None
        self.velocity: Point = (0.0, 0.0)
        self.previous_velocity: Point = (0.0, 0.0)
        self._cached_neighbor_velocity: Point | None = None

    def _features(self) -> tuple[float, ...]:
        acceleration = (self.velocity[0] - self.previous_velocity[0],
                        self.velocity[1] - self.previous_velocity[1])
        return (
            math.tanh(self.velocity[0] * 5.0), math.tanh(self.velocity[1] * 5.0),
            math.tanh(acceleration[0] * 5.0), math.tanh(acceleration[1] * 5.0),
            math.tanh(math.hypot(*self.velocity) * 5.0),
        )

    @property
    def training_samples(self) -> int:
        return len(self.examples)

    def predict(self, frame: int) -> Point | None:
        if (self.last_position is None or self.last_frame is None or
                len(self.examples) < self.minimum_samples):
            return None
        if self._cached_neighbor_velocity is None:
            features = self._features()
            distances = []
            for stored_features, target_velocity in self.examples:
                distance = sum((left - right) ** 2 for left, right in zip(features, stored_features))
                distances.append((distance, target_velocity))
            nearest = sorted(distances, key=lambda item: item[0])[:self.neighbors]
            weights = [1.0 / (math.sqrt(distance) + 0.03) for distance, _ in nearest]
            total_weight = sum(weights)
            self._cached_neighbor_velocity = (
                sum(weight * target[0] for weight, (_, target) in zip(weights, nearest)) / total_weight,
                sum(weight * target[1] for weight, (_, target) in zip(weights, nearest)) / total_weight,
            )
        velocity_x, velocity_y = self._cached_neighbor_velocity
        elapsed = frame - self.last_frame
        return (self.last_position[0] + velocity_x * elapsed,
                self.last_position[1] + velocity_y * elapsed)

    def update(self, frame: int, actual: Point) -> None:
        if self.last_position is None or self.last_frame is None:
            self.last_position, self.last_frame = actual, frame
            return
        elapsed = max(frame - self.last_frame, 1)
        target_velocity = ((actual[0] - self.last_position[0]) / elapsed,
                           (actual[1] - self.last_position[1]) / elapsed)
        self.examples.append((self._features(), target_velocity))
        self.previous_velocity, self.velocity = self.velocity, target_velocity
        self.last_position, self.last_frame = actual, frame
        self._cached_neighbor_velocity = None


class AdaptiveEnsemblePredictor(Predictor):
    """Online mixture of persistence and several smoothed-velocity experts."""

    def __init__(self, learning_rate: float = 12.0) -> None:
        # Alpha zero is exact persistence; larger values react more strongly to motion.
        self.alphas = (0.0, 0.03, 0.08, 0.20, 0.50, 1.0)
        self.velocities: list[Point] = [(0.0, 0.0) for _ in self.alphas]
        self.weights = [1.0 / len(self.alphas) for _ in self.alphas]
        self.learning_rate = learning_rate
        self.last_frame: int | None = None
        self.last_position: Point | None = None
        self.training_samples = 0

    def _expert_predictions(self, frame: int) -> list[Point]:
        assert self.last_position is not None and self.last_frame is not None
        elapsed = frame - self.last_frame
        return [(self.last_position[0] + velocity[0] * elapsed,
                 self.last_position[1] + velocity[1] * elapsed)
                for velocity in self.velocities]

    def predict(self, frame: int) -> Point | None:
        if self.last_position is None or self.last_frame is None:
            return None
        predictions = self._expert_predictions(frame)
        return (
            sum(weight * point[0] for weight, point in zip(self.weights, predictions)),
            sum(weight * point[1] for weight, point in zip(self.weights, predictions)),
        )

    def update(self, frame: int, actual: Point) -> None:
        if self.last_position is None or self.last_frame is None:
            self.last_position, self.last_frame = actual, frame
            return
        predictions = self._expert_predictions(frame)
        # Exponential weighting rewards experts using only the error just revealed.
        updated = []
        for weight, prediction in zip(self.weights, predictions):
            clipped_loss = min(math.dist(actual, prediction), 1.0)
            updated.append(max(weight * math.exp(-self.learning_rate * clipped_loss), 1e-15))
        total = sum(updated)
        self.weights = [weight / total for weight in updated]

        elapsed = max(frame - self.last_frame, 1)
        measured = ((actual[0] - self.last_position[0]) / elapsed,
                    (actual[1] - self.last_position[1]) / elapsed)
        self.velocities = [
            ((1 - alpha) * velocity[0] + alpha * measured[0],
             (1 - alpha) * velocity[1] + alpha * measured[1])
            for alpha, velocity in zip(self.alphas, self.velocities)
        ]
        self.last_position, self.last_frame = actual, frame
        self.training_samples += 1

    @property
    def persistence_weight(self) -> float:
        return self.weights[0]


class ComparisonViewer(tk.Tk):
    def __init__(
        self, sequence: ShapeSequence, entity: str, interval_ms: int,
        window: int, trail: int, horizon: int,
    ) -> None:
        super().__init__()
        self.sequence, self.entity = sequence, entity
        self.interval_ms, self.trail, self.horizon = interval_ms, trail, horizon
        self.prediction_steps = 1
        self.prediction_dots_only = False
        self.playing = False
        self.after_id: str | None = None
        full_x = max(sequence.bounds[2], 0.1) * 1.06
        full_y = max(sequence.bounds[3], 0.1) * 1.06
        self.zoom = 1.0
        self.view_center: Point = (full_x / 2, full_y / 2)
        self.predictors: dict[str, Predictor] = {
            "Persistence": PersistencePredictor(),
            "Constant velocity": VelocityPredictor(),
            "Rolling regression": RegressionPredictor(window),
            "Alpha-beta filter": AlphaBetaPredictor(),
            "Online neural net": OnlineNeuralPredictor(),
            "Memory k-NN": KNearestMotionPredictor(),
            "Adaptive ensemble": AdaptiveEnsemblePredictor(),
        }
        self.track: list[Point | None] = []
        self.predictions: dict[str, list[Point | None]] = {name: [] for name in self.predictors}
        self.errors: dict[str, list[float | None]] = {name: [] for name in self.predictors}
        self.forecasts: dict[str, list[list[Point]]] = {name: [] for name in self.predictors}
        self.sample_counts: list[int] = []
        self.total_samples = 0

        self.title(f"Trajectory predictor comparison — {entity}")
        self.geometry("1180x740")
        self.minsize(760, 600)
        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_by(1.25, event))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_by(0.8, event))

        controls = ttk.Frame(self, padding=(10, 7))
        controls.pack(fill="x")
        ttk.Button(controls, text="◀", width=4, command=lambda: self.move(-1)).pack(side="left")
        self.play_button = ttk.Button(controls, text="Play", width=7, command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(controls, text="▶", width=4, command=lambda: self.move(1)).pack(side="left")
        ttk.Button(controls, text="−", width=3, command=lambda: self.zoom_by(0.8)).pack(side="left", padx=(10, 1))
        ttk.Button(controls, text="100%", width=6, command=self.reset_zoom).pack(side="left", padx=1)
        ttk.Button(controls, text="+", width=3, command=lambda: self.zoom_by(1.25)).pack(side="left", padx=(1, 5))
        self.frame_var = tk.IntVar(value=1)
        self.slider = ttk.Scale(controls, from_=1, to=len(sequence.files), variable=self.frame_var,
                                command=lambda _value: self.draw())
        self.slider.pack(side="left", fill="x", expand=True, padx=12)
        self.status = ttk.Label(controls, width=34, anchor="e")
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

    def plot_geometry(self) -> tuple[float, float, float, float, float, float]:
        left, panel_w, top, bottom = 78.0, 275.0, 48.0, 62.0
        plot_w = max(self.canvas.winfo_width() - left - panel_w - 24, 1)
        plot_h = max(self.canvas.winfo_height() - top - bottom, 1)
        return left, panel_w, top, bottom, plot_w, plot_h

    def view_bounds(self) -> tuple[float, float, float, float]:
        full_x = max(self.sequence.bounds[2], 0.1) * 1.06
        full_y = max(self.sequence.bounds[3], 0.1) * 1.06
        span_x, span_y = full_x / self.zoom, full_y / self.zoom
        half_x, half_y = span_x / 2, span_y / 2
        center_x = min(max(self.view_center[0], half_x), full_x - half_x)
        center_y = min(max(self.view_center[1], half_y), full_y - half_y)
        self.view_center = center_x, center_y
        return center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y

    def on_mouse_wheel(self, event: tk.Event) -> None:
        self.zoom_by(1.25 if event.delta > 0 else 0.8, event)

    def zoom_by(self, factor: float, event: tk.Event | None = None) -> None:
        left, _panel_w, top, _bottom, plot_w, plot_h = self.plot_geometry()
        old_x0, old_y0, old_x1, old_y1 = self.view_bounds()
        ratio_x = 0.5
        ratio_y = 0.5
        if event and left <= event.x <= left + plot_w and top <= event.y <= top + plot_h:
            ratio_x = (event.x - left) / plot_w
            ratio_y = (event.y - top) / plot_h
        anchor_x = old_x0 + ratio_x * (old_x1 - old_x0)
        anchor_y = old_y1 - ratio_y * (old_y1 - old_y0)
        self.zoom = min(max(self.zoom * factor, 1.0), 20.0)
        full_x = max(self.sequence.bounds[2], 0.1) * 1.06
        full_y = max(self.sequence.bounds[3], 0.1) * 1.06
        new_span_x, new_span_y = full_x / self.zoom, full_y / self.zoom
        self.view_center = (
            anchor_x + (0.5 - ratio_x) * new_span_x,
            anchor_y + (ratio_y - 0.5) * new_span_y,
        )
        self.view_bounds()
        self.draw()

    def reset_zoom(self) -> None:
        full_x = max(self.sequence.bounds[2], 0.1) * 1.06
        full_y = max(self.sequence.bounds[3], 0.1) * 1.06
        self.zoom = 1.0
        self.view_center = full_x / 2, full_y / 2
        self.draw()

    def ensure_processed(self, index: int) -> None:
        """Make every prediction before revealing the corresponding observation."""
        while len(self.track) <= index:
            frame = len(self.track)
            points, _ = self.sequence.frame(frame)
            actual = next(((p["x"], p["y"]) for p in points if p["name"] == self.entity), None)
            self.track.append(actual)
            if actual:
                self.total_samples += 1
            for name, predictor in self.predictors.items():
                predicted = predictor.predict(frame)
                self.predictions[name].append(predicted)
                self.errors[name].append(math.dist(actual, predicted) if actual and predicted else None)
                if actual:
                    predictor.update(frame, actual)
                self.forecasts[name].append([
                    point for future_frame in range(frame + 1, min(frame + self.horizon + 1, len(self.sequence.files)))
                    if (point := predictor.predict(future_frame)) is not None
                ])
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
        left, panel_w, top, bottom, plot_w, plot_h = self.plot_geometry()
        x0, y0, x1, y1 = self.view_bounds()

        def screen(point: Point) -> Point:
            return (left + (point[0] - x0) / (x1 - x0) * plot_w,
                    top + (y1 - point[1]) / (y1 - y0) * plot_h)

        def visible(point: Point) -> bool:
            return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

        x_step, y_step = nice_step(x1 - x0), nice_step(y1 - y0)
        tick = math.ceil(x0 / x_step) * x_step
        while tick <= x1 + x_step * 1e-6:
            sx, _ = screen((tick, y0))
            self.canvas.create_line(sx, top, sx, top + plot_h, fill=GRID)
            self.canvas.create_text(sx, top + plot_h + 18, text=f"{tick:g}", fill=AXIS, font=("Segoe UI", 9))
            tick += x_step
        tick = math.ceil(y0 / y_step) * y_step
        while tick <= y1 + y_step * 1e-6:
            _, sy = screen((x0, tick))
            self.canvas.create_line(left, sy, left + plot_w, sy, fill=GRID)
            self.canvas.create_text(left - 12, sy, text=f"{tick:g}", anchor="e", fill=AXIS, font=("Segoe UI", 9))
            tick += y_step
        self.canvas.create_rectangle(left, top, left + plot_w, top + plot_h, outline=AXIS)
        self.canvas.create_text(left + plot_w / 2, height - 17, text="X (m)", fill=AXIS, font=("Segoe UI", 10))
        self.canvas.create_text(18, top + plot_h / 2, text="Y (m)", angle=90, fill=AXIS, font=("Segoe UI", 10))
        self.canvas.create_text(left + plot_w - 8, top + 8, anchor="ne", text=f"zoom {self.zoom:.2f}×",
                                fill=AXIS, font=("Segoe UI Semibold", 9))

        start = max(0, self.index - self.trail + 1)
        actual_path = [screen(point) for point in self.track[start:self.index + 1] if point and visible(point)]
        if len(actual_path) > 1:
            self.canvas.create_line(*sum(actual_path, ()), fill=ACTUAL_COLOR, width=4, smooth=True)
        current_actual = self.track[self.index]
        if current_actual and visible(current_actual):
            ax, ay = screen(current_actual)
            fill = COLOR_CLASSES[NAME_TO_RGB[self.entity]][1]
            self.canvas.create_oval(ax - 8, ay - 8, ax + 8, ay + 8, fill=fill, outline=ACTUAL_COLOR, width=3)
            self.canvas.create_text(ax + 13, ay, text=self.entity, anchor="w", fill=ACTUAL_COLOR,
                                    font=("Segoe UI Semibold", 9))

        for name in self.predictors:
            color, dash = ALGORITHM_STYLES[name]
            predicted_path = [screen(point) for point in self.predictions[name][start:self.index + 1]
                              if point and visible(point)]
            if self.prediction_dots_only:
                for px, py in predicted_path:
                    self.canvas.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5,
                                            fill=color, outline=BACKGROUND, width=1)
                if predicted_path:
                    px, py = predicted_path[-1]
                    self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4,
                                            fill=color, outline=BACKGROUND, width=1)
            elif len(predicted_path) > 1:
                self.canvas.create_line(*sum(predicted_path, ()), fill=color, width=2, dash=dash, smooth=True)
            future_path = [screen(point) for point in self.forecasts[name][self.index] if visible(point)]
            origin = screen(current_actual) if current_actual and visible(current_actual) else (
                predicted_path[-1] if predicted_path else None
            )
            if origin and future_path:
                if not self.prediction_dots_only:
                    self.canvas.create_line(*(origin + sum(future_path, ())), fill=color, width=2, dash=dash)
                spacing = 1 if self.prediction_dots_only else max(1, len(future_path) // 5)
                for px, py in future_path[::spacing]:
                    self.canvas.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5,
                                            fill=color, outline=BACKGROUND, width=1)
                end_x, end_y = future_path[-1]
                self.canvas.create_oval(end_x - 4, end_y - 4, end_x + 4, end_y + 4,
                                        fill=color, outline=BACKGROUND, width=1)

        self.draw_legend(left, 18)
        self.draw_metrics(left + plot_w + 18, top, panel_w)
        source = f" • source frame {frame_id}" if frame_id is not None else ""
        self.status.configure(text=f"{self.index + 1} / {len(self.sequence.files)} • "
                                   f"{self.sample_counts[self.index]} observations{source}")

    def draw_legend(self, x: float, y: float) -> None:
        items = [("actual", ACTUAL_COLOR, (), 4)] + [
            (name, *ALGORITHM_STYLES[name], 2) for name in self.predictors
        ]
        for index, (name, color, dash, width) in enumerate(items):
            item_x = x + (index % 4) * 170
            item_y = y + (index // 4) * 17
            self.canvas.create_line(item_x, item_y, item_x + 22, item_y,
                                    fill=color, width=width, dash=dash)
            self.canvas.create_text(item_x + 27, item_y, text=name, anchor="w", fill=color,
                                    font=("Segoe UI Semibold", 8))

    def draw_metrics(self, x: float, y: float, width: float) -> None:
        horizon_label = "one-step" if self.prediction_steps == 1 else f"{self.prediction_steps}-frame-ahead"
        self.canvas.create_text(x, y, anchor="nw", text=f"Causal {horizon_label} comparison",
                                fill="#17202a", font=("Segoe UI Semibold", 11))
        self.canvas.create_text(x, y + 22, anchor="nw", text="Each estimate precedes its observation",
                                fill=AXIS, font=("Segoe UI", 8))
        ranking = []
        for name in self.predictors:
            values = [value for value in self.errors[name][:self.index + 1] if value is not None]
            mae = sum(values) / len(values) if values else math.inf
            rmse = math.sqrt(sum(value * value for value in values) / len(values)) if values else math.inf
            ranking.append((mae, name, rmse, values))
        row_y = y + 58
        for rank, (mae, name, rmse, values) in enumerate(sorted(ranking), 1):
            color = ALGORITHM_STYLES[name][0]
            self.canvas.create_text(x, row_y, anchor="nw", text=f"{rank}. {name}", fill=color,
                                    font=("Segoe UI Semibold", 10))
            if values:
                current = self.errors[name][self.index]
                current_text = f"current {current:.3f} m" if current is not None else "current: no observation"
                trained = ""
                learned = max(self.sample_counts[self.index] - 1, 0)
                if isinstance(self.predictors[name], OnlineNeuralPredictor):
                    trained = f"   trained={learned}"
                elif isinstance(self.predictors[name], KNearestMotionPredictor):
                    trained = f"   memory={min(learned, self.predictors[name].examples.maxlen)}"
                elif isinstance(self.predictors[name], AdaptiveEnsemblePredictor):
                    trained = f"   adapted={learned}"
                detail = f"MAE {mae:.3f} m   RMSE {rmse:.3f} m\n{current_text}   n={len(values)}{trained}"
            else:
                if isinstance(self.predictors[name], KNearestMotionPredictor):
                    available = max(self.sample_counts[self.index] - 1, 0)
                    detail = f"data warm-up: {available}/{self.predictors[name].minimum_samples} examples"
                else:
                    detail = "cold start — waiting for a test observation"
            self.canvas.create_text(x + 12, row_y + 21, anchor="nw", text=detail,
                                    width=width - 18, fill="#475569", font=("Segoe UI", 8))
            row_y += 62
        self.canvas.create_text(x, row_y + 8, anchor="nw",
                                text="Lower MAE ranks higher. Missing detections\nare skipped for every algorithm.",
                                fill=AXIS, font=("Segoe UI", 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent,
                        help="directory containing the shapefiles")
    parser.add_argument("--pattern", default="camera01_centroids_*.shp", help="shapefile glob pattern")
    parser.add_argument("--entity", choices=sorted(NAME_TO_RGB), default="circulator",
                        help="single entity evaluated by all predictors")
    parser.add_argument("--window", type=int, default=45, help="rolling-regression observations")
    parser.add_argument("--trail", type=int, default=80, help="past frames displayed")
    parser.add_argument("--horizon", type=int, default=20, help="future frames forecast")
    parser.add_argument("--interval", type=int, default=100, help="animation interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    if args.directory == Path(__file__).parent and (directory / "shp").is_dir():
        directory = directory / "shp"
    try:
        sequence = ShapeSequence(directory, args.pattern)
        viewer = ComparisonViewer(sequence, args.entity, max(args.interval, 10),
                                  max(args.window, 2), max(args.trail, 2), max(args.horizon, 1))
    except (FileNotFoundError, ValueError, shapefile.ShapefileException) as exc:
        raise SystemExit(str(exc)) from exc
    viewer.mainloop()


if __name__ == "__main__":
    main()
