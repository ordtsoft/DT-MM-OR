"""Clear, reduced-model comparison for configurable multi-frame prediction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import shapefile

from viewer_comparison import ComparisonViewer
from viewer_prediction import NAME_TO_RGB, ShapeSequence


DEFAULT_STEPS = 10
VISIBLE_MODELS = ("Persistence", "Online neural net", "Adaptive ensemble")


class MultiFrameComparisonViewer(ComparisonViewer):
    """Show one prediction dot per target frame for three selected models."""

    def __init__(self, *args, prediction_steps: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_steps = prediction_steps
        self.prediction_dots_only = True
        self.predictors = {name: self.predictors[name] for name in VISIBLE_MODELS}
        self.predictions = {name: self.predictions[name] for name in VISIBLE_MODELS}
        self.errors = {name: self.errors[name] for name in VISIBLE_MODELS}
        self.forecasts = {name: self.forecasts[name] for name in VISIBLE_MODELS}
        self.model_filter.configure(values=("All models", *self.predictors))
        self.display_model_var.set("All models")
        self.scheduled_predictions: dict[str, dict[int, tuple[float, float]]] = {
            name: {} for name in self.predictors
        }
        self.title(f"{prediction_steps}-frame trajectory prediction — {self.entity}")

    def ensure_processed(self, index: int) -> None:
        while len(self.track) <= index:
            frame = len(self.track)
            points, _ = self.sequence.frame(frame)
            actual = next(((point["x"], point["y"]) for point in points
                           if point["name"] == self.entity), None)
            self.track.append(actual)
            if actual:
                self.total_samples += 1

            for name, predictor in self.predictors.items():
                # Retrieve the prediction frozen `prediction_steps` frames ago.
                predicted = self.scheduled_predictions[name].pop(frame, None)
                self.predictions[name].append(predicted)
                self.errors[name].append(math.dist(actual, predicted) if actual and predicted else None)

                if actual:
                    predictor.update(frame, actual)

                target = frame + self.prediction_steps
                target_prediction = predictor.predict(target) if target < len(self.sequence.files) else None
                if target_prediction is not None:
                    self.scheduled_predictions[name][target] = target_prediction

                # Only one future dot: the currently scheduled target frame.
                self.forecasts[name].append([target_prediction] if target_prediction else [])
            self.sample_counts.append(self.total_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent,
                        help="directory containing the shapefiles")
    parser.add_argument("--pattern", default="camera01_centroids_*.shp", help="shapefile glob pattern")
    parser.add_argument("--entity", choices=sorted(NAME_TO_RGB), default="circulator",
                        help="single entity evaluated by all predictors")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="number of frames ahead to predict (default: 10)")
    parser.add_argument("--window", type=int, default=45, help="regression window inherited by the viewer")
    parser.add_argument("--trail", type=int, default=60, help="past target frames displayed")
    parser.add_argument("--interval", type=int, default=100, help="animation interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    if args.directory == Path(__file__).parent and (directory / "shp").is_dir():
        directory = directory / "shp"
    steps = max(args.steps, 2)
    try:
        sequence = ShapeSequence(directory, args.pattern)
        viewer = MultiFrameComparisonViewer(
            sequence, args.entity, max(args.interval, 10),
            max(args.window, 2), max(args.trail, 2), steps,
            prediction_steps=steps,
        )
    except (FileNotFoundError, ValueError, shapefile.ShapefileException) as exc:
        raise SystemExit(str(exc)) from exc
    viewer.mainloop()


if __name__ == "__main__":
    main()
