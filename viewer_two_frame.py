"""Compare strictly causal two-frame-ahead trajectory predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import shapefile

from viewer_comparison import ComparisonViewer
from viewer_prediction import NAME_TO_RGB, ShapeSequence


PREDICTION_STEPS = 2


class TwoFrameComparisonViewer(ComparisonViewer):
    """Freeze each prediction two frames before its target is observed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_steps = PREDICTION_STEPS
        self.scheduled_predictions: dict[str, dict[int, tuple[float, float]]] = {
            name: {} for name in self.predictors
        }
        self.title(f"Two-frame trajectory prediction — {self.entity}")

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
                # This value was generated and frozen at frame - 2.
                predicted = self.scheduled_predictions[name].pop(frame, None)
                self.predictions[name].append(predicted)
                self.errors[name].append(math.dist(actual, predicted) if actual and predicted else None)

                # Reveal the current observation only after scoring its old prediction.
                if actual:
                    predictor.update(frame, actual)

                target_frame = frame + PREDICTION_STEPS
                if target_frame < len(self.sequence.files):
                    future_prediction = predictor.predict(target_frame)
                    if future_prediction is not None:
                        self.scheduled_predictions[name][target_frame] = future_prediction

                self.forecasts[name].append([
                    point for future_frame in range(frame + 1,
                    min(frame + self.horizon + 1, len(self.sequence.files)))
                    if (point := predictor.predict(future_frame)) is not None
                ])
            self.sample_counts.append(self.total_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent,
                        help="directory containing the shapefiles")
    parser.add_argument("--pattern", default="camera01_centroids_*.shp", help="shapefile glob pattern")
    parser.add_argument("--entity", choices=sorted(NAME_TO_RGB), default="circulator",
                        help="single entity evaluated by all predictors")
    parser.add_argument("--window", type=int, default=45, help="rolling-regression observations")
    parser.add_argument("--trail", type=int, default=80, help="past frames displayed")
    parser.add_argument("--horizon", type=int, default=20, help="future frames displayed")
    parser.add_argument("--interval", type=int, default=100, help="animation interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    if args.directory == Path(__file__).parent and (directory / "shp").is_dir():
        directory = directory / "shp"
    try:
        sequence = ShapeSequence(directory, args.pattern)
        viewer = TwoFrameComparisonViewer(
            sequence, args.entity, max(args.interval, 10),
            max(args.window, 2), max(args.trail, 2), max(args.horizon, PREDICTION_STEPS),
        )
    except (FileNotFoundError, ValueError, shapefile.ShapefileException) as exc:
        raise SystemExit(str(exc)) from exc
    viewer.mainloop()


if __name__ == "__main__":
    main()
