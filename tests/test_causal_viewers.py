from typing import Iterable

import pytest

from viewer_comparison import ComparisonViewer
from viewer_multi_frame import MultiFrameComparisonViewer
from viewer_two_frame import PREDICTION_STEPS, TwoFrameComparisonViewer


Point = tuple[float, float]


class FakeSequence:
    def __init__(self, track: Iterable[Point | None], entity: str = "circulator") -> None:
        self.track = list(track)
        self.entity = entity
        self.files = [object() for _ in self.track]

    def frame(self, index: int) -> tuple[list[dict], int]:
        actual = self.track[index]
        points = [] if actual is None else [
            {"x": actual[0], "y": actual[1], "name": self.entity}
        ]
        return points, index


class RecordingPredictor:
    """Encode how many observations were available into every prediction."""

    def __init__(self) -> None:
        self.observed_frames: list[int] = []

    def predict(self, frame: int) -> Point | None:
        if not self.observed_frames:
            return None
        return float(len(self.observed_frames)), float(frame)

    def update(self, frame: int, actual: Point) -> None:
        assert frame not in self.observed_frames
        self.observed_frames.append(frame)


class FakeStringVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def configure_viewer(viewer: ComparisonViewer, track: list[Point | None]) -> RecordingPredictor:
    predictor = RecordingPredictor()
    viewer.sequence = FakeSequence(track)
    viewer.entity = "circulator"
    viewer.horizon = 4
    viewer.predictors = {"recorder": predictor}
    viewer.track = []
    viewer.predictions = {"recorder": []}
    viewer.errors = {"recorder": []}
    viewer.forecasts = {"recorder": []}
    viewer.sample_counts = []
    viewer.total_samples = 0
    return predictor


def test_model_filter_can_isolate_one_prediction_track() -> None:
    viewer = object.__new__(ComparisonViewer)
    viewer.predictors = {"first": object(), "second": object()}
    viewer.display_model_var = FakeStringVar("second")

    assert viewer.displayed_predictors() == ["second"]

    viewer.display_model_var = FakeStringVar("All models")
    assert viewer.displayed_predictors() == ["first", "second"]


def test_one_frame_processing_scores_before_revealing_observation() -> None:
    viewer = object.__new__(ComparisonViewer)
    predictor = configure_viewer(viewer, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    ComparisonViewer.ensure_processed(viewer, 2)

    assert viewer.predictions["recorder"] == [
        None,
        pytest.approx((1.0, 1.0)),
        pytest.approx((2.0, 2.0)),
    ]
    assert predictor.observed_frames == [0, 1, 2]


def test_missing_observation_does_not_update_predictor() -> None:
    viewer = object.__new__(ComparisonViewer)
    predictor = configure_viewer(viewer, [(0.0, 0.0), None, (2.0, 0.0)])

    ComparisonViewer.ensure_processed(viewer, 2)

    assert predictor.observed_frames == [0, 2]
    assert viewer.errors["recorder"][1] is None
    assert viewer.sample_counts == [1, 1, 2]


def test_two_frame_prediction_is_frozen_two_frames_early() -> None:
    viewer = object.__new__(TwoFrameComparisonViewer)
    configure_viewer(viewer, [(float(frame), 0.0) for frame in range(4)])
    viewer.scheduled_predictions = {"recorder": {}}

    TwoFrameComparisonViewer.ensure_processed(viewer, 2)

    assert viewer.predictions["recorder"][0:2] == [None, None]
    # The target-frame 2 prediction was made just after observing frame 0.
    assert viewer.predictions["recorder"][2] == pytest.approx((1.0, 2.0))
    assert PREDICTION_STEPS == 2


def test_multi_frame_prediction_is_frozen_at_configured_horizon() -> None:
    viewer = object.__new__(MultiFrameComparisonViewer)
    configure_viewer(viewer, [(float(frame), 0.0) for frame in range(6)])
    viewer.prediction_steps = 3
    viewer.scheduled_predictions = {"recorder": {}}

    MultiFrameComparisonViewer.ensure_processed(viewer, 3)

    assert viewer.predictions["recorder"][0:3] == [None, None, None]
    # The target-frame 3 prediction was made just after observing frame 0.
    assert viewer.predictions["recorder"][3] == pytest.approx((1.0, 3.0))
