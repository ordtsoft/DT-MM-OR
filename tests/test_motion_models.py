from collections import deque
from pathlib import Path

import pytest

from viewer_prediction import (
    build_causal_predictions,
    fit_motion_model,
    forecast,
    nice_step,
    sequence_number,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("camera01_centroids_2.shp", 2),
        ("camera01_centroids_10.shp", 10),
        ("frame_without_number.shp", 0),
    ],
)
def test_sequence_number_uses_numeric_suffix(filename: str, expected: int) -> None:
    assert sequence_number(Path(filename)) == expected


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        (8.0, 1.0),
        (16.0, 2.0),
        (40.0, 5.0),
        (80.0, 10.0),
        (0.0, 2e-13),
    ],
)
def test_nice_step_returns_readable_grid_intervals(span: float, expected: float) -> None:
    assert nice_step(span) == pytest.approx(expected)


def test_single_observation_produces_stationary_model() -> None:
    model = fit_motion_model(deque([(4, 2.5, -1.0)]))

    assert forecast(model, 20) == pytest.approx((2.5, -1.0))


def test_motion_model_recovers_exact_linear_track() -> None:
    observations = deque(
        [
            (0, 1.0, 5.0),
            (1, 3.0, 4.0),
            (2, 5.0, 3.0),
            (3, 7.0, 2.0),
        ]
    )

    assert forecast(fit_motion_model(observations), 5) == pytest.approx((11.0, 0.0))


def test_causal_predictions_do_not_use_target_observation() -> None:
    normal_track = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    changed_target = [(0.0, 0.0), (10.0, 0.0), (999.0, 999.0)]

    normal_predictions, _, _, _ = build_causal_predictions(normal_track, window=10)
    changed_predictions, _, _, _ = build_causal_predictions(changed_target, window=10)

    assert normal_predictions[0] is None
    assert normal_predictions[1] == pytest.approx((0.0, 0.0))
    assert normal_predictions[2] == pytest.approx((20.0, 0.0))
    assert changed_predictions[2] == pytest.approx(normal_predictions[2])


def test_missing_detection_is_not_scored_or_added_to_sample_count() -> None:
    track = [(0.0, 0.0), None, (2.0, 0.0)]

    predictions, _, errors, sample_counts = build_causal_predictions(track, window=10)

    assert predictions[1] == pytest.approx((0.0, 0.0))
    assert errors[1] is None
    assert sample_counts == [1, 1, 2]
