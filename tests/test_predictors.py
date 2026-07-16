import math

import pytest

from viewer_comparison import (
    AdaptiveEnsemblePredictor,
    KNearestMotionPredictor,
    OnlineNeuralPredictor,
    PersistencePredictor,
    VelocityPredictor,
    blend_with_white,
)


def test_secondary_prediction_color_is_lighter() -> None:
    assert blend_with_white("#000000", amount=0.5) == "#808080"
    assert blend_with_white("#ffffff") == "#ffffff"


def test_persistence_has_a_cold_start_then_holds_last_position() -> None:
    predictor = PersistencePredictor()

    assert predictor.predict(0) is None
    predictor.update(0, (3.0, 4.0))

    assert predictor.predict(25) == (3.0, 4.0)


def test_constant_velocity_extrapolates_after_two_observations() -> None:
    predictor = VelocityPredictor(smoothing=1.0)
    predictor.update(0, (1.0, 2.0))
    predictor.update(2, (5.0, 0.0))

    assert predictor.predict(5) == pytest.approx((11.0, -3.0))


def test_knn_waits_for_its_configured_warmup() -> None:
    predictor = KNearestMotionPredictor(neighbors=1, minimum_samples=2)
    predictor.update(0, (0.0, 0.0))
    predictor.update(1, (1.0, 0.0))

    assert predictor.predict(2) is None

    predictor.update(2, (2.0, 0.0))

    assert predictor.training_samples == 2
    assert predictor.predict(3) == pytest.approx((3.0, 0.0))


def test_online_neural_predictor_is_reproducible() -> None:
    left = OnlineNeuralPredictor()
    right = OnlineNeuralPredictor()
    observations = [(0, (0.0, 0.0)), (1, (0.1, 0.2)), (2, (0.3, 0.5))]

    for frame, point in observations:
        left.update(frame, point)
        right.update(frame, point)

    assert left.predict(3) == pytest.approx(right.predict(3))
    assert left.training_samples == right.training_samples == 2


def test_adaptive_ensemble_keeps_normalized_finite_weights() -> None:
    predictor = AdaptiveEnsemblePredictor()
    predictor.update(0, (0.0, 0.0))
    predictor.update(1, (1.0, 0.0))
    predictor.update(2, (2.0, 0.0))

    assert sum(predictor.weights) == pytest.approx(1.0)
    assert all(math.isfinite(weight) and weight > 0 for weight in predictor.weights)
    assert predictor.predict(3) is not None
