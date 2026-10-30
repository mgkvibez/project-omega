import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from omega.vision.model import StubPredictor, load_predictor, torch_available


def batch(n=2):
    return np.random.default_rng(0).normal(0, 1, (n, 1, 28, 28)).astype("float32")


def test_stub_returns_one_distribution_per_image():
    out = StubPredictor().predict(batch(3))
    assert out.shape == (3, 4)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_stub_is_deterministic():
    p = StubPredictor()
    assert np.array_equal(p.predict(batch()), p.predict(batch()))


def test_stub_rejects_wrong_rank():
    with pytest.raises(ValueError):
        StubPredictor().predict(np.zeros((2, 28, 28)))


def test_stub_is_named_as_a_stub_so_nobody_mistakes_it_for_a_model():
    assert StubPredictor().source == "stub"


def test_no_checkpoint_falls_back_with_a_reason():
    predictor, reason = load_predictor("")
    assert predictor.source == "stub" and "no checkpoint" in reason


def test_missing_checkpoint_says_so():
    predictor, reason = load_predictor("/tmp/definitely-not-here.pt")
    assert predictor.source == "stub" and "not found" in reason


def test_torch_availability_is_a_bool():
    assert isinstance(torch_available(), bool)
