"""Only runs where torch exists: the CI `ml` job installs it, plain runs skip."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is optional; the ml job installs it")

from omega.vision.model import load_predictor
from omega.vision.torch_model import TinyCNN, TorchPredictor
from omega.vision.train import make_dataset, normalise, train


def test_forward_pass_shape():
    model = TinyCNN(4)
    out = model(torch.zeros(2, 1, 28, 28))
    assert tuple(out.shape) == (2, 4)


def test_checkpoint_round_trip_and_probabilities(tmp_path):
    model = TinyCNN(4)
    path = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "meta": {"classes": ["a", "b", "c", "d"], "version": "test-1"}}, path)
    predictor = TorchPredictor(str(path))
    probs = predictor.predict(np.zeros((3, 1, 28, 28), dtype="float32"))
    assert probs.shape == (3, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert predictor.version == "test-1" and predictor.source == "torch"


def test_dataset_is_reproducible():
    a, _ = make_dataset(8, seed=3)
    b, _ = make_dataset(8, seed=3)
    assert np.array_equal(a, b)


def test_training_learns_something_on_synthetic_shapes(tmp_path):
    path = tmp_path / "tiny.pt"
    meta = train(str(path), epochs=4, samples=600, seed=0)
    assert path.exists()
    assert meta["val_accuracy"] >= 0.5, "a tiny CNN on four synthetic shapes should beat chance clearly"
    predictor, reason = load_predictor(str(path))
    assert predictor.source == "torch", reason
