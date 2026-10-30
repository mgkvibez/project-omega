import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from omega.vision.preprocess import normalize, resize, to_batch, to_tensor


def blank(side=28):
    return [[0] * side for _ in range(side)]


def test_flat_square_image_is_reshaped():
    assert to_tensor([0] * 784).shape == (1, 28, 28)


def test_2d_and_3d_both_accepted():
    assert to_tensor(blank()).shape == (1, 28, 28)
    assert to_tensor([blank()]).shape == (1, 28, 28)


def test_non_square_flat_image_rejected():
    with pytest.raises(ValueError) as exc:
        to_tensor([0] * 99)          # 100 would be a legitimate 10x10 image
    assert "not a square" in str(exc.value)


def test_nan_rejected():
    bad = np.zeros((28, 28)); bad[3, 3] = np.nan
    with pytest.raises(ValueError) as exc:
        to_tensor(bad.tolist())
    assert "NaN" in str(exc.value)


def test_colour_image_is_collapsed_to_luminance():
    rgb = np.zeros((3, 28, 28)); rgb[0] = 255
    out = to_tensor(rgb.tolist(), channels=1)
    assert out.shape == (1, 28, 28)


def test_wrong_channel_count_rejected():
    with pytest.raises(ValueError):
        to_tensor(np.zeros((2, 28, 28)).tolist(), channels=1)


def test_resize_scales_to_requested_side():
    img = np.zeros((1, 10, 10)); img[0, 2:6, 2:6] = 1
    assert resize(img, 28).shape == (1, 28, 28)
    assert resize(img, 10).shape == (1, 10, 10)


def test_normalize_maps_byte_range_to_minus_one_one():
    out = normalize(np.array([[[0.0, 255.0]]]))
    assert out.min() == pytest.approx(-1.0) and out.max() == pytest.approx(1.0)


def test_normalize_rejects_zero_std():
    with pytest.raises(ValueError):
        normalize(np.zeros((1, 2, 2)), std=0)


def test_batch_shape_and_order():
    batch = to_batch([blank(), blank()])
    assert batch.shape == (2, 1, 28, 28)


def test_empty_batch_rejected():
    with pytest.raises(ValueError) as exc:
        to_batch([])
    assert "empty" in str(exc.value)


def test_oversized_batch_rejected():
    with pytest.raises(ValueError) as exc:
        to_batch([blank()] * 5, max_batch=4)
    assert "exceeds" in str(exc.value)


def test_mixed_input_sizes_are_resized_into_one_batch():
    batch = to_batch([blank(28), blank(14)])
    assert batch.shape == (2, 1, 28, 28)


def test_preprocessing_is_deterministic():
    img = np.random.default_rng(1).random((28, 28)).tolist()
    assert np.array_equal(to_tensor(img), to_tensor(img))
