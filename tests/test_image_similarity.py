import builtins

import numpy as np
import pytest

import synthetic_imaging_validation.metrics.image_similarity as image_similarity_module
from synthetic_imaging_validation.metrics.image_similarity import (
    _validate_ms_ssim_candidate,
    mae,
    mse,
    ms_ssim,
    nrmse,
    psnr,
    rmse,
    ssim,
)


def test_identical_inputs_have_optimal_scores():
    image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    assert mae(image, image) == 0.0
    assert mse(image, image) == 0.0
    assert rmse(image, image) == 0.0
    assert nrmse(image, image) == 0.0
    assert np.isinf(psnr(image, image, data_range=1.0))
    assert ssim(image, image, data_range=1.0) == pytest.approx(1.0)


def test_different_inputs_have_plausible_scores():
    real = np.zeros((32, 32), dtype=np.float32)
    synthetic = np.ones_like(real) * 0.25
    assert mae(real, synthetic) == pytest.approx(0.25)
    assert mse(real, synthetic) == pytest.approx(0.0625)
    assert rmse(real, synthetic) == pytest.approx(0.25)
    assert psnr(real, synthetic, data_range=1.0) == pytest.approx(12.0411998)
    assert ssim(real, synthetic, data_range=1.0) < 1.0


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae(np.zeros((4, 4)), np.zeros((5, 4)))
    bad = np.zeros((4, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        mse(bad, np.zeros_like(bad))


def test_nrmse_normalizations_and_zero_denominators():
    real = np.array([1.0, 2.0, 3.0])
    synthetic = real + 1.0
    assert nrmse(real, synthetic, normalization="range") == pytest.approx(0.5)
    assert nrmse(real, synthetic, normalization="mean") == pytest.approx(0.5)
    assert nrmse(real, synthetic, normalization="l2") == pytest.approx(1 / np.sqrt(14 / 3))
    assert nrmse(np.zeros(3), np.zeros(3)) == 0.0
    assert np.isinf(nrmse(np.zeros(3), np.ones(3)))
    with pytest.raises(ValueError, match="normalization must be"):
        nrmse(real, synthetic, normalization="invalid")


def test_psnr_infers_combined_data_range():
    assert psnr(np.array([0.0, 1.0]), np.array([0.0, 2.0])) == pytest.approx(9.03089987)
    with pytest.raises(ValueError, match="strictly positive"):
        psnr(np.zeros(4), np.ones(4), data_range=-1)


def test_ssim_window_and_axis_validation():
    image = np.arange(6 * 6, dtype=np.float32).reshape(6, 6)
    assert ssim(image, image, data_range=35.0) == pytest.approx(1.0)
    assert ssim(image, image, data_range=35.0, win_size=3, gaussian_weights=False) == pytest.approx(1.0)
    for invalid in (2, 4, 7):
        with pytest.raises(ValueError, match="win_size"):
            ssim(image, image, data_range=35.0, win_size=invalid)
    with pytest.raises(ValueError, match="must be different"):
        ssim(np.zeros((2, 8, 8)), np.zeros((2, 8, 8)), batch_axis=0, channel_axis=0)
    with pytest.raises(ValueError, match="SSIM expects"):
        ssim(np.zeros((2, 3, 8, 8, 1)), np.zeros((2, 3, 8, 8, 1)), data_range=1.0)
    batch_channels = np.zeros((2, 8, 8, 3), dtype=np.float32)
    assert ssim(
        batch_channels,
        batch_channels,
        data_range=1.0,
        batch_axis=0,
        channel_axis=3,
    ) == pytest.approx(1.0)


@pytest.mark.torch
def test_numpy_and_torch_inputs_match():
    torch = pytest.importorskip("torch")
    real = np.arange(16, dtype=np.float32).reshape(4, 4)
    synthetic = real + 1
    assert mae(torch.from_numpy(real), torch.from_numpy(synthetic)) == mae(real, synthetic)


def test_tensor_like_inputs_are_converted_without_a_torch_dependency():
    class TensorLike:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    real = np.arange(16, dtype=np.float32).reshape(4, 4)
    synthetic = real + 1
    assert mae(TensorLike(real), TensorLike(synthetic)) == mae(real, synthetic)


def test_native_ms_ssim_identical_input():
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    assert ms_ssim(image, image, data_range=1.0) == pytest.approx(1.0, abs=1e-6)


def test_native_ms_ssim_regression_values_for_2d_and_3d():
    real_2d = np.zeros((64, 64), dtype=np.float32)
    synthetic_2d = real_2d.copy()
    synthetic_2d[20:44, 20:44] = 0.25
    assert ms_ssim(real_2d, synthetic_2d, data_range=1.0) == pytest.approx(
        0.93259758,
        abs=1e-5,
    )

    real_3d = np.zeros((24, 28, 32), dtype=np.float32)
    synthetic_3d = real_3d.copy()
    synthetic_3d[6:18, 7:21, 8:24] = 0.25
    assert ms_ssim(real_3d, synthetic_3d, data_range=1.0, max_scales=2) == pytest.approx(
        0.32841346,
        abs=1e-5,
    )


def test_ms_ssim_axes_and_shape_validation():
    images = np.linspace(0, 1, 2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    assert ms_ssim(images, images, batch_axis=0, data_range=1.0, max_scales=2) == pytest.approx(
        1.0, abs=1e-6
    )
    multichannel = np.stack([images[0], images[0]], axis=-1)
    assert ms_ssim(multichannel, multichannel, channel_axis=-1, data_range=1.0, max_scales=2) == pytest.approx(
        1.0, abs=1e-6
    )
    with pytest.raises(ValueError, match="must be different"):
        ms_ssim(images, images, batch_axis=0, channel_axis=0, data_range=1.0)
    with pytest.raises(ValueError, match="expects 2D or 3D"):
        ms_ssim(np.zeros(8), np.zeros(8), data_range=1.0)
    batch_channels = np.zeros((2, 2, 64, 64), dtype=np.float32)
    assert ms_ssim(
        batch_channels,
        batch_channels,
        batch_axis=0,
        channel_axis=1,
        data_range=1.0,
        max_scales=1,
    ) == pytest.approx(1.0, abs=1e-6)


def test_ms_ssim_backend_selection_and_exhausted_attempts(monkeypatch):
    def always_fail(*args, **kwargs):
        raise ValueError("unsupported test configuration")

    monkeypatch.setattr(
        image_similarity_module,
        "_numpy_ms_ssim_candidate",
        always_fail,
    )
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="could not be computed.*unsupported test configuration"):
        ms_ssim(image, image, data_range=1.0, max_scales=2)

    monkeypatch.setattr(
        image_similarity_module,
        "_torchmetrics_ms_ssim_candidate",
        lambda *args, **kwargs: 0.75,
    )
    assert ms_ssim(image, image, data_range=1.0, max_scales=1, backend="torchmetrics") == 0.75
    with pytest.raises(ValueError, match="backend must be"):
        ms_ssim(image, image, data_range=1.0, backend="unknown")


def test_torchmetrics_backend_has_a_clear_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    image = np.zeros((16, 16), dtype=np.float32)
    assert ms_ssim(image, image, data_range=1.0, max_scales=1) == pytest.approx(1.0)
    with pytest.raises(ImportError, match="install the 'torch' extra"):
        ms_ssim(image, image, data_range=1.0, max_scales=1, backend="torchmetrics")


def test_ms_ssim_candidate_size_checks_are_explicit():
    with pytest.raises(ValueError, match="height and width"):
        _validate_ms_ssim_candidate((1, 1, 3, 8), scales=2, kernel_size=3)
    with pytest.raises(ValueError, match="height and width"):
        _validate_ms_ssim_candidate((1, 1, 8, 3), scales=2, kernel_size=3)
    with pytest.raises(ValueError, match="height is too small"):
        _validate_ms_ssim_candidate((1, 1, 4, 8), scales=2, kernel_size=5)
    with pytest.raises(ValueError, match="width is too small"):
        _validate_ms_ssim_candidate((1, 1, 8, 4), scales=2, kernel_size=5)
    with pytest.raises(ValueError, match="reflection padding"):
        _validate_ms_ssim_candidate((1, 1, 2, 16, 16), scales=1, kernel_size=5)


@pytest.mark.torch
@pytest.mark.parametrize(
    ("shape", "batch_axis", "channel_axis", "max_scales"),
    [
        ((72, 68), None, None, 3),
        ((2, 64, 60), 0, None, 2),
        ((64, 60, 2), None, -1, 2),
        ((24, 28, 32), None, None, 2),
        ((2, 2, 24, 28, 32), 0, 1, 2),
    ],
)
def test_numpy_ms_ssim_matches_torchmetrics(
    shape,
    batch_axis,
    channel_axis,
    max_scales,
):
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    generator = np.random.default_rng(42)
    real = generator.random(shape, dtype=np.float32)
    synthetic = np.clip(
        real + generator.normal(0.0, 0.08, size=shape).astype(np.float32),
        0.0,
        1.0,
    )
    arguments = {
        "data_range": 1.0,
        "batch_axis": batch_axis,
        "channel_axis": channel_axis,
        "max_scales": max_scales,
    }
    native = ms_ssim(real, synthetic, backend="numpy", **arguments)
    reference = ms_ssim(real, synthetic, backend="torchmetrics", **arguments)
    assert native == pytest.approx(reference, rel=1e-5, abs=1e-5)


@pytest.mark.torch
def test_torchmetrics_ms_ssim_reports_exhausted_adaptive_attempts(monkeypatch):
    pytest.importorskip("torch")
    image_module = pytest.importorskip("torchmetrics.functional.image")

    def always_fail(*args, **kwargs):
        raise ValueError("unsupported reference configuration")

    monkeypatch.setattr(
        image_module,
        "multiscale_structural_similarity_index_measure",
        always_fail,
    )
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="could not be computed.*unsupported reference configuration"):
        ms_ssim(image, image, data_range=1.0, max_scales=2, backend="torchmetrics")
