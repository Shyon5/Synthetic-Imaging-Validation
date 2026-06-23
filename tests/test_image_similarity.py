import numpy as np
import pytest

from synthetic_imaging_validation.metrics.image_similarity import mae, mse, ms_ssim, nrmse, psnr, rmse, ssim


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


@pytest.mark.torch
def test_numpy_and_torch_inputs_match():
    torch = pytest.importorskip("torch")
    real = np.arange(16, dtype=np.float32).reshape(4, 4)
    synthetic = real + 1
    assert mae(torch.from_numpy(real), torch.from_numpy(synthetic)) == mae(real, synthetic)


@pytest.mark.torch
def test_optional_ms_ssim_identical_input():
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    assert ms_ssim(image, image, data_range=1.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.torch
def test_ms_ssim_axes_and_shape_validation():
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
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
