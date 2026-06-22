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


def test_numpy_and_torch_inputs_match():
    torch = pytest.importorskip("torch")
    real = np.arange(16, dtype=np.float32).reshape(4, 4)
    synthetic = real + 1
    assert mae(torch.from_numpy(real), torch.from_numpy(synthetic)) == mae(real, synthetic)


def test_optional_ms_ssim_identical_input():
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    image = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    assert ms_ssim(image, image, data_range=1.0) == pytest.approx(1.0, abs=1e-6)
