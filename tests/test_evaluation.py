import numpy as np
import pytest

from synthetic_imaging_validation import evaluate_pairs, parallel_map, resolve_num_workers
from synthetic_imaging_validation.io import load_paired_directories
from synthetic_imaging_validation.metrics.image_similarity import mae, mse


def test_evaluate_pairs_parallel_api_matches_sequential(tmp_path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    for index in range(3):
        real = np.full((2, 2), index, dtype=np.float32)
        synthetic = real + index
        np.save(real_dir / f"case_{index}.npy", real)
        np.save(synthetic_dir / f"case_{index}.npy", synthetic)

    pairs = load_paired_directories(real_dir, synthetic_dir)

    def metric_function(pair):
        return {
            "mae": mae(pair.real.array, pair.synthetic.array),
            "mse": mse(pair.real.array, pair.synthetic.array),
        }

    sequential = evaluate_pairs(pairs, metric_function, num_workers=1)
    parallel = evaluate_pairs(pairs, metric_function, num_workers=2, show_progress=True)
    assert parallel == sequential
    assert [record["key"] for record in parallel] == ["case_0", "case_1", "case_2"]
    assert parallel[0]["metadata"] == {}
    assert parallel[1]["metrics"] == {"mae": 1.0, "mse": 1.0}
    assert parallel[1]["real"].endswith("case_1.npy")
    assert parallel[1]["synthetic"].endswith("case_1.npy")


def test_evaluation_helpers_validate_worker_count_and_metric_output(monkeypatch, tmp_path):
    assert parallel_map([], lambda value: value, num_workers=0, show_progress=True) == []
    assert resolve_num_workers(2, 0) == 1
    monkeypatch.setattr("synthetic_imaging_validation.evaluation.os.cpu_count", lambda: 2)
    assert resolve_num_workers(0, 5) == 2
    with pytest.raises(ValueError, match="num_workers"):
        resolve_num_workers(-1, 2)

    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    np.save(real_dir / "case.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(synthetic_dir / "case.npy", np.zeros((2, 2), dtype=np.float32))
    pairs = load_paired_directories(real_dir, synthetic_dir)

    with pytest.raises(TypeError, match="metric_function"):
        evaluate_pairs(pairs, lambda pair: 1.0)
