"""Export a small array comparison; install the report extra for the PDF."""

from pathlib import Path

import numpy as np

from synthetic_imaging_validation import mae, rmse, write_report


def main():
    real = np.zeros((32, 32), dtype=np.float32)
    synthetic = real.copy()
    synthetic[10:20, 10:20] = 0.25
    scores = {"mae": mae(real, synthetic), "rmse": rmse(real, synthetic)}
    output = Path("outputs")
    for suffix in (".pdf", ".tex"):
        path = write_report(scores, output / f"array_comparison{suffix}", title="Array comparison")
        print(path)


if __name__ == "__main__":
    main()
