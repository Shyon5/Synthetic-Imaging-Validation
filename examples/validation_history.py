"""Record scores for five controlled intensity offsets and save headless plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synthetic_imaging_validation import (
    append_history, intensity_distribution_score, plot_history, plot_results,
    similarity_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/history_example"))
    args = parser.parse_args()
    grid = np.linspace(-1, 1, 48)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    real = (0.1 + 0.6 * np.exp(-4 * (x * x + y * y))).astype(np.float32)
    history_path = args.output_dir / "history.json"
    for step, offset in enumerate((0.20, 0.14, 0.09, 0.05, 0.02), start=1):
        synthetic = real + offset
        similarity = similarity_score(real, synthetic, value_range=(0, 1), return_details=True)
        distribution = intensity_distribution_score(
            real, synthetic, value_range=(0, 1), return_details=True,
        )
        results = {
            "score_protocol": {
                "similarity_score": similarity.pop("protocol"),
                "intensity_distribution_score": distribution.pop("protocol"),
            },
            "similarity_score": similarity,
            "intensity_distribution_score": distribution,
        }
        append_history(history_path, results, step=step, run="controlled_offset")
        print(f"Step {step}: similarity={similarity['value']:.2f}, "
              f"intensity distribution={distribution['value']:.2f}")
    fields = ["similarity_score", "intensity_distribution_score"]
    plot_history(history_path, metrics=fields, output=args.output_dir / "scores.png")
    plot_results(results, metrics=fields, output=args.output_dir / "last_step.svg")
    print(f"Saved history and plots in {args.output_dir}")


if __name__ == "__main__":
    main()
