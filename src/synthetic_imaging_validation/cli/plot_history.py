"""Plot an existing history without loading images or recalculating metrics."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from ..plotting import plot_history


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for previously saved validation histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True, help="Destination .png, .svg or .pdf.")
    parser.add_argument("--metrics", nargs="+")
    args = parser.parse_args(argv)
    try:
        plot_history(args.history, metrics=args.metrics, output=args.output)
    except (OSError, ImportError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
