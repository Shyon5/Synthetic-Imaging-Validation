"""Pair real and synthetic image files without assuming a dataset layout."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .loading import ImageData, PathLike, is_supported_image_path, load_pair

KNOWN_IMAGE_SUFFIXES = (".nii.gz", ".nii", ".npy", ".npz")
DIRECTORY_PAIRING_MODES = ("stem", "sorted")


@dataclass(frozen=True)
class ImagePair:
    """A loaded real/synthetic pair with a stable user-facing key.

    Parameters
    ----------
    key:
        Identifier used in reports. For directory pairing this is usually the
        filename without the image suffix. For manifest pairing it can come from
        a dedicated CSV column.
    real:
        Loaded real image and optional metadata.
    synthetic:
        Loaded synthetic image and optional metadata.
    metadata:
        Optional string metadata copied from manifest columns that are not used
        as file paths.
    """

    key: str
    real: ImageData
    synthetic: ImageData
    metadata: Optional[dict[str, str]] = None


def image_file_key(path: PathLike) -> str:
    """Return a filename key with common medical-image suffixes removed."""

    name = Path(path).name
    lower = name.lower()
    for suffix in KNOWN_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _supported_paths(directory: PathLike, *, recursive: bool) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted((p for p in iterator if is_supported_image_path(p)), key=lambda p: str(p).lower())


def _index_by_key(paths: list[Path], *, role: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        key = image_file_key(path)
        if key in indexed:
            raise ValueError(
                f"Duplicate {role} key '{key}'. Directory pairing by stem requires unique filenames "
                "after removing .nii/.nii.gz/.npy/.npz."
            )
        indexed[key] = path
    return indexed


def _format_missing(keys: list[str]) -> str:
    preview = ", ".join(keys[:5])
    if len(keys) > 5:
        preview += f", ... ({len(keys)} total)"
    return preview


def pair_directory_files(
    real_dir: PathLike,
    synthetic_dir: PathLike,
    *,
    pairing: str = "stem",
    recursive: bool = False,
) -> list[tuple[str, Path, Path]]:
    """Pair supported files from two directories.

    Parameters
    ----------
    real_dir, synthetic_dir:
        Directories containing real and synthetic files.
    pairing:
        ``"stem"`` pairs files by the filename without image suffix. This is
        the safest convention when both directories contain the same case IDs.
        ``"sorted"`` pairs files by stable alphabetical order and is intended
        only for carefully controlled folders.
    recursive:
        Include files from nested directories when ``True``.

    Returns
    -------
    list of tuple
        Tuples of ``(key, real_path, synthetic_path)``. The files are not loaded.
    """

    if pairing not in DIRECTORY_PAIRING_MODES:
        raise ValueError(f"pairing must be one of {DIRECTORY_PAIRING_MODES}, got '{pairing}'.")

    real_paths = _supported_paths(real_dir, recursive=recursive)
    synthetic_paths = _supported_paths(synthetic_dir, recursive=recursive)
    if not real_paths:
        raise ValueError(f"No supported real files found in: {real_dir}")
    if not synthetic_paths:
        raise ValueError(f"No supported synthetic files found in: {synthetic_dir}")

    if pairing == "sorted":
        if len(real_paths) != len(synthetic_paths):
            raise ValueError(
                "Sorted directory pairing requires the same number of real and synthetic files. "
                f"Got {len(real_paths)} real and {len(synthetic_paths)} synthetic files."
            )
        return [
            (image_file_key(real_path), real_path, synthetic_path)
            for real_path, synthetic_path in zip(real_paths, synthetic_paths)
        ]

    real_index = _index_by_key(real_paths, role="real")
    synthetic_index = _index_by_key(synthetic_paths, role="synthetic")
    missing_synthetic = sorted(set(real_index) - set(synthetic_index))
    missing_real = sorted(set(synthetic_index) - set(real_index))
    if missing_synthetic or missing_real:
        parts = []
        if missing_synthetic:
            parts.append(f"missing synthetic for: {_format_missing(missing_synthetic)}")
        if missing_real:
            parts.append(f"missing real for: {_format_missing(missing_real)}")
        raise ValueError("Directory pairing mismatch; " + "; ".join(parts) + ".")

    return [(key, real_index[key], synthetic_index[key]) for key in sorted(real_index)]


def _load_path_pair(
    key: str,
    real_path: Path,
    synthetic_path: Path,
    *,
    metadata: Optional[dict[str, str]] = None,
    require_same_shape: bool = True,
    require_spatial_match: bool = True,
    atol: float = 1e-5,
) -> ImagePair:
    real, synthetic = load_pair(
        real_path,
        synthetic_path,
        require_same_shape=require_same_shape,
        require_spatial_match=require_spatial_match,
        atol=atol,
    )
    return ImagePair(key=key, real=real, synthetic=synthetic, metadata=metadata)


def load_paired_directories(
    real_dir: PathLike,
    synthetic_dir: PathLike,
    *,
    pairing: str = "stem",
    recursive: bool = False,
    require_same_shape: bool = True,
    require_spatial_match: bool = True,
    atol: float = 1e-5,
) -> list[ImagePair]:
    """Load real/synthetic pairs from two directories.

    The function deliberately supports only explicit pairing rules. It does not
    infer subject IDs from arbitrary naming patterns or metadata.
    """

    file_pairs = pair_directory_files(real_dir, synthetic_dir, pairing=pairing, recursive=recursive)
    return [
        _load_path_pair(
            key,
            real_path,
            synthetic_path,
            require_same_shape=require_same_shape,
            require_spatial_match=require_spatial_match,
            atol=atol,
        )
        for key, real_path, synthetic_path in file_pairs
    ]


def _resolve_manifest_path(value: str, *, base_dir: Path) -> Path:
    if not value.strip():
        raise ValueError("Manifest paths cannot be empty.")
    path = Path(value.strip())
    return path if path.is_absolute() else base_dir / path


def load_manifest_pairs(
    manifest_path: PathLike,
    *,
    real_column: str = "real",
    synthetic_column: str = "synthetic",
    key_column: Optional[str] = None,
    base_dir: Optional[PathLike] = None,
    require_same_shape: bool = True,
    require_spatial_match: bool = True,
    atol: float = 1e-5,
) -> list[ImagePair]:
    """Load real/synthetic pairs described by a CSV manifest.

    Relative paths are resolved against ``base_dir`` when provided, otherwise
    against the directory containing the manifest. A minimal manifest has two
    columns named ``real`` and ``synthetic`` by default.
    """

    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    root = Path(base_dir) if base_dir is not None else manifest.parent
    pairs: list[ImagePair] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest must contain a header row.")
        required_columns = [real_column, synthetic_column]
        if key_column is not None:
            required_columns.append(key_column)
        missing_columns = [column for column in required_columns if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {missing_columns}")

        for row_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                real_path = _resolve_manifest_path(row.get(real_column, ""), base_dir=root)
                synthetic_path = _resolve_manifest_path(row.get(synthetic_column, ""), base_dir=root)
            except ValueError as exc:
                raise ValueError(f"Manifest row {row_number}: {exc}") from exc
            if key_column is None:
                key = image_file_key(real_path)
            else:
                key = (row.get(key_column) or "").strip()
                if not key:
                    raise ValueError(f"Manifest row {row_number} has an empty key column '{key_column}'.")
            metadata = {
                str(column): str(value)
                for column, value in row.items()
                if column is not None
                and column not in {real_column, synthetic_column}
                and value is not None
                and str(value).strip()
            }
            try:
                pairs.append(
                    _load_path_pair(
                        key,
                        real_path,
                        synthetic_path,
                        metadata=metadata or None,
                        require_same_shape=require_same_shape,
                        require_spatial_match=require_spatial_match,
                        atol=atol,
                    )
                )
            except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
                raise type(exc)(f"Manifest row {row_number}: {exc}") from exc

    if not pairs:
        raise ValueError(f"Manifest contains no data rows: {manifest}")
    return pairs
