"""Load arrays, tensors, NumPy files, and NIfTI images into one representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from ..utils.checks import to_numpy, validate_pair, validate_spacing

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ImageData:
    """An image array and optional spatial metadata.

    ``spacing`` follows the array-axis order. For NIfTI files this is obtained
    from the header zooms; no automatic reorientation or resampling is applied.
    """

    array: np.ndarray
    spacing: Optional[tuple[float, ...]] = None
    affine: Optional[np.ndarray] = None
    path: Optional[Path] = None


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _load_path(path: Path, *, npz_key: Optional[str]) -> ImageData:
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if _is_nifti(path):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - dependency is normally installed
            raise ImportError("NIfTI loading requires nibabel.") from exc
        image = nib.load(str(path))
        array = np.asarray(image.get_fdata(dtype=np.float32))
        spacing = tuple(float(v) for v in image.header.get_zooms()[: array.ndim])
        return ImageData(
            array=to_numpy(array, name="image"),
            spacing=validate_spacing(spacing, array.ndim),
            affine=np.asarray(image.affine, dtype=np.float64),
            path=path,
        )

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return ImageData(array=to_numpy(np.load(path, allow_pickle=False), name="image"), path=path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            key = npz_key
            if key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ input contains {len(keys)} arrays; pass npz_key. Available keys: {keys}"
                    )
                key = keys[0]
            if key not in archive:
                raise KeyError(f"NPZ key '{key}' was not found. Available keys: {keys}")
            array = np.asarray(archive[key])
        return ImageData(array=to_numpy(array, name="image"), path=path)

    raise ValueError(f"Unsupported input format for '{path}'. Use .nii, .nii.gz, .npy, or .npz.")


def load_image(source: Any, *, spacing: Optional[Iterable[float]] = None, npz_key: Optional[str] = None) -> ImageData:
    """Load one image from a path, NumPy array, PyTorch tensor, or ``ImageData``.

    Non-finite and complex values are rejected. Explicit ``spacing`` overrides
    missing metadata but cannot override spacing already attached to ImageData.
    """

    if isinstance(source, ImageData):
        data = source
        to_numpy(data.array, name="image")
    elif isinstance(source, (str, Path)):
        data = _load_path(Path(source), npz_key=npz_key)
    else:
        data = ImageData(array=to_numpy(source, name="image"))

    if spacing is None:
        return data
    if data.spacing is not None:
        raise ValueError("Spacing is already present in ImageData; do not provide it twice.")
    checked = validate_spacing(spacing, data.array.ndim)
    return ImageData(array=data.array, spacing=checked, affine=data.affine, path=data.path)


def load_pair(
    real: Any,
    synthetic: Any,
    *,
    require_same_shape: bool = True,
    require_spatial_match: bool = True,
    atol: float = 1e-5,
) -> tuple[ImageData, ImageData]:
    """Load a real/synthetic pair and optionally validate shape and NIfTI geometry."""

    real_data = load_image(real)
    synthetic_data = load_image(synthetic)
    if require_same_shape:
        validate_pair(real_data.array, synthetic_data.array)

    if require_spatial_match:
        if real_data.spacing is not None and synthetic_data.spacing is not None:
            if len(real_data.spacing) != len(synthetic_data.spacing) or not np.allclose(
                real_data.spacing, synthetic_data.spacing, rtol=0.0, atol=atol
            ):
                raise ValueError(
                    f"Spacing mismatch: real {real_data.spacing}, synthetic {synthetic_data.spacing}. "
                    "Resample before spatial comparison."
                )
        if real_data.affine is not None and synthetic_data.affine is not None:
            if not np.allclose(real_data.affine, synthetic_data.affine, rtol=0.0, atol=atol):
                raise ValueError("NIfTI affine mismatch. Resample images onto a common grid first.")
    return real_data, synthetic_data


def load_directory(directory: PathLike, *, recursive: bool = False) -> list[ImageData]:
    """Load a sorted directory of supported image files.

    The function does not pair files or infer subject identifiers. Nested
    directories are included only when ``recursive=True``.
    """

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = [p for p in iterator if p.is_file() and (_is_nifti(p) or p.suffix.lower() in {".npy", ".npz"})]
    return [load_image(path) for path in sorted(paths, key=lambda p: str(p).lower())]

