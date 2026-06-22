"""Overlap, surface-distance, volume, and connected-component mask metrics."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from scipy import ndimage

from ..utils.checks import to_numpy, validate_pair, validate_spacing


def _binary(values: Any, threshold: float, name: str) -> np.ndarray:
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    array = to_numpy(values, name=name)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D or 3D scalar mask, got shape {array.shape}.")
    return array >= float(threshold)


def _binary_pair(predicted: Any, reference: Any, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    predicted_array, reference_array = validate_pair(predicted, reference)
    return _binary(predicted_array, threshold, "predicted"), _binary(reference_array, threshold, "reference")


def dice(predicted: Any, reference: Any, *, threshold: float = 0.5) -> float:
    """Return Dice overlap in [0, 1]; two empty masks score 1."""

    pred, ref = _binary_pair(predicted, reference, threshold)
    denominator = int(pred.sum()) + int(ref.sum())
    return 1.0 if denominator == 0 else float(2 * np.count_nonzero(pred & ref) / denominator)


def iou(predicted: Any, reference: Any, *, threshold: float = 0.5) -> float:
    """Return intersection over union (Jaccard index) in [0, 1]; two empty masks score 1."""

    pred, ref = _binary_pair(predicted, reference, threshold)
    union = int(np.count_nonzero(pred | ref))
    return 1.0 if union == 0 else float(np.count_nonzero(pred & ref) / union)


def foreground_fraction(mask: Any, *, threshold: float = 0.5) -> float:
    """Return foreground pixels/voxels divided by all elements, in [0, 1].

    The input may be a scalar 2D mask ``[H, W]`` or 3D mask ``[D, H, W]``.
    """

    binary = _binary(mask, threshold, "mask")
    return float(np.mean(binary))


def active_voxel_fraction(mask: Any, *, threshold: float = 0.5) -> float:
    """Backward-compatible alias for :func:`foreground_fraction`.

    In 2D this is a foreground pixel fraction despite the historical name.
    """

    return foreground_fraction(mask, threshold=threshold)


def foreground_measure_ratio(predicted: Any, reference: Any, *, threshold: float = 0.5) -> float:
    """Return predicted/reference foreground area (2D) or volume (3D) ratio.

    Equal spacing is assumed and therefore cancels. Two empty masks return 1;
    a non-empty prediction against an empty reference returns infinity.
    """

    pred, ref = _binary_pair(predicted, reference, threshold)
    pred_count, ref_count = int(pred.sum()), int(ref.sum())
    if ref_count == 0:
        return 1.0 if pred_count == 0 else float("inf")
    return float(pred_count / ref_count)


def volume_ratio(predicted: Any, reference: Any, *, threshold: float = 0.5) -> float:
    """Backward-compatible alias for :func:`foreground_measure_ratio`.

    The measured quantity is area for 2D masks and volume for 3D masks.
    """

    return foreground_measure_ratio(predicted, reference, threshold=threshold)


def _structure(ndim: int, connectivity: Optional[int]) -> np.ndarray:
    if connectivity is None:
        rank = ndim
    else:
        value = int(connectivity)
        if ndim == 2:
            if value not in (1, 2, 4, 8):
                raise ValueError("2D connectivity must be 1/4 or 2/8.")
            rank = 1 if value in (1, 4) else 2
        else:
            if value not in (1, 2, 3, 6, 18, 26):
                raise ValueError("3D connectivity must be 1/6, 2/18, or 3/26.")
            rank = 1 if value in (1, 6) else 2 if value in (2, 18) else 3
    return ndimage.generate_binary_structure(ndim, rank)


def connected_component_statistics(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> dict[str, Any]:
    """Return connected-component counts and size/volume distribution.

    Component measures are areas for 2D masks and volumes for 3D masks. They
    use physical spacing units when spacing is supplied and pixel/voxel units
    otherwise. Maximal connectivity is used by default.
    """

    binary = _binary(mask, threshold, "mask")
    checked_spacing = validate_spacing(spacing, binary.ndim)
    labels, count = ndimage.label(binary, structure=_structure(binary.ndim, connectivity))
    sizes = np.bincount(labels.ravel())[1:].astype(np.int64) if count else np.empty(0, dtype=np.int64)
    element_measure = float(np.prod(checked_spacing))
    measures = sizes.astype(np.float64) * element_measure
    dimension_name = "area" if binary.ndim == 2 else "volume"
    element_name = "pixels" if binary.ndim == 2 else "voxels"
    result = {
        "spatial_dims": int(binary.ndim),
        "component_count": int(count),
        "foreground_elements": int(binary.sum()),
        "foreground_fraction": float(np.mean(binary)),
        "element_name": element_name,
        "component_elements": sizes.tolist(),
        "component_measures": measures.tolist(),
        "component_elements_mean": float(sizes.mean()) if sizes.size else 0.0,
        "component_elements_median": float(np.median(sizes)) if sizes.size else 0.0,
        "component_elements_std": float(sizes.std()) if sizes.size else 0.0,
        "largest_component_elements": int(sizes.max()) if sizes.size else 0,
        "measure_name": dimension_name,
        "element_measure": element_measure,
        "total_measure": float(binary.sum() * element_measure),
        "largest_component_measure": float(measures.max()) if measures.size else 0.0,
    }
    # Dimension-specific keys make serialized reports self-explanatory while
    # retaining the established 3D names for existing callers.
    result[f"active_{element_name}"] = int(binary.sum())
    result[f"active_{element_name[:-1]}_fraction"] = float(np.mean(binary))
    result[f"component_{element_name}"] = sizes.tolist()
    result[f"component_{dimension_name}s"] = measures.tolist()
    result[f"component_{element_name}_mean"] = float(sizes.mean()) if sizes.size else 0.0
    result[f"component_{element_name}_median"] = float(np.median(sizes)) if sizes.size else 0.0
    result[f"component_{element_name}_std"] = float(sizes.std()) if sizes.size else 0.0
    result[f"largest_component_{element_name}"] = int(sizes.max()) if sizes.size else 0
    result[f"total_{dimension_name}"] = float(binary.sum() * element_measure)
    result[f"largest_component_{dimension_name}"] = float(measures.max()) if measures.size else 0.0
    return result


def component_measure_distribution(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> np.ndarray:
    """Return component areas (2D) or volumes (3D) in spacing units."""

    stats = connected_component_statistics(
        mask, threshold=threshold, spacing=spacing, connectivity=connectivity
    )
    return np.asarray(stats["component_measures"], dtype=np.float64)


def component_area_distribution(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> np.ndarray:
    """Return one physical (or pixel-unit) area per 2D component."""

    array = to_numpy(mask, name="mask")
    if array.ndim != 2:
        raise ValueError(f"component_area_distribution expects a 2D mask, got {array.shape}.")
    return component_measure_distribution(
        array, threshold=threshold, spacing=spacing, connectivity=connectivity
    )


def component_volume_distribution(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> np.ndarray:
    """Return one physical (or voxel-unit) volume per 3D component."""

    array = to_numpy(mask, name="mask")
    if array.ndim != 3:
        raise ValueError(f"component_volume_distribution expects a 3D mask, got {array.shape}.")
    return component_measure_distribution(
        array, threshold=threshold, spacing=spacing, connectivity=connectivity
    )


def _surface(binary: np.ndarray, connectivity: Optional[int]) -> np.ndarray:
    structure = _structure(binary.ndim, connectivity)
    return binary & ~ndimage.binary_erosion(binary, structure=structure, border_value=0)


def _directed_surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    connectivity: Optional[int],
) -> np.ndarray:
    source_surface = _surface(source, connectivity)
    target_surface = _surface(target, connectivity)
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return np.asarray(distance_map[source_surface], dtype=np.float64)


def surface_distances(
    predicted: Any,
    reference: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed surface distances (prediction-to-reference, reference-to-prediction).

    Distances use the units of ``spacing``. Two empty masks return two empty
    arrays; exactly one empty mask has undefined/infinite distances.
    """

    pred, ref = _binary_pair(predicted, reference, threshold)
    checked_spacing = validate_spacing(spacing, pred.ndim)
    pred_empty, ref_empty = not np.any(pred), not np.any(ref)
    if pred_empty and ref_empty:
        return np.empty(0), np.empty(0)
    if pred_empty or ref_empty:
        return np.asarray([np.inf]), np.asarray([np.inf])
    return (
        _directed_surface_distances(pred, ref, checked_spacing, connectivity),
        _directed_surface_distances(ref, pred, checked_spacing, connectivity),
    )


def hausdorff_distance(
    predicted: Any,
    reference: Any,
    *,
    percentile: float = 100.0,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> float:
    """Return symmetric Hausdorff distance, or a robust percentile such as HD95.

    Two empty masks return 0; exactly one empty mask returns infinity.
    """

    if not np.isfinite(percentile) or percentile <= 0.0 or percentile > 100.0:
        raise ValueError("percentile must lie in (0, 100].")
    forward, backward = surface_distances(
        predicted, reference, threshold=threshold, spacing=spacing, connectivity=connectivity
    )
    if forward.size == 0 and backward.size == 0:
        return 0.0
    combined = np.concatenate([forward, backward])
    if np.isinf(combined).any():
        return float("inf")
    return float(np.percentile(combined, percentile))


def average_surface_distance(
    predicted: Any,
    reference: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> float:
    """Return the symmetric average surface distance in spacing units."""

    forward, backward = surface_distances(
        predicted, reference, threshold=threshold, spacing=spacing, connectivity=connectivity
    )
    if forward.size == 0 and backward.size == 0:
        return 0.0
    return float(0.5 * (np.mean(forward) + np.mean(backward)))


def surface_distance_statistics(
    predicted: Any,
    reference: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> dict[str, float]:
    """Return ASSD, median, HD95, and maximum symmetric surface distance."""

    forward, backward = surface_distances(
        predicted, reference, threshold=threshold, spacing=spacing, connectivity=connectivity
    )
    if forward.size == 0 and backward.size == 0:
        return {"average": 0.0, "median": 0.0, "hausdorff_95": 0.0, "hausdorff": 0.0}
    combined = np.concatenate([forward, backward])
    if np.isinf(combined).any():
        return {
            "average": float("inf"),
            "median": float("inf"),
            "hausdorff_95": float("inf"),
            "hausdorff": float("inf"),
        }
    return {
        "average": float(0.5 * (np.mean(forward) + np.mean(backward))),
        "median": float(np.median(combined)),
        "hausdorff_95": float(np.percentile(combined, 95.0)),
        "hausdorff": float(np.max(combined)),
    }
