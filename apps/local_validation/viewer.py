"""Read-only slice preparation for the local viewer, independent of Streamlit.

NIfTI 3D spatial axes are permuted/flipped to closest RAS for display only.
This is not resampling, registration or removal of obliquity. NumPy axes have
no anatomical meaning unless supplied by the user outside this viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap

from synthetic_imaging_validation import load_image


@dataclass(frozen=True)
class PreviewImage:
    array: np.ndarray
    spacing: tuple[float, ...]
    affine: Optional[np.ndarray]
    anatomical: bool
    oblique: bool
    metadata: dict


def prepare_image(path: Path, channel_axis: Optional[int] = None, channel: int = 0) -> PreviewImage:
    """Load one selected case/channel; never modify source data or metric inputs."""
    data = load_image(path)
    array = data.array
    original_shape = array.shape
    spacing = data.spacing or (1.0,) * array.ndim
    affine = data.affine
    axis = None
    if channel_axis is not None:
        if not -array.ndim <= channel_axis < array.ndim:
            raise ValueError("Channel axis is outside the image dimensions.")
        axis = channel_axis % array.ndim
        if not 0 <= channel < array.shape[axis]:
            raise ValueError("Selected channel is outside the image dimensions.")
        if affine is not None and array.ndim == 4 and axis != 3:
            raise ValueError("For 4D NIfTI, select the last axis (3 or -1) as the channel/frame axis.")
        array = np.take(array, channel, axis=axis)
        spacing = tuple(s for i, s in enumerate(spacing) if i != axis)
    if array.ndim not in (2, 3):
        raise ValueError("The viewer needs 2D/3D spatial data. For a 4D file, set the channel axis and select a channel/frame.")
    anatomical = affine is not None and array.ndim == 3
    oblique = False
    orientation = "Unknown (array axes)"
    if anatomical:
        orientation = "".join(nib.aff2axcodes(affine))
        image = nib.as_closest_canonical(nib.Nifti1Image(array, affine))
        array = np.asanyarray(image.dataobj)
        affine = image.affine
        spacing = tuple(float(s) for s in nib.affines.voxel_sizes(affine))
        oblique = bool(np.max(nib.affines.obliquity(affine)) > 1e-3)
    # 2D/channel-reduced images stay in array order. Retain the original affine
    # only for grid comparisons; do not invent anatomical plane labels.
    metadata = {"File": path.name, "Stored shape": str(original_shape), "Display shape": str(array.shape),
                "Display spacing": str(tuple(round(s, 5) for s in spacing)),
                "Stored orientation": orientation, "Minimum": float(array.min()), "Maximum": float(array.max()),
                "Channel / frame": str(channel) if axis is not None else "None",
                "Units": "Header spacing units" if data.spacing is not None else "Pixels / voxels (unit spacing)"}
    return PreviewImage(array, tuple(spacing), affine, anatomical, oblique, metadata)


def grids_match(real: PreviewImage, synthetic: PreviewImage) -> bool:
    """Allow overlays only when displayed shapes and available geometry agree."""
    if real.array.shape != synthetic.array.shape or not np.allclose(real.spacing, synthetic.spacing, atol=1e-5, rtol=0):
        return False
    if (real.affine is None) != (synthetic.affine is None):
        return False
    return real.affine is None or bool(np.allclose(real.affine, synthetic.affine, atol=1e-5, rtol=0))


def extract_slice(image: PreviewImage, axis: int, index: int) -> tuple[np.ndarray, float, str, str]:
    """Return display pixels, physical aspect and axis labels for a 2D slice."""
    if image.array.ndim == 2:
        return image.array, image.spacing[0] / image.spacing[1], "Array axis 1 →", "Array axis 0 →"
    if axis not in (0, 1, 2) or not 0 <= index < image.array.shape[axis]:
        raise ValueError("Invalid slice axis or index.")
    plane = np.take(image.array, index, axis=axis).T
    remaining = [i for i in range(3) if i != axis]
    aspect = image.spacing[remaining[1]] / image.spacing[remaining[0]]
    if image.anatomical:
        labels = {0: ("P → A", "I → S"), 1: ("L → R", "I → S"), 2: ("L → R", "P → A")}
        xlabel, ylabel = labels[axis]
    else:
        xlabel, ylabel = f"Array axis {remaining[0]} →", f"Array axis {remaining[1]} →"
    return plane, aspect, xlabel, ylabel


def slice_figure(real: PreviewImage, synthetic: PreviewImage, *, axis: int = 2, index: int = 0,
                 window: tuple[float, float] = (0.0, 1.0), masks: bool = False,
                 threshold: float = 0.5, difference: bool = False, cmap: str = "gray") -> Figure:
    """Plot shared-window slices and an optional exact-grid difference/overlap panel.

    3D slices use neurological display (R on the right for axial/coronal RAS).
    Binary masks use the same ``>= threshold`` rule as the metric package.
    Windowing, channel selection and orientation here affect only this figure.
    """
    if real.array.ndim != synthetic.array.ndim:
        raise ValueError("Select images with the same spatial dimensionality to compare slices.")
    low, high = window
    if not np.isfinite([low, high, threshold]).all() or high <= low:
        raise ValueError("Display bounds must be finite and upper must exceed lower; threshold must be finite.")
    compatible = grids_match(real, synthetic)
    if difference and not compatible:
        raise ValueError("Overlay/difference requires matching geometry and shape.")
    a, aspect_a, x_a, y_a = extract_slice(real, axis, index)
    b, aspect_b, x_b, y_b = extract_slice(synthetic, axis, index)
    fig = Figure(figsize=(14 if difference else 10, 4.8), layout="constrained", facecolor="#0e1926")
    FigureCanvasAgg(fig)
    planes = [a, b]
    titles = ["Real", "Synthetic"]
    if masks:
        planes = [(a >= threshold).astype(np.uint8), (b >= threshold).astype(np.uint8)]
    if difference:
        planes.append(planes[0] + 2 * planes[1] if masks else np.abs(a.astype(float) - b.astype(float)))
        titles.append("Overlap" if masks else "Absolute difference")
    origin = "upper" if real.array.ndim == 2 else "lower"
    for i, (plane, title) in enumerate(zip(planes, titles)):
        ax = fig.add_subplot(1, len(planes), i + 1)
        ax.set_facecolor("#0e1926")
        ax.set_title(title, color="#e7f0f7")
        if masks:
            colors = ["#0e1926", "#33c7da", "#efad52", "#ecf5fc"] if i == 2 else ["#0e1926", "#33c7da" if i == 0 else "#efad52"]
            ax.imshow(plane, cmap=ListedColormap(colors), vmin=0, vmax=3 if i == 2 else 1, interpolation="nearest", origin=origin,
                      aspect=aspect_b if i == 1 else aspect_a)
        else:
            ax.imshow(plane, cmap="magma" if i == 2 else cmap, vmin=0 if i == 2 else low,
                      vmax=high - low if i == 2 else high, interpolation="nearest", origin=origin,
                      aspect=aspect_b if i == 1 else aspect_a)
        ax.set_xlabel(x_b if i == 1 else x_a, color="#b9cbd9")
        ax.set_ylabel(y_b if i == 1 else y_a, color="#b9cbd9")
        ax.set_xticks([])
        ax.set_yticks([])
    return fig
