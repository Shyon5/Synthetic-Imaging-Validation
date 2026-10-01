"""Self-contained browser slice viewer, with no remote assets or patient cache."""
import base64
import gzip
import json
import math
from pathlib import Path
import numpy as np
from .viewer import grids_match

# Two 256-cubed float32 volumes already occupy 128 MiB before compression.
# Browser decoding and rendering need additional copies; reject larger payloads
# before allocating them, rather than silently reducing an Original preview.
MAX_PREVIEW_BYTES = 128 * 1024 * 1024


def viewer_html(real, synthetic, *, masks=False, threshold=0.5, max_side=128):
    """Encode bounded float32 previews; metrics still use original arrays.

    ``max_side`` is an upper bound, not a target size. ``None`` keeps every
    voxel (Original mode), subject to the explicit browser memory guard.
    Values are still float32 for display. Slice/window changes run in the browser.
    """
    if real.array.ndim != synthetic.array.ndim:
        raise ValueError("The images have different spatial dimensionality.")
    if max_side not in (128, 256, None) or not np.isfinite(threshold):
        raise ValueError("Choose a preview limit of 128, 256 or Original, and a finite threshold.")
    step = 1 if max_side is None else max(1, math.ceil(max(*real.array.shape, *synthetic.array.shape) / max_side))
    size = sum(math.prod((n + step - 1) // step for n in image.array.shape) * 4
               for image in (real, synthetic))
    if size > MAX_PREVIEW_BYTES:
        raise ValueError(f"This preview needs {size / 1024**2:.1f} MiB for voxel values alone, above the "
                         f"{MAX_PREVIEW_BYTES / 1024**2:g} MiB browser safety limit. Choose Up to 128 or Up to 256. "
                         "No voxels were silently skipped; metric calculations still use the original files.")
    volumes = []
    for image in (real, synthetic):
        data = image.array[tuple(slice(None, None, step) for _ in image.array.shape)]
        if masks:
            # Threshold at source precision, before float32 conversion. Values
            # just below the threshold must not round up into the foreground.
            data = data >= threshold
        if np.max(np.abs(data)) > np.finfo(np.float32).max:
            raise ValueError("Intensities exceed the float32 preview range.")
        data = np.ascontiguousarray(data, dtype="<f4")
        volumes.append({"data": base64.b64encode(gzip.compress(data.tobytes(), compresslevel=1)).decode("ascii"),
                        "shape": list(data.shape), "spacing": list(image.spacing), "anatomical": image.anatomical})
    low = min(real.metadata["Minimum"], synthetic.metadata["Minimum"])
    high = max(real.metadata["Maximum"], synthetic.metadata["Maximum"])
    if high <= low:
        high = low + max(1.0, abs(low) * 0.01)
    from matplotlib import colormaps
    palette = np.rint(colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
    payload = {"volumes": volumes, "step": step, "masks": masks, "threshold": threshold, "palette": palette,
               "compatible": grids_match(real, synthetic), "low": low, "high": high}
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    return Path(__file__).with_name("viewer.html").read_text(encoding="utf-8").replace("/*PAYLOAD*/", encoded)
