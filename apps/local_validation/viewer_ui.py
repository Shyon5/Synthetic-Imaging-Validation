"""Isolated preview controls; slice navigation runs entirely in the browser."""
from pathlib import Path
import tempfile
import streamlit as st
from .service import make_demo, plan_pairs
from .viewer import prepare_image
from .client_viewer import viewer_html


@st.fragment
def render_viewer(data_root: Path, mode: str, *, real: str, synthetic: str, manifest: str,
                  recursive: bool, masks: bool, channel_text: str, threshold: float,
                  pairing_options=None) -> None:
    """Keep one selected pair per session, never in a shared patient-data cache."""
    if not st.toggle("Open image viewer", key="open_viewer",
                     help="Read-only preview. Slice and window controls run in your browser without recalculating metrics."):
        st.session_state.pop("_preview_cache", None)
        temporary = st.session_state.pop("_preview_demo", None)
        if temporary:
            temporary.cleanup()
        return
    with st.container(border=True):
        st.markdown("**Image viewer**")
        try:
            root = data_root
            if mode == "Try example data":
                if "_preview_demo" not in st.session_state:
                    st.session_state["_preview_demo"] = tempfile.TemporaryDirectory(prefix="siv-viewer-")
                root = Path(st.session_state["_preview_demo"].name) / ("masks" if masks else "images")
                if not root.exists():
                    make_demo(root, "masks" if masks else "images")
                manifest = "pairs.csv"
            selected_mode = {"Two files": "files", "Two folders": "directories", "CSV manifest": "manifest"}.get(mode, "manifest")
            pairs = plan_pairs(root, selected_mode, real=real, synthetic=synthetic, manifest=manifest,
                               recursive=recursive, **(pairing_options or {}))
            selected = st.selectbox("Case to inspect", range(len(pairs)), format_func=lambda i: pairs[i].key,
                                    key="preview_case", help="Loads one real/synthetic pair at a time. Check that the two files represent the anatomy you intend to compare.")
            pair = pairs[selected]
            axis = int(channel_text) if channel_text.strip() else None
            channel = int(st.number_input("Preview channel / frame", min_value=0, value=0,
                          help="Preview only. For 4D NIfTI set Channel axis to 3 or -1.")) if axis is not None else 0
            quality = st.selectbox("Preview resolution", [128, 256, "original"],
                                    format_func=lambda value: "Original (all voxels)" if value == "original" else f"Up to {value} per axis",
                                    help="128 and 256 are maximum sizes, not a forced resize. If both images fit, no voxels are skipped. Original keeps every voxel. All options affect only the viewer.")
            st.caption("For example, two 192 × 192 × 192 volumes stay at 192³ with Up to 256. Larger images are shown using regularly spaced pixels/voxels; they are not stretched to a fixed size.")
            if quality == "original":
                st.warning("Original loads every voxel of both images into the browser. Large volumes can take longer to open and use a lot of RAM. Previews above 128 MiB of uncompressed float32 voxel data are refused; choose a smaller preview if needed. Metric inputs are never reduced.")
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in (pair.real, pair.synthetic)) + (axis, channel, masks, threshold, quality)
            cached = st.session_state.get("_preview_cache")
            if cached is None or cached[0] != signature:
                with st.spinner("Preparing browser preview…"):
                    a, b = prepare_image(pair.real, axis, channel), prepare_image(pair.synthetic, axis, channel)
                    html = viewer_html(a, b, masks=masks, threshold=threshold,
                                       max_side=None if quality == "original" else quality)
                cached = (signature, html, a.metadata, b.metadata, a.oblique or b.oblique, a.anatomical and b.anatomical)
                st.session_state["_preview_cache"] = cached
            _, html, real_info, synth_info, oblique, anatomical = cached
            if anatomical:
                st.caption("NIfTI images are turned to a consistent viewing orientation. In axial and coronal views, the patient's right is on the right of the screen. This does not align the images or change the files.")
            if oblique:
                st.warning("These images were acquired at an angle. The viewer shows their existing slice planes, so the anatomical direction labels are approximate.")
            # Only our trusted template plus JSON-encoded numeric data enters
            # this JavaScript-enabled iframe; never accept uploaded HTML.
            st.iframe(html, height="content")
            with st.expander("Image details"):
                st.dataframe([{"Role": "Real", **real_info}, {"Role": "Synthetic", **synth_info}], width="stretch")
                st.caption("Image proportions follow the spacing stored in the file. Arrays without spacing use equal-sized pixels/voxels. Changing spacing in the metric settings does not change this preview. Original preserves the grid, not the stored number precision: display values use float32.")
        except (ValueError, TypeError, OSError, MemoryError) as exc:
            st.session_state.pop("_preview_cache", None)
            st.error(f"Preview unavailable: {exc}")
