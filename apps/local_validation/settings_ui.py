"""Evaluation controls with reusable, explicitly applied settings."""
import streamlit as st
from .metric_labels import metric_label, metric_help
from .profiles import IMAGE_METRICS, MASK_METRICS, defaults, validate_settings, export_profile, import_profile


def _text(values):
    return ", ".join(map(str, values)) if values is not None else ""


def evaluation_settings(masks):
    """Return active settings; importing a profile never restores file paths."""
    key = "mask" if masks else "image"
    current = st.session_state.get(f"settings_{key}", defaults(masks))
    with st.expander("Save or reuse evaluation settings"):
        st.caption("Reuse the same metrics and parameters on another dataset. Files, case labels and folders are not included.")
        uploaded = st.file_uploader("Settings JSON", type=["json"], key=f"profile_{key}",
                                    help="Choose an exported profile or settings.json from a previous paired evaluation.")
        if uploaded is not None:
            st.caption("Settings file received. Ready to load.")
        if st.button("Load settings", key=f"load_{key}", disabled=uploaded is None):
            try:
                if uploaded is None:
                    raise ValueError("Choose a settings file first.")
                restored = import_profile(uploaded.getvalue(), masks)
                st.session_state[f"settings_{key}"] = restored
                st.session_state[f"settings_revision_{key}"] = st.session_state.get(f"settings_revision_{key}", 0) + 1
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    initial = current
    revision = st.session_state.get(f"settings_revision_{key}", 0)
    choices = MASK_METRICS if masks else IMAGE_METRICS
    with st.popover("Metric guide"):
        for name in choices:
            st.markdown(f"**{metric_label(name)}** — {metric_help(name)}")
    metrics = st.multiselect("Metrics", choices, default=initial["metrics"], format_func=metric_label,
                            key=f"metrics_{key}_{revision}", help="Select metrics, then apply the settings below.")
    scores = [] if masks else st.multiselect(
        "Experimental scores (optional)", ["similarity", "intensity_distribution"],
        default=initial["scores"], format_func=metric_label, key=f"scores_{key}_{revision}",
        help=("Experimental 0–100 summaries, not clinical quality ratings. Scores use a fixed intensity interval width. "
              "All input values must fit within the declared bounds. Normalization to [0, 1] is not required."))
    st.caption("Only settings used by your selection are shown. Apply changes before running.")
    with st.form(f"evaluation_{key}_{revision}"):
        left, right = st.columns(2)
        width = initial["data_range"]
        if not masks and set(metrics) & {"psnr", "ssim", "ms_ssim"} and "similarity" not in scores:
            raw_width = left.text_input("Intensity range width", value="" if width is None else str(width),
                                       help="Expected maximum minus minimum. For [0, 20], enter 20. This does not rescale images.")
        else:
            raw_width = "" if width is None else str(width)
        workers = right.number_input("Parallel workers", min_value=1, max_value=16, value=initial["workers"],
                                     help="Pairs evaluated at once. Start with 1 for large volumes; more workers need more RAM.")
        low, high = initial["score_range"]
        if scores:
            low = left.number_input("Score interval: lower bound", value=float(low), help="Use fixed bounds for the study, not different bounds for each image.")
            high = right.number_input("Score interval: upper bound", value=float(high), help="All voxels must fit within these bounds. No clipping is applied.")
        threshold = initial["threshold"]
        if masks:
            threshold = right.number_input("Binary mask threshold", value=float(threshold), help="Values at or above the threshold are foreground.")
        bins = initial["bins"]
        if set(metrics) & {"js", "kl", "histogram", "compare_distributions"} or "intensity_distribution" in scores:
            bins = left.number_input("Histogram bins", min_value=2, max_value=4096, value=bins)
        spacing, channel = _text(initial["spacing"]), initial["channel_axis"]
        if masks:
            spacing = left.text_input("Spacing override (optional)", value=spacing, placeholder="1, 1, 2",
                                      help="Pixel/voxel size in array-axis order. Leave blank to use file metadata.")
        else:
            channel = right.text_input("Channel axis (optional)", value="" if channel is None else str(channel),
                                       help="Leave blank for a single-channel image. -1 means the last array axis.")
        with st.expander("Advanced parameters"):
            batch = initial["batch_axis"]
            if not masks and set(metrics) & {"ssim", "ms_ssim"} and not scores:
                batch = st.text_input("Batch axis (optional)", value="" if batch is None else str(batch),
                                     help="For arrays with multiple images. Applies to SSIM/MS-SSIM only; separate files give clearer per-case reports.")
            normalization = initial.get("nrmse_normalization", "range")
            if "nrmse" in metrics:
                normalization = st.selectbox("NRMSE normalization", ["range", "mean", "l2"],
                                             index=["range", "mean", "l2"].index(normalization),
                                             help="Reference range, absolute mean, or root-mean-square intensity. Keep the same choice across comparisons.")
            border = _text(initial["border_width"])
            connectivity = initial["connectivity"]
            if masks:
                border = st.text_input("Border width (pixels/voxels)", value=border)
                connectivity = st.selectbox("Connectivity for extended mask reports", [None, 1, 2, 3],
                                            index=[None, 1, 2, 3].index(connectivity),
                                            format_func=lambda n: "Default" if n is None else str(n),
                                            help="1 connects direct neighbours. Higher values include diagonals. At most 2 in 2D, 3 in 3D. Extended reports only.")
            percentiles = _text(initial.get("percentiles", [1, 5, 25, 50, 75, 95, 99]))
            if set(metrics) & {"intensity_statistics", "compare_distributions"}:
                percentiles = st.text_input("Intensity percentiles", value=percentiles)
            backend = initial["ms_ssim_backend"]
            if "ms_ssim" in metrics or "similarity" in scores:
                backend = st.selectbox("MS-SSIM backend", ["numpy", "torchmetrics"],
                                       index=["numpy", "torchmetrics"].index(backend),
                                       help="NumPy is included. TorchMetrics needs extra dependencies and is not in the standard Docker image.")
            mismatch = st.checkbox("Allow different NIfTI geometry", value=initial.get("allow_spatial_mismatch", False),
                                   help="Usually leave off. This skips geometry checks; it does not align images.")
            plot_fields = st.text_input("Plot fields (optional)", value=_text(initial.get("plot_metrics")),
                                        help="Exported metric names, separated by commas. Blank uses the selected metrics.")
            pdf_font = st.text_input("PDF font file (optional)", value=current.get("pdf_font") or "",
                                     help="A .ttf file inside the data folder, for characters missing from the default font.")
        apply = st.form_submit_button("Apply evaluation settings")
    if apply:
        try:
            candidate = dict(metrics=metrics, scores=scores, score_range=[low, high],
                             data_range=float(raw_width) if raw_width.strip() else None,
                             workers=int(workers), threshold=threshold, bins=int(bins),
                             spacing=[float(v) for v in spacing.split(",")] if spacing.strip() else None,
                             channel_axis=int(channel) if channel not in (None, "") else None,
                             batch_axis=None if scores else (int(batch) if batch not in (None, "") else None),
                             border_width=[int(v) for v in border.split(",")], connectivity=connectivity,
                             percentiles=[float(v) for v in percentiles.split(",")],
                             ms_ssim_backend=backend, allow_spatial_mismatch=mismatch,
                             plot_metrics=[v.strip() for v in plot_fields.split(",") if v.strip()] or None,
                             nrmse_normalization=normalization)
            current = validate_settings(candidate, masks)
            current["pdf_font"] = pdf_font.strip() or None
            st.session_state[f"settings_{key}"] = current
            st.success("Settings applied.")
        except ValueError as exc:
            st.error(f"Settings not applied: {exc}")
    if metrics != current["metrics"] or scores != current["scores"]:
        st.warning("Your selection has changed. Apply the settings before checking or running.")
    st.caption("Active: " + ", ".join(metric_label(n) for n in current["metrics"] + current["scores"]))
    st.download_button("Download evaluation settings", export_profile(current, masks),
                       file_name=f"{key}_evaluation_settings.json", mime="application/json",
                       on_click="ignore", key=f"export_{key}")
    return current.copy()
