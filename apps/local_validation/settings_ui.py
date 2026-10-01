"""Batch metric-setting changes rather than rerunning on every selection."""
import streamlit as st
from .metric_labels import metric_label, metric_help

IMAGE_METRICS = ["mae", "mse", "rmse", "nrmse", "psnr", "ssim", "ms_ssim", "wasserstein", "js", "kl",
                 "intensity_statistics", "histogram", "compare_distributions"]
MASK_METRICS = ["dice", "iou", "hausdorff", "hausdorff95", "average_surface_distance", "measure_ratio",
                "foreground_fraction", "connected_components", "border", "surface_statistics",
                "component_measures", "distance_to_border", "centroid", "mask_spatial_report"]


def evaluation_settings(masks):
    """Return the last explicitly submitted, validated settings for this image kind."""
    key = "mask" if masks else "image"
    default_metrics = ["dice", "iou", "hausdorff95"] if masks else ["mae", "rmse", "ssim", "ms_ssim", "wasserstein"]
    defaults = {"metrics": default_metrics, "scores": [], "score_range": [0.0, 1.0], "data_range": 1.0,
                "workers": 1, "threshold": 0.5, "bins": 64, "spacing": None, "channel_axis": None,
                "batch_axis": None, "border_width": [1], "connectivity": None, "ms_ssim_backend": "numpy"}
    choices = MASK_METRICS if masks else IMAGE_METRICS
    with st.popover("Metric guide"):
        for name in choices:
            st.markdown(f"**{metric_label(name)}** — {metric_help(name)}")
    with st.form(f"evaluation_{key}"):
        metrics = st.multiselect("Metrics", choices, default=default_metrics, format_func=metric_label,
                                 help="Changes take effect when you press Apply evaluation settings.")
        scores = [] if masks else st.multiselect("Experimental scores (optional)", ["similarity", "intensity_distribution"], format_func=metric_label,
                                                help="Combine metric values into an experimental 0–100 index. Higher means closer agreement under the chosen settings, not clinical quality. Unlike raw errors, scores require a fixed interval containing all input intensities.")
        left, right = st.columns(2)
        width = left.text_input("Intensity range width", value="1", help="The expected maximum minus minimum, used by PSNR, SSIM and MS-SSIM. For an interval [0, 20], enter 20. This does not rescale the images. With similarity score enabled, the app uses the width of the score interval instead.")
        workers = right.number_input("Parallel workers", min_value=1, max_value=16, value=1, help="How many image pairs to evaluate at once. Start with 1 for large volumes; more workers use more RAM and are not always faster.")
        low, high = 0.0, 1.0
        if not masks:
            low = left.number_input("Score interval: lower bound", value=0.0, help="Must include the minimum of every input, not just the displayed slice. Use Inspect intensity bounds below if unsure.")
            high = right.number_input("Score interval: upper bound", value=1.0, help="Must include all inputs. Keep these bounds fixed across compared runs. No clipping or rescaling is performed.")
            st.caption("Score bounds describe the values in your files, not the viewer's display window. Keep [0, 1] only if the data are already in that interval. CT, PET and other intensity scales can use different bounds; normalization to [0, 1] is not required.")
            with st.expander("Why do scores check the interval when other metrics still run?"):
                st.markdown("MAE and Wasserstein report an error in the data's own units, so they do not need fixed lower and upper bounds. Scores use a shared scale to turn those errors into a 0–100 index: they divide MAE or Wasserstein by the interval width. The distribution score also uses that interval for its histogram bins.")
                st.markdown("For example, values of 100 and 110 have an absolute error of 10. That calculation is valid, but declaring that both values lie in [0, 1] is not. The score stops instead of accepting that inconsistent setup or dropping values from the histogram.")
                st.markdown("PSNR, SSIM and MS-SSIM also depend on the intensity range width, but do not perform the score's full interval check. A returned value does not prove that the range was appropriate. Use Inspect intensity bounds below, choose bounds consistent with your preprocessing, and keep them fixed across comparisons. Widening the range just to improve a score makes comparisons misleading.")
        spacing = left.text_input("Spacing override (optional)", placeholder="1, 1, 2", help="Pixel/voxel sizes in array-axis order, such as 1, 1, 2. Leave blank to use file metadata, or unit spacing if none exists. This changes distance units, not the image grid.")
        channel = right.text_input("Channel axis (optional)", placeholder="-1", help="Leave blank for a single-channel image or volume. For multiple channels, enter the array axis that holds them; -1 means the last axis.")
        bins = left.number_input("Histogram bins", min_value=2, max_value=4096, value=64, help="How many intensity intervals to use for histogram comparisons. Keep this number fixed between experiments; more bins is not automatically better.")
        threshold = right.number_input("Binary mask threshold", value=0.5, help="Foreground includes values equal to the threshold.")
        with st.expander("Advanced parameters"):
            batch = st.text_input("Batch axis (optional)", help="Use only if one array contains several images along an extra axis. This option applies to SSIM/MS-SSIM, not all metrics, and cannot be used with scores. Separate files are clearer when you need one result per case.")
            border = st.text_input("Border width (pixels/voxels)", value="1", help="One width for every axis, or comma-separated per-axis widths.")
            connectivity = st.selectbox("Connectivity for extended mask reports", [None, 1, 2, 3], format_func=lambda n: "Default" if n is None else str(n), help="Controls which neighbouring foreground pixels/voxels count as connected. 1 uses direct neighbours; higher values also connect diagonals. Choose at most 2 for 2D, or 3 for 3D. Applies to extended mask reports, not the original CLI metrics.")
            percentiles = st.text_input("Intensity percentiles", value="1, 5, 25, 50, 75, 95, 99")
            backend = st.selectbox("MS-SSIM backend", ["numpy", "torchmetrics"], help="NumPy is the lightweight default. Torchmetrics requires the optional torch extra; it is not bundled in the standard Docker image.")
            mismatch = st.checkbox("Allow different NIfTI geometry", help="Usually leave this off. It bypasses checks on file spacing and orientation, but does not align images or change their shapes. Only use it when you have independently established that the comparison is meaningful.")
            plot_fields = st.text_input("Plot fields (optional)", help="Comma-separated exported identifiers. Defaults to scores, or up to six scalar metric fields.")
            pdf_font = st.text_input("PDF font file (optional)", help="Path to a .ttf font inside the data folder, for labels not covered by the default PDF font.")
        apply = st.form_submit_button("Apply evaluation settings")
    if apply:
        try:
            options = {"metrics": metrics, "scores": scores, "score_range": [low, high],
                       "data_range": high - low if "similarity" in scores else (float(width) if width.strip() else None),
                       "spacing": [float(v) for v in spacing.split(",")] if spacing.strip() else None,
                       "channel_axis": int(channel) if channel.strip() else None,
                       "batch_axis": int(batch) if batch.strip() else None,
                       "workers": int(workers), "bins": int(bins), "threshold": threshold,
                       "border_width": [int(v) for v in border.split(",")], "connectivity": connectivity,
                       "percentiles": [float(v) for v in percentiles.split(",")],
                       "ms_ssim_backend": backend, "allow_spatial_mismatch": mismatch,
                       "pdf_font": pdf_font.strip() or None,
                       "plot_metrics": [v.strip() for v in plot_fields.split(",") if v.strip()] or [s + "_score" for s in scores] or metrics[:6]}
            if scores and high <= low:
                raise ValueError("Score upper bound must exceed lower bound.")
            st.session_state[f"settings_{key}"] = options
            st.success("Settings applied.")
        except ValueError as exc:
            st.error(f"Settings not applied: {exc}")
    options = st.session_state.get(f"settings_{key}", defaults)
    st.caption("Ready to calculate: " + ", ".join(metric_label(name) for name in options["metrics"] + options["scores"]) + ". Press Apply evaluation settings after making changes, then Run validation.")
    return options.copy()
