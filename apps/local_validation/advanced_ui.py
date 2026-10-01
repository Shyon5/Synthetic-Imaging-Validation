"""Additional workflows, kept separate from the ordinary paired-image form."""
import json
from pathlib import Path
import uuid
import streamlit as st
from synthetic_imaging_validation import append_history, plot_history
from .advanced import FEATURE_METRICS, COHORT_METRICS, evaluate_features, evaluate_cohorts
from .service import catalogue, save_report
from .metric_labels import metric_label


def downloads(destination, *, key="extra"):
    """Download existing files without triggering a rerun or recalculation."""
    for file in sorted(destination.iterdir()):
        if file.is_file():
            st.download_button(file.name, file.read_bytes(), file_name=file.name,
                               key=f"{key}_{file.name}", on_click="ignore")


def render_advanced(workflow, data_root, output_root):
    """Show explicit inputs for cohort, feature and history/report operations."""
    try:
        if workflow == "Reports and history":
            _reports(output_root)
            return
        images, manifests, folders = catalogue(data_root)
        if workflow == "Feature distributions":
            st.subheader("Compare feature distributions")
            st.info("Choose two .npy or .npz files containing features you have already extracted: one row per sample, one column per feature. Both datasets must use the same encoder and preprocessing. The app compares these vectors; it does not extract them from images.")
            options = [p for p in images if Path(p).suffix in (".npy", ".npz")]
            with st.form("feature_form"):
                real = st.selectbox("Real features", options or [""])
                synthetic = st.selectbox("Synthetic features", options or [""])
                metrics = st.multiselect("Feature metrics", FEATURE_METRICS, default=["frechet"], format_func=metric_label)
                left, right = st.columns(2)
                k = left.number_input("Neighbours (precision/recall)", min_value=1, value=3, help="Each domain needs more samples than k, including each label group.")
                epsilon = right.number_input("Covariance regularization", min_value=0.0, value=0.000001, format="%.8f", help="A small value added to the covariance diagonal to help the Fréchet calculation remain numerically stable. Keep the default unless your evaluation protocol specifies another value.")
                sigma = left.text_input("RBF bandwidth (blank = median heuristic)", help="Controls the distance scale used by RBF MMD. Leave blank to estimate it from median feature distances, or enter a strictly positive value and keep it fixed across comparisons.")
                projections = right.number_input("Sliced Wasserstein projections", min_value=1, value=128)
                seed = left.number_input("Projection seed", min_value=0, value=42)
                label_real = st.selectbox("Real label CSV (optional)", [""] + manifests)
                label_synth = st.selectbox("Synthetic label CSV (optional)", [""] + manifests)
                label_column = st.text_input("Label column", value="label", help="Exactly one label per feature row, in matching order. Not voxel labels.")
                submit = st.form_submit_button("Evaluate features", type="primary")
            st.caption("Feature metrics compare groups of samples, not the quality of an individual patient. Small groups give less reliable estimates, and KID/MMD can be negative. Metrics that compare every sample with every other sample are limited to 5,000 samples per dataset to control memory use.")
            if submit:
                if bool(label_real) != bool(label_synth):
                    raise ValueError("Select both label CSV files or neither.")
                kwargs = {"frechet": {"covariance_epsilon": epsilon}, "feature_precision_recall": {"k": k},
                          "rbf_mmd": {"sigma": float(sigma) if sigma.strip() else None},
                          "sliced_wasserstein": {"num_projections": projections, "seed": seed}}
                with st.spinner("Comparing features…"):
                    report = evaluate_features(data_root, real, synthetic, metrics, kwargs=kwargs,
                                               labels=(label_real, label_synth, label_column) if label_real else None)
                    st.session_state["advanced_completed"] = save_report(report, {"metrics": metrics, "kwargs": kwargs,
                        "real": real, "synthetic": synthetic, "labels": [label_real, label_synth, label_column]}, output_root)
        else:
            st.subheader("Compare independent intensity cohorts")
            st.info("Use this when the real and synthetic datasets have no one-to-one pairing. The app combines the intensity values from each folder and compares the two distributions. Larger images contribute more values, so they carry more weight. This does not compare the position or shape of anatomy.")
            with st.form("cohort_form"):
                real = st.selectbox("Real cohort folder", folders)
                synthetic = st.selectbox("Synthetic cohort folder", folders)
                metrics = st.multiselect("Cohort metrics", COHORT_METRICS, default=["wasserstein", "js"], format_func=metric_label)
                bins = st.number_input("Histogram bins", min_value=2, max_value=4096, value=64)
                bounds = st.text_input("Fixed bounds (optional, required for score)", placeholder="-1000, 3000", help="Must include all voxels. No clipping or normalization.")
                voxel_limit = st.number_input("Maximum pooled voxels per domain (millions)", min_value=1, max_value=100, value=5,
                                              help="Memory safety limit, not a sampling size. Raising it may require several GB of RAM for exact sorting and histogram calculations.")
                submit = st.form_submit_button("Evaluate cohorts", type="primary")
            st.caption("Top-level files only. Exact calculation, no random sampling. The default 5-million-voxel limit can be raised if memory allows; exceeding it stops evaluation rather than sampling the data.")
            if submit:
                value_range = tuple(float(v.strip()) for v in bounds.split(",")) if bounds.strip() else None
                if value_range is not None and len(value_range) != 2:
                    raise ValueError("Provide two intensity bounds separated by a comma.")
                with st.spinner("Comparing intensity distributions…"):
                    report = evaluate_cohorts(data_root, real, synthetic, metrics, bins=bins, value_range=value_range,
                                              max_voxels=int(voxel_limit)*1_000_000)
                    st.session_state["advanced_completed"] = save_report(report, {"metrics": metrics, "bins": bins,
                          "value_range": value_range, "real": real, "synthetic": synthetic, "weighting": "pooled voxels",
                          "max_voxels": int(voxel_limit)*1_000_000}, output_root)
        if "advanced_completed" in st.session_state:
            report, destination = st.session_state["advanced_completed"]
            st.subheader("Last completed cohort / feature evaluation")
            st.json(report, expanded=False)
            if (destination / "metrics.png").exists():
                st.image(str(destination / "metrics.png"))
            with st.expander("Download results", expanded=True):
                downloads(destination)
    except (ValueError, TypeError, OSError, ImportError, MemoryError) as exc:
        st.error(f"Operation could not finish: {exc}")


def _reports(output_root):
    st.subheader("Reports and epoch history")
    st.caption("Open a result JSON saved by this package to create reports or record it at a training epoch, without recalculating metrics. Do not update the same history file from two processes at once. If you change the evaluation settings or preprocessing, start a new run name.")
    source = st.file_uploader("Result JSON", type=["json"], help="Choose a saved results.json, not a history file. Identifiers in reports are preserved.")
    if source is None:
        st.info("Select a result file to export it or append it to an epoch history.")
    else:
        report = json.loads(source.getvalue())
        if not isinstance(report, dict):
            raise ValueError("Result JSON must contain an object.")
        with st.form("report_form"):
            plot_metrics = st.text_input("Fields to plot (comma-separated, optional)", help="Use exported metric identifiers, e.g. mae, ssim, similarity_score.value. Empty selects up to six fields.")
            export = st.form_submit_button("Export JSON, CSV, PDF, LaTeX and plots")
        if export:
            _, destination = save_report(report, {"plot_metrics": [v.strip() for v in plot_metrics.split(",") if v.strip()]}, output_root)
            st.session_state["converted_report"] = destination
        with st.form("history_form"):
            filename = st.text_input("History filename", value="history.json", help="Relative to the results folder. An existing history is appended, never replaced by a new series.")
            run = st.text_input("Run name", value="validation")
            step = st.number_input("Epoch / step", min_value=0, value=0)
            protocol = st.text_area("Evaluation protocol (JSON object)", value="{}", help='Record the settings that should remain the same across epochs, for example {"preprocessing": "shared normalization", "dataset": "validation set"}. Score settings are saved automatically. Enter the epoch separately above.')
            append = st.form_submit_button("Append this result to history")
        if append:
            path = (output_root / filename).resolve()
            if not path.is_relative_to(output_root.resolve()):
                raise ValueError("History must stay inside the results folder.")
            append_history(path, report, step=step, run=run, protocol=json.loads(protocol))
            st.success(f"Recorded step {step} in {path.name}.")
            st.download_button("Download updated history", path.read_bytes(), file_name=path.name, on_click="ignore")
    if "converted_report" in st.session_state:
        with st.expander("Converted report downloads", expanded=True):
            downloads(st.session_state["converted_report"], key="converted")
    history_file = st.file_uploader("History JSON to plot", type=["json"], help="Select a history created by append_history or the CLI --history option.")
    if history_file is not None:
        with st.form("history_plot_form"):
            fields = st.text_input("History fields (comma-separated, optional)")
            draw = st.form_submit_button("Plot history")
        if draw:
            history = json.loads(history_file.getvalue())
            destination = output_root / f"history_plot_{uuid.uuid4().hex[:10]}"
            destination.mkdir(parents=True, exist_ok=False)
            fig = plot_history(history, metrics=[v.strip() for v in fields.split(",") if v.strip()] or None,
                               output=destination / "history.png")
            for suffix in ("svg", "pdf"):
                fig.savefig(destination / f"history.{suffix}")
            fig.clear()
            st.session_state["history_plot"] = destination
        if "history_plot" in st.session_state:
            destination = st.session_state["history_plot"]
            st.image(str(destination / "history.png"))
            downloads(destination, key="history_plot")
