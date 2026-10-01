"""Local validation workspace; all numerical definitions stay in the core package."""
import os
import tempfile
from pathlib import Path
import streamlit as st
from apps.local_validation.service import catalogue, make_demo, plan_pairs, run_validation, inside
from apps.local_validation.metric_labels import metric_label
from apps.local_validation.viewer_ui import render_viewer
from apps.local_validation.workspace import workspace_controls
from apps.local_validation.settings_ui import evaluation_settings
from apps.local_validation.advanced import observed_range
from apps.local_validation.advanced_ui import render_advanced

st.set_page_config(page_title="Synthetic Imaging Validation", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1200px; padding-top:5rem;}
.hero {padding:2rem 2.4rem; border-radius:18px; color:#f5fbff;
 background:linear-gradient(115deg,#102c46,#136d73); margin-bottom:1.5rem;}
.hero h1 {color:#fff; font-size:2.5rem; line-height:1.2; padding:.5rem 0; margin:0; letter-spacing:-1px;}
.hero p {color:#d6eeed; max-width:780px; font-size:1.05rem;}
.eyebrow {font-size:.75rem; font-weight:700; letter-spacing:2px; color:#9de5d8;}
div[data-testid="stMetric"] {background:rgba(128,128,128,.09); border-radius:12px; padding:1rem;}
@media(max-width:650px){.hero{padding:1.3rem}.hero h1{font-size:1.9rem}}
</style>
<div class="hero"><div class="eyebrow">LOCAL APPLICATION</div>
<h1>Synthetic Imaging Validation</h1>
<p>Compare real and synthetic images, masks or precomputed features.
View your data, select the metrics and download the results.</p></div>
""", unsafe_allow_html=True)

default_data = Path(os.environ.get("SIV_DATA_DIR", "data")).resolve()
default_output = Path(os.environ.get("SIV_OUTPUT_DIR", "outputs/local_app")).resolve()
with st.sidebar:
    st.title("Your workspace")
    st.caption("CPU evaluation · NumPy MS-SSIM · no PyTorch required")
    st.caption("Theme: top-right ⋮ menu → Light, Dark or System.")
data_root, output_root = workspace_controls(default_data, default_output)
with st.sidebar:
    st.markdown("**Data folder**")
    st.code(str(data_root), language=None)
    st.markdown("**Results folder**")
    st.code(str(output_root), language=None)
    st.info("Local research tool, not a diagnostic device. Reports can contain case IDs and paths.")
    refresh = st.button("Refresh file list", help="Refresh after adding or moving files. Image contents are never cached globally.")

workflow = st.selectbox("Workspace", ["Paired images and masks", "Independent intensity cohorts", "Feature distributions", "Reports and history"],
                       help="Choose paired images when each synthetic image has an aligned real reference. Choose independent cohorts when the datasets have no one-to-one pairing. Feature comparisons use vectors you have already extracted. Reports and history works with saved results.")
if workflow != "Paired images and masks":
    render_advanced(workflow, data_root, output_root)
    st.stop()

kind = st.radio("What are you evaluating?", ["Images", "Binary masks"], horizontal=True,
                help="Choose Images for intensity comparisons, or Binary masks for overlap and shape. For masks with several labels, prepare a binary mask for each label. Grouping a manifest by label groups cases, not voxels.")
mode = st.radio("Input mode", ["Try example data", "Two files", "Two folders", "CSV manifest"], horizontal=True)
is_mask = kind == "Binary masks"
st.subheader("1 · Select your data")
images, manifests, directories = [], [], ["."]
if mode != "Try example data":
    try:
        cache = st.session_state.get("_catalogue")
        if refresh or cache is None or cache[0] != str(data_root):
            cache = (str(data_root), catalogue(data_root))
            st.session_state["_catalogue"] = cache
        images, manifests, directories = cache[1]
    except (OSError, ValueError) as exc:
        st.error(str(exc))
real = synthetic = manifest = group_by = ""
recursive = False
pairing_options = {}
if mode == "Try example data":
    st.info("Try the app with four small artificial 2D images, divided into two example groups. These are demonstration data, not patient scans.")
    group_by = "label"
elif mode == "Two files":
    left, right = st.columns(2)
    real = left.selectbox("Real file", images or [""], key="real_file")
    synthetic = right.selectbox("Synthetic file", images or [""], key="synthetic_file")
    if not images:
        st.warning("No supported files found. Connect .nii, .nii.gz, .npy or .npz files, then refresh the file list.")
elif mode == "Two folders":
    left, right = st.columns(2)
    real = left.selectbox("Real folder", directories, index=directories.index("real") if "real" in directories else 0)
    synthetic = right.selectbox("Synthetic folder", directories, index=directories.index("synthetic") if "synthetic" in directories else 0)
    recursive = st.checkbox("Include subfolders")
    pairing_options["pairing"] = st.selectbox("Pair matching", ["stem", "sorted"], help="Stem matches filenames without extensions. Sorted pairs alphabetically: use only if the order is known to correspond.")
    if pairing_options["pairing"] == "sorted":
        st.warning("Alphabetical order is not evidence of anatomical correspondence. Check the pairing preview.")
else:
    manifest = st.selectbox("Manifest file", manifests or [""])
    group_by = st.text_input("Group results by column (optional)", placeholder="label", help="Sample categories such as diagnosis or site, not voxel labels. Blank means no grouped summary.")
    with st.expander("Manifest columns and path base"):
        pairing_options["real_column"] = st.text_input("Real path column", value="real")
        pairing_options["synthetic_column"] = st.text_input("Synthetic path column", value="synthetic")
        pairing_options["key_column"] = st.text_input("Case ID column", value="case_id")
        pairing_options["base_dir"] = st.text_input("Path base override", help="Blank: CSV folder. Otherwise a folder inside the data root.")

st.subheader("2 · Choose your evaluation")
if not is_mask:
    st.caption("Use similarity metrics for images showing the same anatomy in the same position. Distribution metrics compare intensity values, but cannot tell whether structures are in the right place.")
options = evaluation_settings(is_mask)
render_viewer(data_root, mode, real=real, synthetic=synthetic, manifest=manifest, recursive=recursive,
              masks=is_mask, channel_text=str(options["channel_axis"]) if options["channel_axis"] is not None else "",
              threshold=options["threshold"], pairing_options=pairing_options)

st.subheader("3 · Review and run")
left, middle, right = st.columns(3)
preview = left.button("Preview pairing", width="stretch")
inspect = middle.button("Inspect intensity bounds", width="stretch", help="Reads all selected images, one at a time. Reports observed bounds; does not alter settings or inputs.")
execute = right.button("Run validation", type="primary", width="stretch")
if preview or inspect or execute:
    with tempfile.TemporaryDirectory(prefix="siv-example-") as temporary:
        try:
            root = data_root
            selected_mode = {"Two files": "files", "Two folders": "directories", "CSV manifest": "manifest"}.get(mode, "manifest")
            if mode == "Try example data":
                root = Path(temporary)
                make_demo(root, "masks" if is_mask else "images")
                manifest = "pairs.csv"
            pairs = plan_pairs(root, selected_mode, real=real, synthetic=synthetic, manifest=manifest,
                               recursive=recursive, group_by=group_by.strip(), **pairing_options)
            if preview:
                st.dataframe([{"Case": p.key, "Real": p.real.name, "Synthetic": p.synthetic.name,
                               "Group": p.metadata.get(group_by, "")} for p in pairs], width="stretch")
                st.caption(f"{len(pairs)} pairs. Full array and geometry checks run during evaluation.")
            if inspect:
                with st.spinner("Scanning full-resolution intensity bounds…"):
                    low, high, rows = observed_range(pairs)
                st.info(f"The selected files contain values from {low!r} to {high!r}. Set score bounds that include these values, then Apply evaluation settings. Use the same bounds for every patient and run you want to compare, rather than choosing new ones each time.")
                st.dataframe(rows, width="stretch")
            if execute:
                options.update(group_by=group_by.strip(), pairing=pairing_options)
                if options.get("pdf_font"):
                    options["pdf_font"] = str(inside(data_root, options["pdf_font"]))
                st.session_state.pop("completed", None)
                bar = st.progress(0.0, text="Checking inputs…")
                with st.spinner("Calculating metrics and preparing reports…"):
                    report, destination = run_validation(pairs, options, output_root,
                        progress=lambda done, total: bar.progress(done / total, text=f"Evaluated {done} / {total} pairs"))
                st.session_state["completed"] = (report, str(destination))
        except (ValueError, OSError, TypeError, ImportError, MemoryError) as exc:
            st.error(f"Validation could not finish: {exc}")
            if "value_range" in str(exc) or "bounds" in str(exc):
                st.info("The score's declared intensity interval does not fit these files. Click Inspect intensity bounds, update the lower and upper score bounds, then Apply evaluation settings. The app has not changed or clipped any input values.")
        except SystemExit:
            st.error("Invalid evaluation settings. Check the selected metrics and parameters.")

if "completed" in st.session_state:
    report, saved_path = st.session_state["completed"]
    destination = Path(saved_path)
    st.divider()
    st.subheader("Results · last completed evaluation")
    st.success(f"Saved {destination.name}. Changing controls does not change these results until you run again.")
    stats = report["summary"]["metrics"]
    columns = st.columns(3)
    columns[0].metric("Evaluated pairs", report["summary"]["count"])
    columns[1].metric("Metric fields", len(stats))
    columns[2].metric("Groups", len(report.get("grouped_summary", {}).get("groups", {})))
    summary_tab, case_tab, group_tab, plot_tab, raw_tab = st.tabs(["Summary", "Per case", "By label", "Plot", "JSON"])
    with summary_tab:
        st.dataframe([{"Metric": metric_label(name), **values} for name, values in stats.items()], width="stretch")
        st.caption("Finite per-case values have equal weight. Counts may differ; std is not a confidence interval. Vector descriptors remain in per-case results, not scalar summaries.")
    with case_tab:
        from synthetic_imaging_validation.cli.validate import _flatten
        st.dataframe([{"Case": p["key"], "Metric": metric_label(name), "Value": str(value)}
                      for p in report["pairs"] for name, value in _flatten(p["metrics"])], width="stretch")
    with group_tab:
        groups = report.get("grouped_summary", {}).get("groups", {})
        if groups:
            st.dataframe([{"Group": group, "Metric": metric_label(metric), **values}
                          for group, summary in groups.items() for metric, values in summary["metrics"].items()], width="stretch")
        else:
            st.info("Choose a manifest group column to obtain label summaries.")
    with plot_tab:
        if (destination / "metrics.png").exists():
            st.image(str(destination / "metrics.png"), caption="Separate panels for metrics with different units.")
        else:
            st.info("No scalar fields selected for plotting. Full vector results are available in JSON/CSV.")
    with raw_tab:
        st.json(report, expanded=False)
    st.markdown("**Download results**")
    downloads = st.columns(6)
    for column, (label, filename, mime) in zip(downloads, [
        ("All (ZIP)", "results_bundle.zip", "application/zip"), ("JSON", "results.json", "application/json"),
        ("CSV", "results.csv", "text/csv"), ("PDF", "results.pdf", "application/pdf"),
        ("LaTeX", "results.tex", "text/plain"), ("Plot", "metrics.png", "image/png"),
    ]):
        if (destination / filename).exists():
            column.download_button(label, (destination / filename).read_bytes(), file_name=filename, mime=mime,
                                   key=f"download_{filename}", width="stretch", on_click="ignore")
    st.caption("ZIP includes settings and PNG/SVG/PDF plots, not input images. Use Reports and history to record this result at an epoch without recalculating metrics.")
