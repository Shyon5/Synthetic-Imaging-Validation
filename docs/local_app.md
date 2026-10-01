# Local application with Docker

The optional browser app runs the existing validation package on your own
computer. It does not upload images to a project server, require an account or
use a cloud service. Docker provides Python and the dependencies; the browser
provides the interface. Internet access is needed to install Docker and build
the image, but not to evaluate local data once the image is available.

The app supports paired 2D/3D images and binary masks, experimental scores,
manifest labels, independent intensity cohorts, precomputed feature matrices,
epoch histories, parallel pair evaluation and JSON/CSV/PDF/LaTeX downloads.
It is a local, single-user research application, not a diagnostic
viewer or a hosted multi-user service.

## 1. Install Docker

### Windows

1. Check the supported Windows release, hardware virtualization and memory
   requirements on the [official Docker Desktop installation page](https://docs.docker.com/desktop/setup/install/windows-install/).
   Institutional computers may require help from IT.
2. Open PowerShell and check WSL:

   ```powershell
   wsl --version
   ```

   If WSL is missing, open PowerShell as administrator and run:

   ```powershell
   wsl --install
   ```

   Restart Windows if prompted. If WSL is already installed but needs updating,
   use `wsl --update`. See Docker's [WSL 2 guide](https://docs.docker.com/desktop/features/wsl/).
3. Download Docker Desktop from the official installation page. Run the installer
   and select the WSL 2 backend when offered. This application uses **Linux
   containers**, not Windows containers.
4. Open Docker Desktop and wait for the engine to start. Review the Docker
   Desktop licence with your organisation; the terms depend on the organisation
   and use. Do not assume that every project partner is covered by a free plan.
5. Open a new terminal and check:

   ```text
   docker version
   docker compose version
   docker run --rm hello-world
   ```

The last command downloads a small test image. `docker version` should show both
client and server information. A server connection error usually means Docker
Desktop has not started or its engine is unavailable.

### macOS and Linux

On macOS install [Docker Desktop for the correct processor](https://docs.docker.com/desktop/setup/install/mac-install/)
(Apple silicon or Intel), then open it. On Linux use [Docker Engine](https://docs.docker.com/engine/install/)
and the [Compose plugin](https://docs.docker.com/compose/install/linux/) for your
distribution, or Docker Desktop. Follow your institution's policy for Docker
permissions. The app image is Linux-based; an Ubuntu image build is exercised in
CI. Native ARM builds are not separately tested yet.

## 2. Start the app

Open a terminal in the repository folder, where `compose.yaml` is located.
Create the two local mount folders before starting. In PowerShell:

```powershell
New-Item -ItemType Directory -Force data, outputs/local_app
docker compose up --build -d
```

On Linux/macOS:

```bash
mkdir -p data outputs/local_app
SIV_UID=$(id -u) SIV_GID=$(id -g) docker compose up --build -d
```

Linux users should keep these UID/GID values for subsequent Compose commands,
for example by setting them in their local `.env` file. This lets the non-root
container write into the result directory owned by that user.

The first build downloads dependencies and can take several minutes. There is
no need to install Python, PyTorch or a CUDA toolkit on the host.

Open **http://localhost:8501**. Select **Try example data**, then **Preview pairing**
and **Run validation**. Four small 2D phantoms are created temporarily; they are
not patient images and do not represent a clinical benchmark. Switch to
**Binary masks** to try the mask metrics. Results persist in `outputs/local_app`.

Useful commands:

```text
docker compose ps
docker compose logs --tail=100 validation
docker compose stop
docker compose start
```

Use `docker compose down` to remove the container; results in the host folder
remain. After updating the repository, run `docker compose up --build -d` again.
No image is published to a registry by the provided workflow.

## 3. Connect your data

Either copy a study into the repository's ignored `data` folder or create a local
`.env` file beside `compose.yaml` to mount an existing folder without copying it:

```dotenv
SIV_DATA_DIR=C:/validation-study
SIV_OUTPUT_DIR=C:/validation-results
```

Use your own existing directories. Forward slashes work well for Windows paths.
For Linux/macOS use absolute paths such as `/home/user/validation-study`. The
`.env` file is ignored by Git and excluded from the Docker build context.
Recreate the container after changing mounts:

```text
docker compose up -d --force-recreate
```

Only the selected data folder appears inside the application as `/data`.
It is mounted read-only. The result folder appears as `/output` and is writable.
Do not mount a home directory or an entire drive. Folder discovery is limited
to 10,000 entries; mount a smaller study if that limit is reached.

**Change folders** in the sidebar switches to a subfolder of either mount.
The data folder must exist; a new results subfolder is created on the first
export. Outside Docker, these fields accept paths accessible to the process
running the app. This is a browser interface, not a native folder picker.

Docker cannot see arbitrary host folders entered in a browser field. To connect
a different host folder, use **Prepare mount configuration** in the sidebar,
download `.env` beside `compose.yaml` and run the recreate command above. Review
an existing `.env` before replacing it: preserve Linux UID/GID settings. The app
does not mount drives itself or need access to the Docker socket.

After adding files, click **Refresh file list**. The app keeps the file catalogue
in the browser session so ordinary selections do not rescan the whole study.

The UI offers:

- **Two files**: pick one real and one synthetic file from the folder.
- **Two folders**: match equal filename stems, optionally including subfolders.
  Missing or duplicate matches raise an error. Expert **sorted** matching is
  available, but order alone does not establish correspondence: inspect the
  pairing before running.
- **CSV manifest**: specify each pair and, optionally, its label.

A convenient layout is:

```text
validation-study/
    real/case_01.nii.gz
    real/case_02.nii.gz
    synthetic/case_01.nii.gz
    synthetic/case_02.nii.gz
    pairs.csv
```

```csv
case_id,real,synthetic,label
case_01,real/case_01.nii.gz,synthetic/case_01.nii.gz,group_a
case_02,real/case_02.nii.gz,synthetic/case_02.nii.gz,group_b
```

Path columns default to `real` and `synthetic`; use **Manifest columns and path
base** for other names or an explicit path base inside the data folder. `case_id`
is optional but supplied IDs must be unique. Enter `label` in **Group results
by column** to produce grouped summaries; leave it blank to ignore labels for
grouping. Relative paths are resolved from the manifest's folder. All referenced
files must remain inside `/data`, including resolved symbolic links. Host paths
such as `C:/...` are not paths inside the Linux container: use relative paths in
portable manifests.

## 4. Inspect images and choose a theme

Turn on **Open image viewer** and choose **Case to inspect**. The viewer loads
only the selected pair and does not run the metrics. Turning it off releases
the preview cache for that browser session. Selecting another pair replaces the
cached preview; data is not shared through a global image cache. Loading a new
pair still takes time, but plane, slice and display-window changes run in a
local HTML canvas without a Python request. Case/channel selection reruns only
the viewer fragment, not the evaluation and result panels.

**Preview resolution** offers **Up to 128 per axis**, **Up to 256 per axis** and
**Original (all voxels)**. The numbers are upper limits, not target sizes. If
both images fit within the selected limit, nothing is skipped: two 192³ volumes
remain 192³ with Up to 256, rather than being resized to 256³.

For larger images, the viewer keeps regularly spaced pixels/voxels along each
axis. The same step is used for both images and is rounded up to meet the limit.
For example, two 192³ volumes with Up to 128 use every second voxel and appear
as 96³ previews. The actual preview shapes and step are shown below the images.
Small structures can disappear in a reduced preview; it is not a diagnostic view.

**Original** keeps every pixel/voxel of both images. It can take longer to load
and use much more browser memory. The app checks the preview size before encoding
it: previews above **128 MiB of combined uncompressed float32 voxel data** are
refused with a message asking you to select a smaller preview. Browser decoding
needs additional copies, so actual RAM use is higher. The app never silently
reduces an Original preview to get around this limit.

Original means the original spatial grid, not bit-for-bit file storage: display
values still use float32, and NIfTI axes may be reordered as described below.
Mask thresholds are applied before float32 conversion to preserve the foreground
decision. Metrics always use the original arrays at full resolution,
regardless of the preview option. Slice indices refer to the reordered display
array and advance by the preview step.
Use a recent Edge, Chrome, Firefox or Safari with `DecompressionStream` support;
there are no external JavaScript, image or model downloads.

- **2D images:** real and synthetic are shown side by side.
- **3D volumes:** choose a plane and move the synchronized slice slider.
- **Intensity images:** set shared display minimum/maximum and choose grayscale
  or Magma. An optional Magma third panel shows absolute intensity difference.
- **Binary masks:** the preview uses the evaluation threshold, including values
  equal to the threshold. Cyan means real foreground, amber synthetic foreground,
  white overlap and dark background. Individual panels and the overlap panel
  share this colour convention.
- **Channels/frames:** set the channel axis in the evaluation settings and choose
  a zero-based preview channel/frame. For a 4D NIfTI the viewer requires axis 3
  (or -1), preserving the three spatial axes. Selecting a frame here does not
  select that frame for metric computation.

NIfTI volumes are reordered/flipped to closest RAS for display only, using their
affine metadata. The convention is neurological: R appears on the right in
axial/coronal views. Direction labels show increasing axes. Oblique scans retain
their obliquity: views are nearest voxel planes, not resampled anatomical planes.
NumPy arrays and 2D views use array-axis labels rather than invented anatomical
directions. This is a slice viewer, not a diagnostic workstation or a 3D surface
renderer.

Displayed proportions use file spacing. NumPy files use unit spacing in the
viewer; an explicit spacing override for metrics does not change the preview.
The **Image details** expander shows shape, display spacing, original orientation
where available, selected channel and intensity bounds.

When displayed grids differ, the app warns and disables difference/overlap.
Side-by-side slices then share an index, not necessarily a physical location.
Matching geometry still does not establish anatomical correspondence. Preview
reorientation never changes the original files or bypasses validation checks.

**Display windows are not metric intensity ranges.** Windowing changes only how
pixels look; it does not clip, normalize, register or resample metric inputs.
The difference panel uses a scale from zero to the display-window width; larger
differences can saturate visually. Stored results remain unchanged.

Use the **top-right three-dot menu** to choose **Light**, **Dark** or **System**.
The configured Streamlit version offers these options directly in the menu.
Both themes have matching application palettes; the image panels retain a dark,
neutral background so their appearance does not depend on the surrounding theme.
Switching themes does not change any calculation.

Question-mark icons next to controls open short explanations. The **Metric guide**
popover explains the available metrics, useful direction and limitations. Names
in selectors and on-screen tables are readable labels (for example, “MS-SSIM —
Multi-scale Structural Similarity”). JSON/CSV/PDF/LaTeX field names and saved plot
labels keep their existing identifiers for compatibility.

## 5. Choose metrics and interpret the output

Inputs must already be preprocessed. The app does not register, resample,
normalize, anonymise or silently repair images. NIfTI geometry checks are kept
enabled by default; an advanced override is explicit and recorded in settings.
The pairing preview checks filenames and metadata, not the contents of
every volume; shape and geometry checks happen during calculation.

Choose **Images** or **Binary masks**, edit the metrics and parameters, then
press **Apply evaluation settings**. Edits are sent together rather than causing
a reload at every selection. The active metric list is shown below the form.
**Run validation** uses the last applied settings, or the initial defaults.

For PSNR/SSIM/MS-SSIM enter the known intensity range width. A width of `1` is
appropriate for data prepared in a unit-width interval, not arbitrary CT or PET.
Scores also need explicit lower and upper bounds covering **every input voxel**.
The default `[0, 1]` assumes inputs already in that interval.

### Why can metrics run while a score rejects the range?

In the app, a short explanation is available through the **?** next to
**Experimental scores (optional)**. The detailed explanation is kept here.

Most raw error metrics do not need a declared intensity interval. If a real
value is 100 and a synthetic value is 110, the absolute error is 10 in the
original units. MAE and Wasserstein can report that kind of difference directly.

Scores take an extra step: they turn errors into a common 0–100 index. They
divide MAE or Wasserstein by the declared interval width, and the distribution
score fixes its histogram bins to that interval. Saying that data lie in
`[0, 1]` when they contain values of 100 or 110 would make that setup inconsistent.
The score rejects it rather than hiding the mistake or excluding values from
the histogram. This is a deliberate check, not a requirement to normalize every
dataset to `[0, 1]`.

PSNR, SSIM and MS-SSIM also depend on the intensity range width, but do not run
the score's full bounds check. **A calculation finishing successfully does not
mean its range was appropriate.** Raw JS/KL, without a fixed score interval, use
bins spanning the observed real/synthetic values instead. When scores are
enabled, the CLI and app reuse their component metrics and matching range settings.

If a score reports intensities outside its bounds:

1. Click **Inspect intensity bounds** to scan all selected files at full resolution.
2. Set lower/upper score bounds that include those values and agree with your
   study's preprocessing protocol. For example, only use `[-1000, 3000]` for CT
   if that interval actually covers the prepared inputs; it is not a CT preset.
3. Apply settings and run again. When similarity score is selected, its bound
   width also sets PSNR/SSIM/MS-SSIM range, avoiding conflicting settings.

No automatic clipping or normalization occurs. Do not fit a separate interval
to each patient or epoch: keep a common protocol for comparisons. Display
windowing has no effect on these checks. See [Scores and validation plots](scores_and_plots.md)
for the 0–100 prototype formulas and their limitations.

Spacing overrides follow array-axis order. NIfTI spacing is otherwise read from
the header; array files without a spacing override use unit spacing. A channel
axis is optional for structural image metrics. Prefer one case per file. The
advanced batch axis follows CLI semantics (SSIM/MS-SSIM only; scores reject it),
and does not turn one file into multiple report cases. Binary-mask metrics do not handle multiclass label maps as separate
classes: prepare binary masks explicitly.

Each worker loads one pair. Start with one worker for large 3D images and raise
the count only if memory allows. Progress counts evaluated pairs, not operations
inside an individual metric. Loading, calculation and report export can still
take time when the bar is stationary. There is no reliable in-app cancellation
in this first version; stopping the container interrupts the evaluation.

Every successful run writes a uniquely named directory with:

- `results.json`, `results.csv`, `results.pdf` and `results.tex`.
- `metrics.png`, `metrics.svg`, `metrics.pdf`: separate panels for scalar fields.
  Vector-only results, such as histogram counts, do not produce a scalar plot.
- `settings.json`: the UI evaluation settings.
- `results_bundle.zip`: all the above, without input images.

Downloads and the result tabs refer to the **last completed evaluation**, even
if you have since edited the controls. Changing a control or downloading a file
does not recalculate metrics. The ZIP is not a complete reproducibility archive:
retain your dataset selection, preprocessing and environment version separately.
Demo input paths refer to temporary files that are deleted after the operation.

Summary means exclude non-finite values and give cases equal weight. Inspect
counts as well as means. Infinite PSNR and empty-mask distance conventions remain
visible in per-case results. Scores are experimental, not clinical ratings.

## 6. Additional workspaces and API coverage

The **Workspace** selector separates different statistical questions rather than
mixing paired and unpaired measurements in one form.

| Workflow | Available in the app |
| --- | --- |
| Paired images | All CLI image metrics and scores; intensity statistics, percentiles, shared-bin histograms and the API distribution report |
| Binary masks | All CLI mask metrics, maximum Hausdorff distance, full surface summaries, component measures, border distances, centroids and morphology report |
| Independent intensity cohorts | Pooled-voxel Wasserstein, JS, KL and intensity-distribution score, without pair matching |
| Precomputed features | Fréchet, KID, RBF MMD, feature precision/recall and sliced Wasserstein, optionally grouped by sample labels |
| Reports and history | Export a saved result to JSON/CSV/PDF/LaTeX, plot chosen scalar fields, append epochs to a history and export curves to PNG/SVG/PDF |

Legacy aliases `volume_ratio` and `active_voxel_fraction` are represented by the
dimension-neutral **Foreground Area / Volume Ratio** and **Foreground Fraction**;
the numerical definitions are unchanged. Component measures give area in 2D
and volume in 3D. Extended mask-report connectivity can be configured separately;
the original CLI metrics retain their defaults. Centroid physical coordinates
are spacing-scaled indices, not full NIfTI world coordinates.

**Independent intensity cohorts** reads top-level files from two folders.
Different shapes and sample counts are allowed. All voxels are pooled, so a
larger image has more weight; these results are not an average of patient scores.
The default memory limit is 5 million voxels per domain; the form allows raising
it up to 100 million, but exact sorting and distribution calculations can then
need several GB of RAM. Exceeding the limit stops evaluation; it never silently
samples voxels. Larger studies can use an explicitly prepared subset or the API. A fixed interval, if supplied,
must contain all values; it is mandatory for the distribution score.

**Feature distributions** expects `.npy` or `.npz` matrices shaped `[samples,
features]`. There is no built-in encoder: feature vectors must come from the
same model and preprocessing, with matching feature dimensions. Fréchet on
these vectors is not automatically a medically validated 3D-FID. Most estimates
are unreliable with very small cohorts; report sample counts. KID/MMD can be
negative without being an implementation error. Quadratic metrics are capped
at 5,000 samples per domain in this UI; high-dimensional covariance calculations
can also be expensive.

For per-class feature results provide two CSV files, each with one non-empty
`label` (or selected column) per matrix row, in exactly the same order. The
domains may have different sizes. Groups with fewer than two samples in either
domain are reported as insufficient; precision/recall additionally requires
more samples than the selected neighbour count. In paired workflows use the
manifest group column instead. Neither grouping mechanism segments voxel labels.

**Reports and history** accepts saved package result JSON through a file picker.
Exporting a result does not run metrics again. To record training progression,
enter a non-negative epoch, a run name and a history filename relative to the
results folder. Include stable preprocessing/dataset/metric settings in the
protocol JSON; score protocols are retained automatically. Duplicate epochs or
changed metric sets/protocols within a run are rejected. Use one writer per
history file. Upload the resulting history in the second picker to plot its
curves; choose up to 12 scalar field names when the history contains many fields.

The advanced paired settings expose spacing/channel/batch axes, border widths,
geometry override, plot fields, a PDF font and the MS-SSIM backend. NumPy is the
standard Docker backend. TorchMetrics remains optional: selecting it without
the `torch` extra produces an installation error, not an automatic download.
To use it, run the app in a local environment installed with `.[app,torch]`, or
build your own image with that extra. The standard image stays lightweight.

Arbitrary Python callables, custom feature encoders, tensor objects, preprocessing
helpers and programmatic training-loop callbacks remain API features. The GUI
does not run user-supplied Python or alter source images. Its workspace selectors
cover built-in metric families, not every possible Python composition.

## Privacy, resource use and troubleshooting

Compose binds port 8501 to **127.0.0.1 only**. Keep that setting: the app has no
authentication and is not intended for network access. Telemetry is disabled;
the app uses no external fonts, images or model downloads. Docker and dependency
installation still contact their distribution services during setup. This is
not a sandbox for untrusted files; only evaluate trusted local datasets.

The container runs as a non-root user, with a read-only root filesystem, no
additional Linux capabilities and a writable temporary directory. Inputs are
read-only, but reports still contain paths, labels and identifiers. Review them
before sharing. The build context is allowlisted in `.dockerignore`: local data,
private demos, Git history, `.env` files and existing reports are not copied into
the image.

- **No files listed:** check the mount folder and supported extensions. Files
  elsewhere on the host are intentionally not visible.
- **Permission denied writing results:** on Linux set `SIV_UID` and `SIV_GID`
  to your user/group IDs and use a folder you own. Do not solve this by running
  the app privileged.
- **Port already in use:** stop the other service, or change the host side to
  `127.0.0.1:8502:8501` in Compose and open port 8502.
- **Out of memory / container exits:** reduce workers or the evaluated volume
  size and check Docker's memory allocation. Metric inputs are not downsampled;
  only the viewer creates a lower-resolution preview.
- **A metric fails:** check range, geometry and the named pair. Errors are not
  silently replaced with zeros. Failed exports may leave partial files in that
  run's directory; existing runs are never overwritten.

Use one active evaluation at a time. The app does not implement a multi-user job
queue. Closing a browser tab is not a guaranteed cancellation mechanism.

## Development and verification

The container uses Python 3.12 and a constrained Streamlit release. The core
package retains its existing Python support and does not import Streamlit.
The `app` extra installs UI/reporting dependencies only when requested and needs
Python 3.10 or newer; it is separate from the core's Python 3.9 support. Docker's
scientific dependencies still follow `pyproject.toml`; this is not a fully locked,
bit-for-bit reproducible image. Keep a built image or record its digest for a study.

To run without Docker in a separate Python 3.12 environment:

```text
python -m pip install -c docker/constraints.txt -e ".[app,test]"
python -m streamlit run apps/local_validation/app.py --server.address=127.0.0.1
python -m pytest apps/local_validation/tests
```

Set `SIV_DATA_DIR` and `SIV_OUTPUT_DIR` for that process to change the local
folders. UI tests are separate from the core's 100% coverage gate. They cover
pairing, path boundaries, labels, CLI numerical parity, parallel execution,
exports, custom score ranges, feature/cohort API parity and Streamlit interactions.
Optional Playwright tests verify actual canvas orientation, slice values and mask
colours. Run them with `SIV_BROWSER_TESTS=1` after installing Playwright and its
Chromium browser; on a local Windows machine `SIV_BROWSER_CHANNEL=msedge` can use
an installed Edge. CI installs Chromium explicitly. The dedicated **Local app and Docker** GitHub
Actions workflow also builds the image and smoke-tests evaluation and HTTP health;
it does not publish an image or upload medical data.
