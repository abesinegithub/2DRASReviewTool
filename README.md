HEC-RAS Review Tool v1.6

Key v1.6 improvements:
- Correct combined **All 2D Flow Areas** georeferenced map.
- Automatic large-model comparison mode to reduce MAAPnext memory pressure and Streamlit connection drops.
- Heavy time-series / velocity / changed-mesh normalization is deferred to selected-area review when appropriate.

HEC-RAS Review Tool v1.5

This maintenance release fixes Streamlit/PyArrow table serialization for nested HEC-RAS comparison values.

HEC-RAS Review Tool v1.4

This version adds continuous HEC-RAS plan-result raster review from actual 2D cell polygons alongside stored/external raster support.

# HEC-RAS Review Tool — v1.2

## v1.2: Map Extent, Multi-Area Display, UI Visibility, and Missing-Depth Compatibility

v1.2 focuses on reviewer usability and HEC-RAS output compatibility.

### Reviewer-map improvements

- The **Georeferenced Map** can now display **All 2D Flow Areas** together or any individual area.
- Combined maps retain separate named layers for each 2D area.
- Initial map extent now includes context padding and caps excessive automatic zoom-in.
- Street/aerial/topographic basemaps use no-wrap behavior to avoid repeated world copies.

### Raster-map extent fix

- Raster maps focus on the approximate **valid-data footprint** rather than blindly fitting to the full raster rectangle.
- Base-map world wrapping is disabled.
- WGS84 bounds are validated before display.
- Initial raster zoom includes a small context buffer and a maximum automatic zoom.

### Clearer detailed-review tabs

- Review tabs use larger, heavier text, more padding, and a stronger active-tab indicator.
- Tabs can wrap on narrower reviewer screens instead of becoming difficult to see.

### Missing 2D `Depth` output no longer crashes comparison

Some HEC-RAS plans do not write a time-series dataset literally named `Depth`. v1.0 assumed it was always present and could raise:

```text
KeyError: object 'Depth' doesn't exist
```

v1.2 treats 2D output datasets as optional. Depth handling now uses this priority:

1. native `Depth` time series;
2. native `Maximum Depth` summary, if available;
3. a clearly labelled wet/dry **proxy** derived from `Water Surface - Cell Minimum Elevation`.

The fallback prevents dry-cell elevations from being misreported as WSEL impacts while allowing the overall plan comparison to continue. The hydraulic UI, JSON report, and Word report indicate when a derived depth proxy was used. Optional face-velocity and same-time WSEL outputs are also skipped gracefully when not written.

### Spatial-review improvements

- The **Spatial Review** tab supports combined or individual 2D flow areas.
- Combined WSEL, wet/dry transition, and mesh-change plots aggregate all selected 2D areas.
- Area-specific normalized and same-time diagnostics remain available below the combined plots.

### Validation

The v1.2 regression suite contains **19 passing tests**. In addition, the exact user-reported failure mode was manually reproduced by deleting the `Depth` dataset from the Revised test HDF; both `compare_plans()` and `spatial_area_review()` completed successfully using the labelled fallback.

---

# HEC-RAS Review Tool — v1.0

## v1.0: Local Workspace & Dependency Manager

v1.0 makes **already-extracted local data the primary production workflow** for MAAPnext/FEMA-scale models. Multi-GB and 100+ GB datasets no longer need to be recompressed, uploaded through the browser, or duplicated just so the review tool can see Terrain/HMS/DSS data.

### Recommended large-model workflow

In the Streamlit app choose:

```text
Source mode: Local extracted model + dependencies
```

Then select/paste:

```text
HEC-RAS model folder
D:\MAAPnext\RAS_Model

Dependency folders (one per line)
D:\MAAPnext\Terrain
D:\MAAPnext\HMS_DSS
E:\Regional_LandCover
...
```

The dependency list is intentionally open-ended. Folder names do not have to follow a tool-specific convention. The validator uses the paths actually referenced by the HEC-RAS project/flow/RAS Mapper files and checks those references against the source model and the selected dependency locations.

When Streamlit is running locally on Windows, **Browse model folder** and **Add dependency folder** can open a native Windows folder picker. Text paths remain available for remote/server deployments where a browser cannot reveal arbitrary local paths.

### Dependency validation

The local dependency table distinguishes:

- `exact` — the HEC-RAS relative/absolute path already resolves from the source model;
- `resolved_dependency` — the missing relative path can be satisfied by one selected dependency folder;
- `missing` / `missing_absolute` — no source is available;
- `ambiguous` — more than one selected location satisfies the reference, so the tool refuses to guess.

The resolver uses deterministic path checks rather than recursively crawling a 100+ GB dependency repository.

### Protected HEC-RAS computation (default)

**Protect original RAS model** is enabled by default. When a run is needed, v1.0 creates a lightweight review workspace:

```text
Protected Review Run/
└── _relative_1/
    ├── RAS_Model/      <- copied RAS model only
    ├── Terrain/        -> Windows directory junction to original Terrain
    └── HMS_DSS/        -> Windows directory junction to original HMS/DSS
```

Large dependency folders are **not copied**. On Windows, local dependency directories are exposed with `mklink /J` directory junctions. A directory symlink is attempted for targets that cannot use a junction (for example some network/UNC layouts). The source RAS model and dependency data remain unchanged.

The copied model is placed with enough artificial parent depth to keep leading `..\\` HEC-RAS references inside the protected sandbox. A final dependency preflight is run against the protected project before `Ras.exe` is launched.

After computation, the application automatically switches review/comparison to the protected working copy so newly generated `.p##.hdf` results are immediately available.

### Direct local run

A reviewer may disable protection, but v1.0 only permits a direct run when every selected plan dependency already resolves **exactly** from the original model. A dependency merely found in a separate search folder is not sufficient for a direct run because HEC-RAS itself would not know that search folder. In that situation the app requires protected mode so it can create junctions safely.

### Existing v0.9 capabilities retained

- local `.zip` and `.7z` archive paths;
- smaller browser ZIP/7z uploads;
- persistent full-width HEC-RAS run controls;
- HEC-RAS installation/version selection and quoted command-line compute;
- model-change inventory and reviewer flags;
- 1D longitudinal profiles and time-series comparisons;
- 2D mesh/WSEL/depth/velocity spatial comparison;
- georeferenced vector review map;
- georeferenced WSE/depth/velocity/difference GeoTIFF/VRT raster map;
- Word review-summary report;
- JSON report export.

### Validation

The v1.0 automated suite contains **16 passing tests**. New v1.0 tests verify explicit external dependency resolution, protected RAS-only run-workspace creation, linked dependency visibility, and ambiguity blocking, in addition to the existing model/hydraulic/map/report tests.

---

# HEC-RAS Review Tool — Prototype v0.9

## v0.9 highlights

### 1. MAAPnext-scale local workspace mode

For multi-GB to 100+ GB projects, **do not upload the model through the browser**. Run the Streamlit app on the Windows workstation that already has the model and choose:

- `Local workspace folder (recommended for large models)` — points directly to the common parent folder. No upload, no archive copy, no extraction.
- `Local archive path (.zip/.7z)` — reads a local archive path and extracts it to a chosen disk location. `.7z` uses the installed 7-Zip command-line executable (`7z.exe`). For very large archives, pre-extract with 7-Zip and use Local Workspace mode.
- `Browser upload (.zip/.7z — smaller datasets)` — retained for small test packages.

The application discovers the runnable HEC-RAS `.prj` recursively, then runs analysis against the **RAS project folder only** so large Terrain/HMS repositories do not have to be indexed by every comparison call. The common workspace root is still retained for HEC-RAS dependency resolution.

Example:

```text
ReviewWorkspace/
├── Test1D2D/       <- d1180000.prj and p## files
├── terrain/
└── D118_HMS34/
```

For local-workspace mode, HEC-RAS runs in place against the selected workspace to avoid duplicating 100+ GB datasets. Use a controlled review copy if the original submission must remain immutable.

### 2. Persistent full-width HEC-RAS Run controls

The modal dialog has been removed. Run controls are now a full-width persistent panel. Clicking elsewhere does not dismiss it, fields persist in Streamlit Session State, the page can scroll normally, and run/dependency history remains visible.

### 3. Georeferenced raster-output map

A new `Raster Map` tab discovers `.tif`, `.tiff`, and `.vrt` files below the HEC-RAS project folder and classifies likely WSE, depth, velocity, and difference rasters. The selected raster is:

1. opened with Rasterio;
2. reprojected to EPSG:4326 for web display;
3. downsampled to a reviewer-controlled display pixel count;
4. shown over street/aerial/topographic basemaps with transparency and a legend.

Only the browser visualization is downsampled. Review calculations continue to use the full-resolution HDF/raster files.

### 4. `.7z` support

The app can extract local or uploaded `.7z` files using 7-Zip (`7z.exe`). For large archives this avoids the Streamlit browser upload limit, but extraction still requires enough free disk space for the uncompressed workspace. The app performs a best-effort uncompressed-size/free-space check before local archive extraction.

### Validation

The v0.9 automated suite contains **14 passing tests**, including model inventory/change detection, spatial hydraulics, georeferencing, raster-map rendering, reviewer flags, 1D profiles/time series, large-workspace project discovery, ZIP workspace assembly, dependency preflight/auto-staging, Word report generation, and HEC-RAS command construction.

---

# HEC-RAS Review Tool — Prototype v0.8

Python/Streamlit reviewer application for comparing an Existing/Effective-like HEC-RAS plan with a Revised/Proposed plan.

The application separates two review questions:

1. **What changed in the model?** — plan settings, 1D geometry, 2D mesh, structures, breaklines and boundary conditions.
2. **What hydraulic response changed?** — WSEL, depth, velocity, wet/dry transitions, 1D profiles and time series.

## v0.8 highlights

### v0.8 command-line quoting fix + v0.7 workspace resolver

The recommended submission format is one ZIP of a **common parent workspace** whose children preserve the original sibling-folder layout used by HEC-RAS relative paths, for example:

```text
ReviewWorkspace/
├── Test1D2D/        # HEC-RAS .prj/.p##/.g##/.u## files
├── terrain/         # referenced as ..\terrain\...
└── D118_HMS34/      # referenced as ..\D118_HMS34\...
```

v0.7 keeps the entire extracted archive sandbox as the run workspace, discovers the project recursively, and resolves dependencies relative to the actual discovered project/flow/RAS Mapper file locations. If an exact relative path is missing but exactly one matching file exists elsewhere in the uploaded workspace, the tool can **auto-stage the dependency into the expected location in the temporary working copy**. It clones the containing dependency folder when possible so companion terrain/DSS files are retained. Ambiguous matches are never guessed. The original uploaded ZIP is never modified.

Dependency preflight statuses are: **exact**, **auto_staged**, **ambiguous_relocated**, and **missing**.

### 1. Optional HEC-RAS execution from the review app

The selected plans are checked for plan-result HDF files before comparison. If a selected plan has no `.p##.hdf`, the application reports that no run output was found and disables hydraulic comparison until output is available.

The **RUN HEC-RAS** button opens a modal dialog that allows the reviewer to choose:

- one or more plans to run;
- an automatically discovered local HEC-RAS installation/version;
- a custom `Ras.exe` path when needed;
- a run timeout.

The runner uses the HEC-RAS command-line compute pattern with **explicit quotes around both file parameters**:

```text
"C:\Program Files\HEC\HEC-RAS\x.y\Ras.exe" -c "C:\workspace\model.prj" "C:\workspace\model.p##"
```

HEC-RAS requires the project and plan filenames to be quoted even when their paths contain no spaces. v0.8 passes this as a raw Windows command line so Python does not remove the quotes.

**Safety/workspace design:** the uploaded ZIP is never overwritten. On the first run, v0.7 extracts the complete model into a temporary working directory and runs HEC-RAS there. New HDF results are detected automatically and the application switches to the working copy for subsequent comparison. The reviewer can optionally export the working model (including new outputs) as a new ZIP.

During a run, the UI shows live elapsed time, process state, whether the expected HDF has appeared, and its current size. The run log is preserved under `.hecras_review_logs` in the working copy. HEC-RAS execution requires the Streamlit application to run on a **Windows machine with HEC-RAS installed**.

### 2. 1D hydraulic line plots

The Hydraulic Results tab now includes interactive longitudinal 1D profiles for common cross sections. Available HDF summary variables are discovered from the selected plan outputs. The development model supports:

- Maximum Water Surface / Max WSEL
- Maximum Channel Velocity
- Maximum Flow
- Minimum Water Surface
- Minimum Channel Velocity
- Minimum Flow

For a selected River/Reach, the app plots:

- Existing result line;
- Revised result line;
- `Revised - Existing` difference line.

A second 1D section provides cross-section time-series comparisons when those datasets exist, including Water Surface, Channel Velocity, Total Velocity, Flow and Lateral Flow.

### 3. Word review-summary report

A new **Review Report** tab generates a `.docx` summary containing:

- project and selected plan metadata;
- HEC-RAS plan versions;
- executive screening summary;
- plan/model changes;
- 2D mesh comparison;
- 1D cross-section geometry summary;
- structure and boundary-condition changes;
- 2D WSEL/depth/velocity summaries;
- 1D maximum-WSEL summary and top changes;
- reviewer screening flags;
- HEC-RAS run history when runs were initiated from the application;
- interpretation/limitations statement.

The report explicitly states that reviewer flags are screening aids and not automatic FEMA/local regulatory pass/fail determinations.

## Existing spatial/georeferenced capabilities retained

- georeferenced Folium/Leaflet map;
- model CRS read from HDF WKT and transformed to WGS84;
- street, aerial and topographic basemaps;
- Existing/Revised 2D-area perimeters;
- wet-to-wet `Revised - Existing` maximum WSEL;
- separate became-wet and became-dry layers;
- Existing-only/Revised-only mesh centers;
- 1D cross sections and hydraulic structures;
- revised BC lines and breaklines;
- direct same-cell and polygon-containing-cell changed-mesh comparisons;
- depth and face-velocity comparisons;
- synchronized WSEL diagnostics;
- reviewer screening flags.

For `Test1D2D.zip`, the source CRS resolves to **EPSG:2278 — NAD83 / Texas South Central (ftUS)**.

## Test model configuration

- Existing: `p03` — `full 1d/2d 100yr`
- Revised: `p05` — `1d/2d 100yr proposed`

Known checks include:

- `north2d`: 19,738 exact common cell centers;
- `south2D`: 22,824 Existing cells vs. 23,632 Revised cells;
- `south2D`: 22,685 exact common centers, 139 Existing-only, 947 Revised-only;
- culvert groups: 6 Existing vs. 7 Revised;
- all 169 common 1D cross-section definitions unchanged for the tracked geometry parameters;
- 1 structure changes from 0 to 1 culvert;
- 1 added and 1 removed unsteady boundary location;
- 43,548 geometry-matched `south2D` faces for direct maximum-velocity comparison.

## Install and run

Use Python on the Windows workstation where HEC-RAS is installed if you want to use the **RUN HEC-RAS** feature.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The included `.streamlit/config.toml` permits large prototype uploads.

## Web tabs

- Model Changes
- 2D Mesh
- Georeferenced Map
- Spatial Review
- Structures
- Boundary Conditions
- Hydraulic Results
- Review Report
- Raw Report

## Command-line review

```bash
python scripts/inspect_model.py /path/to/Test1D2D.zip \
  --existing p03 \
  --revised p05 \
  --output review.json
```

The comparison engine accepts either a ZIP file or an extracted model directory.

## Important interpretation notes

- The original uploaded ZIP remains immutable; HEC-RAS computes only in an extracted working copy.
- Prefer the plan's original HEC-RAS version for reproducibility when it is available. Running an older project in a newer HEC-RAS version can be useful for testing but may introduce version-related numerical differences that should be documented.
- Maximum-vs-maximum WSEL and same-time WSEL are intentionally separate metrics.
- WSEL/depth are cell-based in the development HEC-RAS 5.04 HDF; velocity is face-based and is handled separately.
- Dry/wet transitions are separated from wet-to-wet WSEL comparisons.
- The tool is reviewer assistance software, not an automated regulatory approval engine.

## Current validation

The v0.7 automated suite contains 12 tests covering model inventory/change detection, spatial hydraulics, georeferencing, screening flags, 1D profiles/time series, directory-source support, Word report generation, and HEC-RAS command construction.

Actual `Ras.exe` execution cannot be tested on non-Windows development hosts and should be validated on the target Windows workstation with the installed HEC-RAS versions used by the review team.


## v0.6 external-data workspace support

HEC-RAS projects often reference terrain, DSS, or HEC-HMS data outside the RAS project directory. v0.6 treats the upload as a **workspace**, not only a model ZIP.

Recommended packaging: zip the common parent directory while preserving the original sibling-folder names and relative structure, for example:

```text
ReviewWorkspace/
├── D118_Model/
│   ├── project.prj
│   ├── project.p01
│   └── ...
├── terrain/
│   └── TerrainName.hdf
└── D118_HMS34/
    └── hydrographs.dss
```

The uploader also accepts multiple component ZIPs. When using separate ZIPs, name each ZIP after the original folder it represents because the ZIP stem becomes that folder name in the temporary workspace.

Before RUN HEC-RAS, a dependency preflight resolves DSS, RAS Mapper terrain, and projection references and stops the run if referenced files are missing. The original uploaded ZIP(s) remain unchanged.
