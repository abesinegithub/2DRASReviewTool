# Changelog

## v1.6
- Fixed **All 2D Flow Areas** georeferenced map aggregation. The combined map now rebuilds one final Leaflet layer control after all area layers are attached, so later 2D areas are not omitted.
- Combined maps now include geometry-only perimeters for 2D areas that exist in only one selected plan.
- Added **automatic large-model optimized comparison mode** for large plan HDF pairs.
- Large-model mode avoids loading full 2D Water Surface/Depth time-series arrays during the initial comparison.
- Face-level velocity matching, same-time WSEL, and changed-mesh normalization are deferred to selected-area review in large-model mode.
- Added lightweight 2D cell reads and lightweight maximum-depth handling to reduce memory use.
- Same-order meshes use direct NumPy array differences rather than Python coordinate dictionaries.
- Very large changed meshes defer exact common-center tuple matching to detailed spatial/raster review to avoid excessive RAM use.
- Georeferenced and Spatial Review tabs now expose detailed-analysis controls so reviewers can keep all-area/large-model views lightweight.

## v1.5
- Fixed Streamlit/PyArrow dataframe serialization warnings/errors when comparison table cells contain nested HEC-RAS values such as lists, dictionaries, tuples, NumPy arrays, or mixed object types.
- Added a global Arrow-safe table renderer used by every detailed comparison dataframe.
- Nested Existing/Revised setting values are displayed as readable compact JSON/text instead of being passed to Arrow as Python objects.
- Numeric and datetime columns remain native where possible; normalization is display-only and does not modify comparison/report data.

## v1.4
- Replaced point/sampled HDF result visualization with **true continuous raster surfaces** created from actual HEC-RAS 2D cell polygons.
- Maximum WSE and Maximum Depth are rasterized cell-by-cell.
- Revised - Existing maps rasterize each plan independently onto one common model-space grid and subtract pixel-by-pixel, supporting changed meshes spatially.
- Maximum Face Velocity is converted to a continuous display surface by assigning each cell the maximum of its bounding face Maximum Face Velocity values; raw face values remain authoritative.
- Added **All 2D Flow Areas** option to the HDF Raster Map.
- WSE surfaces suppress dry cells; Delta WSE is limited to overlap where both plan rasters contain wetted results.
- Delta Depth preserves wetting/drying differences but hides both-dry areas.

## v1.3
- Added **Raster source** selector with two modes: **HEC-RAS Plan Results** and **Stored / External Raster Files**.
- Raster Map can now display **Maximum WSE**, **Maximum Depth**, and **Maximum Face Velocity** directly from selected plan HDF results without requiring GeoTIFF export from RAS Mapper.
- Added raster-like browser visualization generated from **cell-centered** and **face-based** HDF results, including **Existing**, **Revised**, and **Revised - Existing** display modes.
- Delta HDF maps use **exact common cells/faces only** so rebuilt meshes do not misstate differences.
- Improved external-raster classification so terrain / land classification / impervious rasters are not all labeled "Other".

# v1.2

- Fixed comparison crash when a plan HDF does not contain 1D `Summary Output/Cross Sections/Maximum Water Surface`.
- 1D Maximum WSEL is now treated as optional output. The rest of the review continues when it is absent.
- Hydraulic Results and Word report explicitly show 1D result availability and the reason a comparison was skipped.
- Added regression coverage for HDFs with cross-section geometry but no saved 1D Maximum WSEL summary dataset.

# v1.1

- Added **All 2D Flow Areas** combined display to the georeferenced reviewer map while retaining individual-area review.
- Added combined/individual selection to the interactive 2D Spatial Review tab.
- Expanded georeferenced map fit bounds with context padding and capped automatic zoom-in.
- Disabled world wrapping on vector-map basemap tiles.
- Changed raster-map initial extent to the valid-data footprint when possible instead of the full raster rectangle.
- Disabled raster basemap wrapping and added transformed WGS84 extent validation.
- Increased detailed-comparison tab font size, weight, spacing, active-state visibility, and wrapping.
- Fixed `KeyError: object 'Depth' doesn't exist` when a HEC-RAS plan HDF does not contain native 2D Depth time-series output.
- Added optional HDF dataset readers and a labelled depth fallback using `Water Surface - Cell Minimum Elevation` when native Depth is unavailable.
- Added graceful handling when Maximum Face Velocity or Water Surface time-series output is unavailable.
- Added depth-source disclosure in the Hydraulic Results UI and Word review report.
- Expanded automated regression suite to **19 passing tests**.

---

# v1.0

- Made **Local extracted model + dependencies** the recommended source mode for MAAPnext/FEMA-scale data.
- Added optional local Windows native folder-picker buttons for the RAS model and dependency folders.
- Added unlimited dependency search-folder input.
- Added deterministic local dependency resolution statuses: exact, resolved dependency, missing, missing absolute, ambiguous.
- Added a protected HEC-RAS run workspace that copies only the RAS model directory.
- Added Windows directory junctions / directory-symlink fallback for large external dependency folders; dependency data are not duplicated.
- Added enough sandbox parent depth to preserve leading `..\\` relative references during protected runs.
- Added final protected-workspace preflight before `Ras.exe` launch.
- Automatically switches review to the protected working copy after a run so new HDF outputs are detected.
- Blocks unsafe direct runs when dependencies resolve only through auxiliary search folders.
- Expanded automated suite to 16 passing tests.

---

# v0.9

- Added Local Workspace mode for 6+ GB to 150+ GB MAAPnext-scale datasets; no browser upload/copy required.
- Added Local Archive mode for `.zip` and `.7z`; `.7z` extraction uses installed 7-Zip/`7z.exe`.
- Added best-effort archive uncompressed-size and free-disk-space checks.
- Added fast runnable-project discovery so analysis can focus on the HEC-RAS project folder rather than indexing huge Terrain/HMS repositories.
- HEC-RAS runner/preflight now accepts the explicitly selected `.prj` when a workspace contains multiple runnable projects.
- Replaced dismissible `st.dialog` runner with a persistent full-width run panel.
- Added georeferenced Raster Map tab for `.tif/.tiff/.vrt` WSE/depth/velocity/difference outputs with browser-only downsampling.
- Browser upload remains capped at 2 GB by design; large models should use local source modes.
- Automated validation suite increased to 14 tests.

# v0.8

- Fixed HEC-RAS CLI launch when project/plan paths contain no spaces.
- HEC-RAS requires project and plan file parameters to be explicitly double-quoted; Python's normal Windows list-to-command-line conversion can omit those quotes.
- Runner now builds and passes the raw command line: `"Ras.exe" -c "project.prj" "plan.p##"`.
- Run logs now record the exact quoted command line used for CreateProcess.
- Added a regression test that requires quotes around both project and plan parameters.

# v0.7

- Keeps the complete extracted ZIP sandbox as the workspace root instead of collapsing a wrapper folder.
- Resolves dependencies from the actual discovered HEC-RAS project/flow/RAS Mapper locations.
- Adds full-workspace fallback search for missing external DSS, terrain, and projection references.
- Auto-stages a uniquely relocated dependency into the exact path HEC-RAS expects in the temporary run workspace.
- Clones the matching dependency folder when possible to retain companion files.
- Blocks ambiguous duplicate matches rather than guessing.
- Adds Exact / Auto-staged / Ambiguous / Missing preflight statuses and clearer diagnostics.
- Exports updated workspace ZIP contents without a random temporary-folder wrapper.
- Automated suite expanded to 12 passing tests.

# v0.6
- Added workspace-aware upload handling for one nested workspace ZIP or multiple component ZIPs.
- Preserves sibling Model/Terrain/HEC-HMS folder relationships during HEC-RAS runs.
- Recursively discovers runnable HEC-RAS projects while ignoring projection .prj files without sibling plans.
- Resolves plan/HDF/flow files relative to the selected HEC-RAS project directory to avoid basename collisions.
- Added run dependency preflight for external DSS, RAS Mapper terrain, and projection references.
- Missing external dependencies now stop the run before HEC-RAS is launched and show the resolved missing paths.

# Changelog

## v0.5

- Added selected-plan HDF output availability checks.
- Added **RUN HEC-RAS** Streamlit modal.
- Added local HEC-RAS installation/version discovery and custom `Ras.exe` path.
- Added safe extracted working-copy workflow; original uploaded ZIP remains unchanged.
- Added HEC-RAS command-line run support (`Ras.exe -c project.prj plan.p##`).
- Added live elapsed-time/output-HDF monitoring and per-run log files.
- Added export of updated working model as a new ZIP.
- Added directory-based model source support in addition to ZIP input.
- Added interactive 1D longitudinal profiles for Max WSEL, Max Channel Velocity, Max Flow and other common summary datasets.
- Added 1D cross-section time-series Existing/Revised/difference plots.
- Added downloadable Word (`.docx`) review-summary report, including optional run history.
- Expanded automated tests to 8 passing tests.

## v0.4

- Added reviewer screening flags and compact impact summary.
- Added georeferenced Folium/Leaflet reviewer map.
- Added detailed 1D cross-section semantic comparison.
- Added map HTML export and CRS validation.

## v0.2

- Added direct and spatially normalized 2D hydraulic comparisons.
- Added wet/dry classification, depth, face velocity, and synchronized WSEL diagnostics.

## v0.1

- Initial read-only HEC-RAS ZIP/project/plan inventory and comparison engine.