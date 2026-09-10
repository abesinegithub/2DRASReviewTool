from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from hecras_review.inspector import (
    compare_plans, inspect_model, spatial_area_review, map_reference_geometry, map_2d_area_perimeters,
    available_1d_results, one_d_profile_review, one_d_timeseries_review,
)
from hecras_review.georef_map import build_review_map, map_html
from hecras_review.raster_map import discover_result_rasters, build_raster_review_map, raster_map_html, build_plan_result_surface_map
from hecras_review.runner import (
    discover_hecras_installations, discover_7zip_executable, discover_runnable_projects,
    prepare_working_copy, run_hecras_plan, zip_working_model, workspace_reference_preflight,
    archive_uncompressed_size, validate_local_workspace, create_protected_run_workspace,
)
from hecras_review.report_docx import build_review_docx
from hecras_review.source import open_model_source

st.set_page_config(page_title="HEC-RAS Review Tool", layout="wide")
st.title("HEC-RAS Review Tool — v1.6")
st.caption("Local workspace review, protected HEC-RAS runs, robust 1D/2D comparison, combined 2D-area mapping, and georeferenced raster review from both stored rasters and direct HEC-RAS plan HDF results")

# Make the detailed-review tabs easier to see and select on wide reviewer screens.
st.markdown(
    """
    <style>
    div[data-baseweb="tab-list"] { gap: 0.35rem; flex-wrap: wrap; }
    button[data-baseweb="tab"] {
        padding: 0.7rem 1rem !important;
        min-height: 3rem !important;
        border-radius: 0.45rem 0.45rem 0 0 !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.08rem !important;
        font-weight: 650 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(128,128,128,0.12) !important;
        border-bottom: 3px solid currentColor !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _reset_review_state() -> None:
    for key in ["report", "report_pair", "run_history", "updated_working_zip"]:
        st.session_state.pop(key, None)


def persist_uploads(uploads) -> list[Path]:
    uploads = list(uploads or [])
    key = "|".join(f"{u.name}:{u.size}" for u in uploads)
    if st.session_state.get("upload_key") == key and st.session_state.get("archive_paths"):
        paths = [Path(x) for x in st.session_state["archive_paths"]]
        if paths and all(p.exists() for p in paths):
            return paths
    paths = []
    for upload in uploads:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hecras_review_upload_"))
        tmp = tmp_dir / upload.name
        tmp.write_bytes(upload.getbuffer())
        paths.append(tmp)
    st.session_state["upload_key"] = key
    st.session_state["archive_paths"] = [str(x) for x in paths]
    _reset_review_state()
    return paths




def _materialize_selected_plan_hdfs(model_path: str | Path, existing_plan: str, revised_plan: str):
    info = inspect_model(model_path)
    ep = info["plans"].get(existing_plan)
    rp = info["plans"].get(revised_plan)
    if ep is None or rp is None:
        raise KeyError("Selected plan was not found in the active model")
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        raise RuntimeError("Both selected plans require HDF output files for direct HEC-RAS result mapping")
    src = open_model_source(Path(model_path))
    td = tempfile.TemporaryDirectory(prefix="hecras_raster_hdf_")
    e_hdf = src.materialize(ep["hdf_archive_member"], td.name)
    r_hdf = src.materialize(rp["hdf_archive_member"], td.name)
    return info, src, td, e_hdf, r_hdf

def _format_bytes(n: int | float) -> str:
    n = float(n or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"



def _json_text(value):
    """Convert nested/non-scalar objects to compact text for Arrow-safe table display."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str, separators=(", ", ": "))
        except Exception:
            return str(value)
    return value


def _arrow_safe_dataframe(data) -> pd.DataFrame:
    """Return a display-only DataFrame that Streamlit/PyArrow can serialize reliably.

    HEC-RAS comparison tables can contain nested lists/dicts in columns such as
    Existing/Revised settings. Pandas permits these object cells but Arrow requires
    a consistent scalar type. Object columns are therefore rendered as readable text.
    Numeric/datetime columns are left unchanged.
    """
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == "object":
            converted = df[col].map(_json_text)
            # Force object columns to one Arrow-safe scalar type while preserving nulls.
            df[col] = converted.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).astype("string")
    # Arrow can also reject object-typed indexes in some environments.
    if getattr(df.index, "dtype", None) == "object":
        df.index = df.index.map(lambda v: str(_json_text(v)))
    return df


def _show_dataframe(data, **kwargs):
    """Render a comparison table after normalizing nested values for PyArrow."""
    return st.dataframe(_arrow_safe_dataframe(data), **kwargs)

def _native_folder_picker(initial: str = "") -> str | None:
    """Open a native Windows folder picker when Streamlit is running on the local workstation.

    Browsers cannot disclose an arbitrary local folder path to a web application. This helper
    therefore opens the picker from the local Streamlit Python process. It is optional and the
    normal text-path fields remain available for remote/server deployments.
    """
    if os.name != "nt":
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askdirectory(initialdir=initial or None, mustexist=True)
        root.destroy()
        return picked or None
    except Exception:
        return None


def _parse_dependency_text(text: str) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for line in str(text or "").splitlines():
        value = line.strip().strip('"')
        if not value:
            continue
        p = Path(value).expanduser()
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def delta_scatter(coords: np.ndarray, delta: np.ndarray, title: str, unit: str = "ft"):
    if coords is None or len(coords) == 0:
        return None
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "delta": delta})
    finite = df["delta"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return None
    lim = float(np.nanpercentile(np.abs(finite), 98))
    lim = max(lim, 0.001)
    fig = px.scatter(
        df, x="x", y="y", color="delta", color_continuous_midpoint=0,
        range_color=(-lim, lim), labels={"delta": f"Δ ({unit})", "x": "State Plane X", "y": "State Plane Y"},
        title=title,
    )
    fig.update_traces(marker={"size": 4})
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=650)
    return fig


st.markdown("### Model / workspace source")
source_mode = st.radio(
    "Source mode",
    [
        "Local extracted model + dependencies (recommended for large models)",
        "Local archive path (.zip/.7z)",
        "Browser upload (.zip/.7z — smaller datasets)",
    ],
    horizontal=True,
    key="source_mode_v10",
)
workspace_root: Path | None = None
original_archive_paths: list[Path] = []
dependency_folders: list[Path] = []
source_is_local_folder = source_mode.startswith("Local extracted")

if source_is_local_folder:
    st.info(
        "Recommended for MAAPnext/FEMA-scale datasets. Select the folder that contains the HEC-RAS project and, optionally, any Terrain/HMS/DSS/LandCover/other dependency folders. "
        "The tool reads them directly from disk; no archive or browser upload is required."
    )
    b1, b2 = st.columns([1, 3])
    if b1.button("Browse model folder…", disabled=os.name != "nt", key="browse_model_v10"):
        picked = _native_folder_picker(st.session_state.get("local_model_folder_v10", ""))
        if picked:
            st.session_state["local_model_folder_v10"] = picked
            st.rerun()
    b2.caption("The Browse button works when Streamlit is running locally on Windows. You can always paste/type a path instead.")
    local_model_text = st.text_input(
        "HEC-RAS model folder",
        key="local_model_folder_v10",
        placeholder=r"D:\MAAPnext\RAS_Model",
        help="Choose the RAS model folder itself. It may contain one or more HEC-RAS .prj projects.",
    )

    d1, d2 = st.columns([1, 3])
    if d1.button("Add dependency folder…", disabled=os.name != "nt", key="browse_dep_v10"):
        picked = _native_folder_picker()
        if picked:
            current = st.session_state.get("dependency_folders_text_v10", "").strip()
            lines = [x.strip() for x in current.splitlines() if x.strip()]
            if picked.lower() not in {x.lower() for x in lines}:
                lines.append(picked)
            st.session_state["dependency_folders_text_v10"] = "\n".join(lines)
            st.rerun()
    d2.caption("Add as many search folders as needed. They can be Terrain, HEC-HMS/DSS, LandCover, or any other external dependency location.")
    dependency_text = st.text_area(
        "Dependency folders — one folder per line",
        key="dependency_folders_text_v10",
        height=120,
        placeholder="D:\\MAAPnext\\Terrain\nD:\\MAAPnext\\HMS_DSS\nE:\\Regional_Data",
    )

    if st.button("Validate & load local model", type="primary", key="load_local_v10"):
        model_candidate = Path(local_model_text.strip().strip('"')).expanduser()
        dep_candidates = _parse_dependency_text(dependency_text)
        bad_deps = [str(x) for x in dep_candidates if not x.is_dir()]
        if not model_candidate.is_dir():
            st.error(f"Model folder not found: {model_candidate}")
        elif bad_deps:
            st.error("Dependency folder(s) not found: " + " | ".join(bad_deps))
        elif not discover_runnable_projects(model_candidate):
            st.error("No runnable HEC-RAS .prj project with sibling .p## plan files was found under the selected model folder.")
        else:
            st.session_state["loaded_source_v10"] = {
                "mode": source_mode,
                "workspace_root": str(model_candidate.resolve()),
                "dependency_folders": [str(x.resolve()) for x in dep_candidates],
            }
            # A new source invalidates any protected run copy from a previous model.
            st.session_state.pop("protected_run_workspace_v10", None)
            _reset_review_state()
            st.rerun()

    loaded = st.session_state.get("loaded_source_v10", {})
    if loaded.get("mode") == source_mode and Path(loaded.get("workspace_root", "")).is_dir():
        workspace_root = Path(loaded["workspace_root"]).resolve()
        dependency_folders = [Path(x) for x in loaded.get("dependency_folders", []) if Path(x).is_dir()]
    if workspace_root is None:
        st.stop()

elif source_mode.startswith("Local archive"):
    st.warning(
        "Archive mode is retained for convenience. For a 100+ GB archive, pre-extract once and use Local extracted model + dependencies mode to avoid repeated extraction and disk duplication."
    )
    detected_7z = discover_7zip_executable() or ""
    with st.form("local_archive_form_v10"):
        archive_text = st.text_input("Local .zip/.7z archive", value=st.session_state.get("local_archive_text_v10", ""), placeholder=r"D:\MAAPnext\MAAPnext.7z")
        destination_text = st.text_input("Extraction destination (optional)", value=st.session_state.get("archive_destination_text_v10", ""), placeholder=r"D:\HECRAS_Review_Workspaces\MAAPnext")
        seven_zip_text = st.text_input("7z.exe path (needed for .7z)", value=st.session_state.get("seven_zip_text_v10", detected_7z), placeholder=r"C:\Program Files\7-Zip\7z.exe")
        extract_archive = st.form_submit_button("Extract and load archive", type="primary")
    if extract_archive:
        archive = Path(archive_text.strip().strip('"')).expanduser()
        if not archive.is_file():
            st.error(f"Archive not found: {archive}")
        else:
            destination = Path(destination_text.strip().strip('"')).expanduser() if destination_text.strip() else Path(tempfile.mkdtemp(prefix="hecras_review_workspace_"))
            destination.mkdir(parents=True, exist_ok=True)
            try:
                estimate = archive_uncompressed_size(archive, seven_zip_text.strip() or None)
                free = shutil.disk_usage(destination).free
                if estimate and estimate > free:
                    st.error(f"Estimated uncompressed size {_format_bytes(estimate)} exceeds free space {_format_bytes(free)} at {destination}.")
                else:
                    if estimate:
                        st.caption(f"Estimated uncompressed size: {_format_bytes(estimate)} | free disk space: {_format_bytes(free)}")
                    with st.spinner("Extracting local archive while preserving the complete workspace tree..."):
                        root = prepare_working_copy(archive, destination=destination, seven_zip_exe=seven_zip_text.strip() or None)
                    st.session_state["local_archive_text_v10"] = str(archive.resolve())
                    st.session_state["archive_destination_text_v10"] = str(destination.resolve())
                    st.session_state["seven_zip_text_v10"] = seven_zip_text
                    st.session_state["loaded_source_v10"] = {"mode": source_mode, "workspace_root": str(root), "archive": str(archive.resolve())}
                    st.session_state.pop("protected_run_workspace_v10", None)
                    _reset_review_state()
                    st.rerun()
            except Exception as exc:
                st.error(f"Archive could not be extracted: {exc}")
    loaded = st.session_state.get("loaded_source_v10", {})
    if loaded.get("mode") == source_mode and Path(loaded.get("workspace_root", "")).is_dir():
        workspace_root = Path(loaded["workspace_root"])
        if loaded.get("archive"):
            original_archive_paths = [Path(loaded["archive"])]
    if workspace_root is None:
        st.stop()

else:
    st.info(
        "Browser upload is retained for smaller review packages. For multi-GB/100+ GB datasets use Local extracted model + dependencies mode instead."
    )
    uploaded_files = st.file_uploader(
        "Upload HEC-RAS review archive(s)", type=["zip", "7z"], accept_multiple_files=True,
        help="Upload one complete workspace archive, or multiple component archives. For large datasets use Local extracted mode.",
    )
    if not uploaded_files:
        st.stop()
    original_archive_paths = persist_uploads(uploaded_files)
    has_7z = any(p.suffix.lower() == ".7z" for p in original_archive_paths)
    upload_7z_exe = st.text_input("7z.exe path for uploaded .7z files" if has_7z else "7z.exe path (not required for ZIP)", value=discover_7zip_executable() or "", key="upload_7z_path_v10")
    key = "|".join(f"{p}:{p.stat().st_size}" for p in original_archive_paths)
    loaded = st.session_state.get("browser_extracted_source_v10", {})
    if loaded.get("key") == key and Path(loaded.get("workspace_root", "")).is_dir():
        workspace_root = Path(loaded["workspace_root"])
    else:
        with st.spinner("Extracting uploaded archive(s) into a working workspace..."):
            try:
                root = prepare_working_copy(original_archive_paths, seven_zip_exe=upload_7z_exe.strip() or None)
                st.session_state["browser_extracted_source_v10"] = {"key": key, "workspace_root": str(root)}
                workspace_root = root
                st.session_state.pop("protected_run_workspace_v10", None)
                _reset_review_state()
            except Exception as exc:
                st.error(f"Uploaded archive(s) could not be prepared: {exc}")
                st.stop()

assert workspace_root is not None
workspace_root = workspace_root.resolve()
st.caption(f"Source search root: {workspace_root}")

projects = discover_runnable_projects(workspace_root)
if not projects:
    st.error("No runnable HEC-RAS project was found. Select a model folder containing a .prj file with sibling .p## plan files.")
    st.stop()
if len(projects) > 1:
    selected_project_file = st.selectbox(
        "HEC-RAS project", projects,
        format_func=lambda p: str(p.relative_to(workspace_root)) if workspace_root in p.parents else str(p),
        key="selected_project_v10",
    )
else:
    selected_project_file = projects[0]
selected_project_file = Path(selected_project_file).resolve()
original_project_file = selected_project_file
original_model_path = original_project_file.parent

# If a protected local run workspace has already been created, automatically review that
# copy so newly generated .p##.hdf files become available without touching the source model.
protected_state = st.session_state.get("protected_run_workspace_v10", {}) if source_is_local_folder else {}
protected_active = (
    protected_state.get("source_project_file", "").lower() == str(original_project_file).lower()
    and Path(protected_state.get("project_file", "")).is_file()
    and Path(protected_state.get("model_directory", "")).is_dir()
)
if protected_active:
    active_project_file = Path(protected_state["project_file"]).resolve()
    active_model_path = Path(protected_state["model_directory"]).resolve()
    st.success("Review source: protected HEC-RAS working copy. Original RAS model remains unchanged.")
    st.caption(f"Original project: {original_project_file}")
    st.caption(f"Protected run project: {active_project_file}")
else:
    active_project_file = original_project_file
    active_model_path = original_model_path
    st.caption(f"HEC-RAS project folder: {active_model_path}")

# Validate the source model's dependency references without forcing a run. Missing external
# data do not prevent review of already-existing HDF outputs; they only block computation.
if source_is_local_folder:
    try:
        local_dependency_validation = validate_local_workspace(original_project_file, dependency_folders)
        with st.expander("Local dependency validation", expanded=not local_dependency_validation.get("ready", False)):
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("References", local_dependency_validation.get("reference_count", 0))
            vc2.metric("Resolved", local_dependency_validation.get("resolved_count", 0))
            vc3.metric("Unresolved", local_dependency_validation.get("missing_count", 0) + local_dependency_validation.get("ambiguous_count", 0))
            if local_dependency_validation.get("references"):
                vdf = pd.DataFrame(local_dependency_validation["references"])
                show = [c for c in ["kind", "plan", "reference", "status", "resolved_source", "search_folder"] if c in vdf.columns]
                _show_dataframe(vdf[show], use_container_width=True, hide_index=True)
            if local_dependency_validation.get("ready"):
                st.success("All currently discovered external references are resolved.")
            else:
                st.warning("Some dependencies are unresolved. Existing HDF results can still be reviewed, but HEC-RAS computation will remain blocked until the selected run plans resolve.")
    except Exception as exc:
        local_dependency_validation = None
        st.warning(f"Dependency validation could not be completed yet: {exc}")
else:
    local_dependency_validation = None

try:
    inventory = inspect_model(active_model_path)
except Exception as exc:
    st.error(f"Could not inspect HEC-RAS project: {exc}")
    st.stop()

project = inventory["project"]
st.subheader(project.get("project_title") or "HEC-RAS Project")
cols = st.columns(4)
cols[0].metric("RAS project files", inventory.get("archive_file_count", 0))
cols[1].metric("Plans", len(inventory.get("plans", {})))
cols[2].metric("Geometries", len(project.get("geometry_files", [])))
cols[3].metric("Unsteady flows", len(project.get("unsteady_files", [])))

plans = inventory.get("plans", {})
plan_codes = list(plans)
if len(plan_codes) < 2:
    st.warning("At least two plans are required for comparison.")
    st.stop()

e_default = plan_codes.index("p03") if "p03" in plan_codes else 0
r_default = plan_codes.index("p05") if "p05" in plan_codes else min(1, len(plan_codes)-1)
left, right = st.columns(2)
with left:
    existing = st.selectbox("Existing / Effective-like plan", plan_codes, index=e_default, format_func=lambda p: f"{p} — {plans[p].get('plan_title','')}")
with right:
    revised = st.selectbox("Revised / Proposed plan", plan_codes, index=r_default, format_func=lambda p: f"{p} — {plans[p].get('plan_title','')}")
if existing == revised:
    st.warning("Choose two different plans.")
    st.stop()

selected_result_rows = []
for role, code in [("Existing", existing), ("Revised", revised)]:
    pinfo = plans[code]
    selected_result_rows.append({
        "Role": role, "Plan": code, "Plan title": pinfo.get("plan_title", ""),
        "Program version": pinfo.get("program_version", ""),
        "HDF output": "Available" if pinfo.get("hdf_archive_member") else "NOT FOUND",
    })
st.markdown("### Selected-plan output status")
_show_dataframe(pd.DataFrame(selected_result_rows), use_container_width=True, hide_index=True)
missing_outputs = [code for code in (existing, revised) if not plans[code].get("hdf_archive_member")]
if missing_outputs:
    st.warning("No HEC-RAS run output (.p##.hdf) was found for: " + ", ".join(missing_outputs) + ". Use the HEC-RAS Run panel below.")

# Persistent full-width runner. v1.0 protects the original local RAS model by default.
with st.expander("HEC-RAS Run — persistent full-width controls", expanded=True):
    if source_is_local_folder:
        st.info(
            "Local large-model mode can create a protected run workspace containing only a copy of the RAS model folder. "
            "External Terrain/HMS/DSS folders are linked in place with Windows directory junctions, so they are not duplicated."
        )
        protect_original = st.checkbox("Protect original RAS model (recommended)", value=True, key="protect_original_v10")
        default_run_parent = str(original_model_path.parent / "_HECRAS_Review_Runs")
        run_workspace_parent = st.text_input(
            "Protected run-workspace parent folder",
            value=st.session_state.get("run_workspace_parent_v10", default_run_parent),
            key="run_workspace_parent_v10",
            help="Only the RAS model folder is copied here. Large dependency folders remain at their original locations and are junction-linked into the run workspace.",
        )
        if protected_active:
            st.success(f"Protected workspace active: {protected_state.get('run_root')}")
    else:
        protect_original = False
        run_workspace_parent = ""
        st.info("This archive source already runs from an extracted working workspace; the original archive file(s) are not modified.")

    default_plans = missing_outputs[:] if missing_outputs else [revised]
    if "run_plans_v10" not in st.session_state:
        st.session_state["run_plans_v10"] = default_plans
    run_plans = st.multiselect(
        "Plan(s) to run", plan_codes, key="run_plans_v10",
        format_func=lambda p: f"{p} — {plans[p].get('plan_title','')} (model version {plans[p].get('program_version','?')})",
    )
    installs = discover_hecras_installations()
    install_labels = [f"HEC-RAS {x.version} — {x.exe_path}" for x in installs]
    choices = install_labels + ["Custom Ras.exe path"]
    if "run_install_v10" not in st.session_state or st.session_state["run_install_v10"] not in choices:
        st.session_state["run_install_v10"] = choices[0] if installs else "Custom Ras.exe path"
    selected_install = st.selectbox("HEC-RAS installation", choices, key="run_install_v10")
    if selected_install == "Custom Ras.exe path":
        ras_exe = st.text_input("Full path to Ras.exe", key="custom_ras_exe_v10", placeholder=r"C:\Program Files\HEC\HEC-RAS\6.6\Ras.exe")
    else:
        ras_exe = installs[install_labels.index(selected_install)].exe_path
    timeout_hours = st.number_input("Run timeout (hours)", min_value=0.25, max_value=72.0, value=float(st.session_state.get("timeout_hours_v10", 8.0)), step=0.25, key="timeout_hours_v10")
    if run_plans:
        versions = sorted({str(plans[x].get("program_version") or "unknown") for x in run_plans})
        st.caption("Selected plan Program Version field(s): " + ", ".join(versions) + ". Prefer the matching installed HEC-RAS version for reproducibility.")

    run_c1, run_c2 = st.columns([1, 1])
    preflight_clicked = run_c1.button("Check run dependencies", key="preflight_v10")
    run_clicked = run_c2.button("RUN HEC-RAS", type="primary", disabled=not run_plans, key="run_hecras_v10")

    if preflight_clicked or run_clicked:
        try:
            run_model_dir = active_model_path
            run_project_file = active_project_file
            preflight_ready = False

            if source_is_local_folder and protect_original:
                # First validate against the original model + user-selected dependency folders.
                validation = validate_local_workspace(original_project_file, dependency_folders, run_plans)
                st.markdown("#### Source dependency validation")
                if validation.get("references"):
                    pf = pd.DataFrame(validation["references"])
                    show = [c for c in ["kind", "plan", "reference", "status", "resolved_source", "search_folder"] if c in pf.columns]
                    _show_dataframe(pf[show], use_container_width=True, hide_index=True)
                if not validation.get("ready", False):
                    st.error(
                        f"Run stopped: {validation.get('missing_count', 0)} missing and {validation.get('ambiguous_count', 0)} ambiguous dependency reference(s). "
                        "Add/correct the dependency folders above and validate again."
                    )
                    run_clicked = False
                else:
                    st.success("Source dependencies resolved. Large external folders can be linked without copying them.")
                    preflight_ready = True

                if run_clicked and preflight_ready:
                    # Reuse the current protected workspace only if it belongs to this source
                    # and it resolves the plans requested now; otherwise create a fresh snapshot.
                    reuse = False
                    current = st.session_state.get("protected_run_workspace_v10", {})
                    if (
                        current.get("source_project_file", "").lower() == str(original_project_file).lower()
                        and Path(current.get("project_file", "")).is_file()
                    ):
                        try:
                            check = workspace_reference_preflight(
                                current["run_root"], run_plans, auto_stage=False, project_file=current["project_file"]
                            )
                            reuse = bool(check.get("ready"))
                        except Exception:
                            reuse = False
                    if not reuse:
                        destination = Path(run_workspace_parent.strip().strip('"')).expanduser() if run_workspace_parent.strip() else None
                        with st.spinner("Creating protected RAS working copy and dependency junctions (large dependency folders are not copied)..."):
                            current = create_protected_run_workspace(
                                original_project_file,
                                dependency_folders=dependency_folders,
                                plan_codes=run_plans,
                                destination_parent=destination,
                            )
                        st.session_state["protected_run_workspace_v10"] = current
                    run_model_dir = Path(current["model_directory"])
                    run_project_file = Path(current["project_file"])
                    st.markdown("#### Protected run workspace")
                    st.caption(f"RAS copy: {run_model_dir}")
                    links = current.get("links", [])
                    if links:
                        ldf = pd.DataFrame(links)
                        show = [c for c in ["kind", "reference", "run_status", "run_path", "link_source"] if c in ldf.columns]
                        _show_dataframe(ldf[show], use_container_width=True, hide_index=True)
                    preflight = current.get("preflight") or workspace_reference_preflight(
                        current["run_root"], run_plans, auto_stage=False, project_file=run_project_file
                    )
                    preflight_ready = bool(preflight.get("ready"))

            elif source_is_local_folder and not protect_original:
                validation = validate_local_workspace(original_project_file, dependency_folders, run_plans)
                st.markdown("#### Direct-run dependency validation")
                if validation.get("references"):
                    pf = pd.DataFrame(validation["references"])
                    show = [c for c in ["kind", "plan", "reference", "status", "resolved_source"] if c in pf.columns]
                    _show_dataframe(pf[show], use_container_width=True, hide_index=True)
                # A selected dependency folder does not alter the source model's relative
                # path. Direct runs are safe only when every reference already resolves exactly.
                non_exact = [r for r in validation.get("references", []) if r.get("status") != "exact"]
                if non_exact:
                    st.error("Direct run is blocked because some dependencies only resolve through selected search folders. Enable Protect original RAS model so v1.2 can create the required junctions in a separate run workspace.")
                    run_clicked = False
                elif not validation.get("ready"):
                    st.error("Direct run dependency validation failed.")
                    run_clicked = False
                else:
                    st.warning("Protection is disabled. HEC-RAS will write outputs directly into the original RAS model folder.")
                    preflight_ready = True
                    run_model_dir = original_model_path
                    run_project_file = original_project_file
            else:
                preflight = workspace_reference_preflight(workspace_root, run_plans, project_file=selected_project_file)
                st.markdown("#### Run dependency preflight")
                if preflight.get("references"):
                    pf = pd.DataFrame(preflight["references"])
                    cols2 = [c for c in ["kind", "plan", "reference", "resolution_status", "exists", "resolved_path", "found_elsewhere"] if c in pf.columns]
                    _show_dataframe(pf[cols2], use_container_width=True, hide_index=True)
                if not preflight.get("ready", False):
                    st.error(f"Run stopped: {preflight.get('missing_count', 0)} dependency reference(s) remain unresolved.")
                    run_clicked = False
                else:
                    st.success("Run dependency preflight passed.")
                    preflight_ready = True
                    run_model_dir = active_model_path
                    run_project_file = active_project_file

            if preflight_clicked and preflight_ready and not run_clicked:
                st.success("Run preflight passed.")

            if run_clicked and preflight_ready:
                if not ras_exe:
                    st.error("Select an installed HEC-RAS version or enter Ras.exe path.")
                else:
                    history = st.session_state.setdefault("run_history", [])
                    for code in run_plans:
                        status = st.status(f"HEC-RAS {code}: starting", expanded=True)
                        live = st.empty()
                        def callback(info, _live=live):
                            msg = f"Running {info['plan_code']} | elapsed {info['elapsed_seconds']:.0f} s"
                            if info.get("result_hdf_exists"):
                                msg += f" | output HDF {_format_bytes(info.get('result_hdf_size_bytes', 0))}"
                            _live.write(msg)
                        result = run_hecras_plan(
                            run_model_dir, code, ras_exe, timeout_seconds=float(timeout_hours) * 3600,
                            progress_callback=callback, project_file=run_project_file,
                        )
                        history.append(result.to_dict())
                        if result.status == "completed":
                            status.update(label=f"HEC-RAS {code}: completed — output HDF detected", state="complete")
                        else:
                            status.update(label=f"HEC-RAS {code}: {result.status}", state="error")
                            if result.error:
                                st.error(result.error)
                            if result.log_tail:
                                st.code("\n".join(result.log_tail[-20:]))
                    st.session_state.pop("report", None)
                    st.session_state.pop("report_pair", None)
                    st.success("HEC-RAS run finished. Refreshing model inventory from the working copy...")
                    st.rerun()
        except Exception as exc:
            st.error(f"HEC-RAS run/preflight could not be completed: {exc}")

    if st.session_state.get("run_history"):
        st.markdown("#### Run history")
        _show_dataframe(pd.DataFrame(st.session_state["run_history"]), use_container_width=True, hide_index=True)

    if not source_is_local_folder:
        st.divider()
        st.caption("Optional: package the extracted working workspace, including newly generated outputs, as a new ZIP.")
        if st.button("Prepare updated workspace ZIP", key="prepare_updated_zip_v10"):
            out = Path(tempfile.gettempdir()) / f"{project.get('project_title') or 'HECRAS'}_working_with_outputs.zip"
            with st.spinner("Packaging the working workspace..."):
                zip_working_model(workspace_root, out)
            st.session_state["updated_working_zip"] = str(out)
        prepared = st.session_state.get("updated_working_zip")
        if prepared and Path(prepared).exists():
            st.download_button("Download updated workspace ZIP", data=open(prepared, "rb"), file_name=Path(prepared).name, mime="application/zip", key="download_updated_zip_v10")

pair_key = (str(active_model_path), existing, revised)
if st.button("Run comparison", type="primary", disabled=bool(missing_outputs)):
    with st.spinner("Reading selected plan HDF files and comparing model inputs, mesh and results..."):
        st.session_state["report"] = compare_plans(active_model_path, existing, revised)
        st.session_state["report_pair"] = pair_key

report = st.session_state.get("report")
if report is None or st.session_state.get("report_pair") != pair_key:
    st.info("Choose the review pair and click Run comparison. HEC-RAS Run controls remain available above even when results are missing.")
    st.stop()

st.success("Comparison complete")
perf = report.get("performance", {})
if perf.get("large_model_mode"):
    st.info(
        "Large-model optimized mode was used for the initial comparison. Full 2D time-series, "
        "face-level velocity geometry matching, and changed-mesh normalization are deferred to "
        "selected-area review so the initial comparison does not overload memory or drop the Streamlit connection."
    )
pair = report["review_pair"]
st.subheader("Review pair")
pair_df = pd.DataFrame([pair["existing"], pair["revised"]], index=["Existing", "Revised"])
_show_dataframe(pair_df, use_container_width=True)

summary = report.get("impact_summary", {})
flagset = report.get("reviewer_flags", {})
st.markdown("### Reviewer screening summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Review flags", flagset.get("flag_count", 0))
c2.metric("Changed 2D areas", len(summary.get("changed_2d_areas", [])))
c3.metric("Modified 1D XS", summary.get("modified_cross_sections", 0))
c4.metric("Added culvert groups", summary.get("added_culvert_groups", 0))
st.caption("Flags prioritize reviewer attention only. They are not FEMA/local regulatory acceptance criteria and do not produce an automatic pass/fail decision.")
if flagset.get("flags"):
    flag_df = pd.DataFrame(flagset["flags"])
    show_cols = [c for c in ["category", "title", "detail", "area", "value", "threshold", "unit"] if c in flag_df.columns]
    _show_dataframe(flag_df[show_cols], use_container_width=True, hide_index=True)

tabs = st.tabs([
    "Model Changes", "2D Mesh", "Georeferenced Map", "Raster Map", "Spatial Review", "Structures",
    "Boundary Conditions", "Hydraulic Results", "Review Report", "Raw Report"
])

with tabs[0]:
    st.markdown("### Plan/computation settings")
    if report["plan_setting_changes"]:
        _show_dataframe(pd.DataFrame(report["plan_setting_changes"]), use_container_width=True)
    else:
        st.success("No tracked computation-setting changes detected.")
    st.markdown("### 1D cross-section identity")
    st.json(report["cross_section_comparison"], expanded=False)
    st.markdown("### Detailed 1D cross-section changes")
    xs_detail = report.get("cross_section_detailed_comparison", {})
    if xs_detail.get("modified_count", 0) == 0 and not xs_detail.get("added") and not xs_detail.get("removed"):
        st.success("All common cross sections match for station/elevation geometry, bank/length attributes, Manning's n, ineffective blocks, and mapped XS polylines.")
    st.json(xs_detail, expanded=False)
    st.markdown("### Breaklines")
    st.json(report["breaklines"], expanded=False)

with tabs[1]:
    mesh_df = pd.DataFrame.from_dict(report["mesh_comparison"], orient="index")
    _show_dataframe(mesh_df, use_container_width=True)
    st.caption("Exact cell-center equality is used only as a geometry identity check. Changed meshes are not forced into index-to-index equivalence.")

with tabs[2]:
    st.markdown("### Georeferenced reviewer map")
    area_names = list(report["mesh_comparison"])
    map_options = ["All 2D Flow Areas"] + area_names if len(area_names) > 1 else area_names
    map_area = st.selectbox("2D flow area / extent", map_options, key="map_area")
    map_threshold = st.number_input("Map |ΔWSEL| threshold (ft)", min_value=0.0, value=0.01, step=0.01, format="%.3f", key="map_threshold")
    large_mode = bool(report.get("performance", {}).get("large_model_mode"))
    detailed_map_layers = st.checkbox(
        "Include velocity, same-time WSEL, and changed-mesh normalized layers (slower)",
        value=False if (large_mode or map_area == "All 2D Flow Areas") else True,
        disabled=(map_area == "All 2D Flow Areas" and large_mode),
        key="map_detailed_layers",
        help="For large/all-area overview maps, leave this off. Select one area when detailed time-series/velocity layers are needed.",
    )
    st.caption("Choose All 2D Flow Areas for a combined map, or select one area for a focused view. The combined view builds every area before creating one final layer control so all area layers remain available.")
    with st.spinner(f"Building georeferenced review map for {map_area}..."):
        selected_areas = area_names if map_area == "All 2D Flow Areas" else [map_area]
        map_spatials = [
            spatial_area_review(
                active_model_path, existing, revised, a, grid_spacing_ft=100.0,
                include_time_series=bool(detailed_map_layers),
                include_normalized=bool(detailed_map_layers),
                include_velocity=bool(detailed_map_layers),
            )
            for a in selected_areas
            if report["mesh_comparison"].get(a, {}).get("status") not in {"added_area", "removed_area"}
        ]
        if not map_spatials:
            st.warning("No 2D flow area is present in both selected plans for hydraulic overlay mapping.")
            st.stop()
        map_spatial = map_spatials if len(map_spatials) > 1 else map_spatials[0]
        e_ref = map_reference_geometry(active_model_path, existing)
        r_ref = map_reference_geometry(active_model_path, revised)
        projection_wkt = map_spatials[0].get("projection_wkt") or e_ref.get("projection_wkt") or r_ref.get("projection_wkt")
        added_culvert_keys = {
            (str(x.get("Type", "")), str(x.get("River", "")), str(x.get("Reach", "")), str(x.get("RS", "")))
            for x in report["structure_comparison"].get("added_culvert_groups", [])
        }
        added_culvert_keys.update({
            tuple(str(v) for v in x.get("key", [])[:4])
            for x in report["structure_comparison"].get("modified_structures", [])
            if len(x.get("key", [])) >= 4
        })
        additional_perimeters = None
        if map_area == "All 2D Flow Areas":
            e_per = map_2d_area_perimeters(active_model_path, existing)
            r_per = map_2d_area_perimeters(active_model_path, revised)
            additional_perimeters = {
                "existing": e_per.get("areas", {}),
                "revised": r_per.get("areas", {}),
            }
        review_map, map_meta = build_review_map(
            map_spatial, projection_wkt, delta_threshold_ft=map_threshold,
            cross_sections=e_ref.get("cross_sections", []),
            structures=r_ref.get("structures", []),
            bc_lines=r_ref.get("boundary_condition_lines", []),
            breaklines=r_ref.get("breaklines", []),
            added_culvert_keys=added_culvert_keys,
            additional_perimeters=additional_perimeters,
        )
        html = map_html(review_map)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2D areas shown", map_meta.get("area_count", 1))
    c2.metric("Mapped ΔWSEL locations", f"{map_meta.get('wse_feature_count', 0):,}")
    c3.metric("Revised-only centers", f"{map_meta.get('revised_only_center_count', 0):,}")
    c4.metric("Mapped XS", f"{map_meta.get('cross_section_feature_count', 0):,}")
    components.html(html, height=820, scrolling=False)
    safe_area_name = "ALL_2D" if map_area == "All 2D Flow Areas" else map_area
    st.download_button(
        "Download interactive HTML map", html,
        file_name=f"{existing}_vs_{revised}_{safe_area_name}_review_map.html",
        mime="text/html", key="download_map",
    )
    st.caption(f"Coordinate system: {map_meta.get('source_crs_name')} → WGS84. Base-map wrapping is disabled and the map is fitted to the selected hydraulic extent with context padding.")



with tabs[3]:
    st.markdown("### Georeferenced raster-output map")
    raster_source = st.radio(
        "Raster source",
        ["HEC-RAS Plan Results", "Stored / External Raster Files"],
        horizontal=True,
        help=(
            "HEC-RAS Plan Results reads Maximum WSE / Depth / Velocity directly from the selected plan HDF results. "
            "Stored / External Raster Files scans GeoTIFF/VRT files already saved in the project folder."
        ),
        key="raster_source_mode",
    )
    if raster_source == "HEC-RAS Plan Results":
        st.caption(
            "Display raster-like maps generated directly from the selected plan HDF results. "
            "This does not require stored GeoTIFF result exports from RAS Mapper. The tool rasterizes the actual HEC-RAS 2D cell polygons to create a continuous hydraulic surface."
        )
        try:
            model_info, _src, _tmpdir, e_hdf, r_hdf = _materialize_selected_plan_hdfs(active_model_path, existing, revised)
            try:
                e_hinv = inspect_model(active_model_path)["plans"][existing]
                # determine common 2D areas by inventory from the HDFs
                from hecras_review.hdf_reader import read_hdf_inventory
                einv = read_hdf_inventory(e_hdf)
                rinv = read_hdf_inventory(r_hdf)
                common_areas = sorted(set(einv.get("two_d_areas", {})) & set(rinv.get("two_d_areas", {})))
                if not common_areas:
                    st.info("No common 2D flow areas were found in the selected plan HDF outputs.")
                else:
                    c1, c2, c3 = st.columns([1.2, 1.2, 1.4])
                    with c1:
                        result_area = st.selectbox("2D flow area", (["All 2D Flow Areas"] + common_areas) if len(common_areas) > 1 else common_areas, key="hdf_raster_area")
                    with c2:
                        result_param = st.selectbox("Result parameter", ["Maximum WSE", "Maximum Depth", "Maximum Face Velocity"], key="hdf_raster_param")
                    with c3:
                        display_mode = st.selectbox("Display", ["Existing", "Revised", "Revised - Existing"], index=2, key="hdf_raster_display")
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        raster_opacity = st.slider("Raster opacity", 0.20, 1.00, 0.74, 0.05, key="hdf_raster_opacity")
                    with rc2:
                        raster_clip = st.slider("Display percentile clip", 0.0, 10.0, 2.0, 0.5, key="hdf_raster_clip")
                    with rc3:
                        raster_pixels = st.selectbox("Browser display pixels", [500_000, 1_000_000, 1_500_000, 2_000_000], index=1, format_func=lambda x: f"{x/1_000_000:.1f} million", key="hdf_raster_pixels")
                    with st.spinner(f"Building HEC-RAS result map for {result_area} — {result_param} ({display_mode})..."):
                        rm, rmeta = build_plan_result_surface_map(
                            e_hdf, r_hdf, result_area, result_param, display_mode=display_mode,
                            max_display_pixels=int(raster_pixels), opacity=float(raster_opacity), percentile_clip=float(raster_clip),
                        )
                    components.html(raster_map_html(rm), height=760, scrolling=False)
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Source", rmeta.get("source", ""))
                    mc2.metric("Parameter", rmeta.get("parameter", ""))
                    mc3.metric("Display", rmeta.get("display_mode", ""))
                    mc4.metric("Mapped values", f"{rmeta.get('result_value_summary',{}).get('count',0):,}")
                    note = (
                        "This browser layer is generated directly from HEC-RAS plan results as a continuous georeferenced raster. "
                        "WSE and Depth are rasterized from the actual 2D cell polygons. "
                        "For Velocity, the maximum of the bounding face Maximum Face Velocity values is assigned to each cell for display. "
                        "Delta maps rasterize Existing and Revised independently onto the same model-space grid, then subtract pixel-by-pixel, so changed meshes are handled spatially."
                    )
                    st.caption(note)
                    if result_param == "Maximum Depth" and "derived" in str(rmeta.get("native_result_source", "")).lower():
                        st.info(f"Depth source: {rmeta.get('native_result_source')}. At least one plan did not contain native depth output, so a clearly labelled proxy was used.")
                    with st.expander("HEC-RAS result map metadata"):
                        st.json(rmeta)
            finally:
                _src.close()
                _tmpdir.cleanup()
        except Exception as exc:
            st.error(f"HEC-RAS plan-result map could not be prepared: {exc}")
    else:
        st.caption(
            "This map reads exported GeoTIFF/VRT results from the HEC-RAS project folder. "
            "Large rasters are reprojected and downsampled only for browser display; the source raster is never modified."
        )
        try:
            rasters = discover_result_rasters(active_model_path)
            if not rasters:
                st.info("No .tif/.tiff/.vrt raster outputs were found below the HEC-RAS project folder. Export the desired RAS Mapper result raster, then refresh the workspace.")
            else:
                categories = sorted({x["category"] for x in rasters})
                preferred_cat = "Delta WSE" if "Delta WSE" in categories else ("WSE" if "WSE" in categories else categories[0])
                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    raster_category = st.selectbox("Raster category", categories, index=categories.index(preferred_cat), key="raster_category")
                filtered = [x for x in rasters if x["category"] == raster_category]
                with rc2:
                    selected_raster = st.selectbox(
                        "Raster", filtered,
                        format_func=lambda x: f"{x['relative_path']}  ({_format_bytes(x['size_bytes'])})",
                        key="raster_output_select",
                    )
                rr1, rr2, rr3 = st.columns(3)
                with rr1:
                    raster_opacity = st.slider("Raster opacity", 0.20, 1.00, 0.72, 0.05, key="raster_opacity")
                with rr2:
                    raster_clip = st.slider("Display percentile clip", 0.0, 10.0, 2.0, 0.5, key="raster_clip")
                with rr3:
                    raster_pixels = st.selectbox("Browser display pixels", [500_000, 1_000_000, 1_500_000, 2_000_000], index=2, format_func=lambda x: f"{x/1_000_000:.1f} million", key="raster_pixels")
                with st.spinner(f"Building raster map for {selected_raster['name']}..."):
                    rm, rmeta = build_raster_review_map(
                        selected_raster["path"], max_display_pixels=int(raster_pixels),
                        opacity=float(raster_opacity), percentile_clip=float(raster_clip),
                    )
                components.html(raster_map_html(rm), height=760, scrolling=False)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Category", rmeta.get("category", ""))
                mc2.metric("Source raster", f"{rmeta.get('source_width',0):,} × {rmeta.get('source_height',0):,}")
                mc3.metric("Display raster", f"{rmeta.get('display_width',0):,} × {rmeta.get('display_height',0):,}")
                mc4.metric("CRS", rmeta.get("source_crs", ""))
                focus_note = "valid-data footprint" if rmeta.get("focused_on_valid_data_bounds") else "full raster footprint"
                st.caption(
                    f"Initial zoom is fitted to the {focus_note}; repeated world-map wrapping is disabled. "
                    "The map image is a visualization layer only. Review calculations should continue to use the source HDF/raster values, not the downsampled browser image."
                )
                with st.expander("Raster metadata"):
                    st.json(rmeta)
        except Exception as exc:
            st.error(f"Raster map could not be prepared: {exc}")

with tabs[4]:
    st.markdown("### Interactive 2D spatial comparison")
    area_names = list(report["mesh_comparison"])
    spatial_options = ["All 2D Flow Areas"] + area_names if len(area_names) > 1 else area_names
    area_choice = st.selectbox("2D flow area", spatial_options, key="spatial_area")
    default_spacing = 100.0
    spacing = st.number_input("Common analysis-grid spacing (ft)", min_value=10.0, max_value=1000.0, value=default_spacing, step=10.0)
    threshold = st.number_input("Highlight |ΔWSEL| threshold (ft)", min_value=0.0, value=0.01, step=0.01, format="%.3f")
    selected_areas = area_names if area_choice == "All 2D Flow Areas" else [area_choice]
    large_mode = bool(report.get("performance", {}).get("large_model_mode"))
    detailed_spatial = st.checkbox(
        "Run detailed time-series / velocity / changed-mesh normalization",
        value=False if (large_mode or area_choice == "All 2D Flow Areas") else True,
        key="spatial_detailed_mode",
        help="This can be memory intensive on MAAPnext-scale models. For large models, select one 2D area and enable only when needed.",
    )
    if large_mode and area_choice == "All 2D Flow Areas" and detailed_spatial:
        st.warning("Detailed all-area spatial analysis can require very large memory. The tool will use lightweight mode instead.")
        detailed_spatial = False
    selected_areas = [a for a in selected_areas if report["mesh_comparison"].get(a, {}).get("status") not in {"added_area", "removed_area"}]

    with st.spinner(f"Preparing spatial review for {area_choice}..."):
        spatials = [
            spatial_area_review(
                active_model_path, existing, revised, a, grid_spacing_ft=spacing,
                include_time_series=bool(detailed_spatial),
                include_normalized=bool(detailed_spatial),
                include_velocity=bool(detailed_spatial),
            ) for a in selected_areas
        ]

    total_both_wet = sum(int(x.get("direct_wet_dry", {}).get("both_wet_count", 0)) for x in spatials)
    total_dry = sum(int(x.get("direct_wet_dry", {}).get("became_dry_count", 0)) for x in spatials)
    total_wet = sum(int(x.get("direct_wet_dry", {}).get("became_wet_count", 0)) for x in spatials)
    total_grid = sum(int(x.get("normalized_max_wse", {}).get("point_count", 0)) for x in spatials)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wet-to-wet common centers", f"{total_both_wet:,}")
    c2.metric("Became dry", f"{total_dry:,}")
    c3.metric("Became wet", f"{total_wet:,}")
    c4.metric("Common-grid points", f"{total_grid:,}")

    combined_coords = []
    combined_delta = []
    transition_fig = go.Figure()
    mesh_fig = go.Figure()
    for spatial in spatials:
        area = spatial["area_name"]
        direct = spatial["direct_max_wse"]
        wet = spatial["direct_wet_dry"]
        coords = np.asarray(direct.get("coordinates", []), dtype=float)
        delta = np.asarray(direct.get("delta", []), dtype=float)
        wet_mask = np.asarray(wet.get("both_wet_mask", []), dtype=bool)
        if len(delta):
            mask = wet_mask & np.isfinite(delta) & (np.abs(delta) >= threshold)
            if np.any(mask):
                combined_coords.append(coords[mask])
                combined_delta.append(delta[mask])
            became_dry = np.asarray(wet.get("became_dry_mask", []), dtype=bool)
            became_wet = np.asarray(wet.get("became_wet_mask", []), dtype=bool)
            if np.any(became_dry):
                transition_fig.add_trace(go.Scattergl(x=coords[became_dry,0], y=coords[became_dry,1], mode="markers", name=f"{area} — became dry ({became_dry.sum():,})", marker={"symbol":"x", "size":7}))
            if np.any(became_wet):
                transition_fig.add_trace(go.Scattergl(x=coords[became_wet,0], y=coords[became_wet,1], mode="markers", name=f"{area} — became wet ({became_wet.sum():,})", marker={"symbol":"cross", "size":7}))
        eo = np.asarray(direct.get("existing_only_coordinates", []), dtype=float)
        ro = np.asarray(direct.get("revised_only_coordinates", []), dtype=float)
        if len(eo):
            mesh_fig.add_trace(go.Scattergl(x=eo[:,0], y=eo[:,1], mode="markers", name=f"{area} — Existing-only ({len(eo):,})", marker={"symbol":"x", "size":7}))
        if len(ro):
            mesh_fig.add_trace(go.Scattergl(x=ro[:,0], y=ro[:,1], mode="markers", name=f"{area} — Revised-only ({len(ro):,})", marker={"symbol":"cross", "size":7}))

    if combined_delta:
        cc = np.vstack(combined_coords)
        cd = np.concatenate(combined_delta)
        fig = delta_scatter(cc, cd, f"{area_choice}: Wet-to-Wet Direct ΔMaximum WSEL")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Only wet-to-wet locations are treated as WSEL comparisons. If native HEC-RAS Depth is unavailable, the tool uses a clearly labelled Water Surface − Cell Minimum Elevation wet/dry proxy.")

    if transition_fig.data:
        transition_fig.update_layout(title=f"{area_choice}: Wet/Dry Transition Locations", height=550, xaxis_title="State Plane X", yaxis_title="State Plane Y")
        transition_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(transition_fig, use_container_width=True)
    if mesh_fig.data:
        mesh_fig.update_layout(title=f"{area_choice}: Mesh Modification Locations", height=600, xaxis_title="State Plane X", yaxis_title="State Plane Y")
        mesh_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(mesh_fig, use_container_width=True)

    for spatial in spatials:
        area = spatial["area_name"]
        if len(spatials) > 1:
            st.markdown(f"#### {area} — normalized / time-synchronized details")
        depth_source = spatial.get("depth_data_source", {})
        if depth_source.get("uses_proxy"):
            st.info(f"{area}: native Depth was not available for one or both plans. Wet/dry/depth screening uses a derived proxy. Existing: {depth_source.get('existing')}; Revised: {depth_source.get('revised')}.")
        norm = spatial["normalized_max_wse"]
        nwet = spatial.get("normalized_wet_dry")
        ncoords = np.asarray(norm.get("coordinates", []), dtype=float)
        ndelta = np.asarray(norm.get("delta", []), dtype=float)
        if len(ndelta) and nwet is not None:
            n_both_wet = np.asarray(nwet.get("both_wet_mask", []), dtype=bool)
            nmask = n_both_wet & np.isfinite(ndelta) & (np.abs(ndelta) >= threshold)
            fig = delta_scatter(ncoords[nmask], ndelta[nmask], f"{area}: Wet-to-Wet Spatially Normalized ΔMaximum WSEL", unit="ft")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        sync = spatial["same_time_wse"]
        if sync.get("compatible"):
            sdelta = np.asarray(sync.get("peak_absolute_same_time_delta", []), dtype=float)
            scoords = np.asarray(sync.get("coordinates", []), dtype=float)
            smask = np.isfinite(sdelta) & (np.abs(sdelta) >= threshold)
            fig = delta_scatter(scoords[smask], sdelta[smask], f"{area}: Largest Same-Time WSEL Difference During Simulation")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        elif sync.get("reason"):
            st.caption(f"{area}: same-time WSEL comparison unavailable — {sync.get('reason')}")

with tabs[5]:
    structs = report["structure_comparison"]
    c1, c2 = st.columns(2)
    c1.metric("Existing culvert groups", structs["culvert_group_count"]["existing"])
    c2.metric("Revised culvert groups", structs["culvert_group_count"]["revised"])
    if structs["added_culvert_groups"]:
        st.markdown("### Added culvert groups")
        _show_dataframe(pd.DataFrame(structs["added_culvert_groups"]), use_container_width=True)
    if structs["removed_culvert_groups"]:
        st.markdown("### Removed culvert groups")
        _show_dataframe(pd.DataFrame(structs["removed_culvert_groups"]), use_container_width=True)
    if structs.get("modified_structures"):
        st.markdown("### Modified structures")
        st.json(structs["modified_structures"], expanded=False)
    if structs["modified_culvert_groups"]:
        st.markdown("### Modified culvert groups")
        st.json(structs["modified_culvert_groups"], expanded=False)

with tabs[6]:
    st.markdown("### Geometry BC lines")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Existing**")
        st.json(report["boundary_condition_lines"]["existing"], expanded=False)
    with c2:
        st.markdown("**Revised**")
        st.json(report["boundary_condition_lines"]["revised"], expanded=False)
    st.markdown("### Unsteady-flow boundary changes")
    bc = report["unsteady_boundary_comparison"]
    if bc["added_locations"]:
        st.markdown("**Added locations**")
        st.json(bc["added_locations"], expanded=False)
    if bc["removed_locations"]:
        st.markdown("**Removed locations**")
        st.json(bc["removed_locations"], expanded=False)
    if bc["common_location_parameter_changes"]:
        st.markdown("**Changed parameters at common locations**")
        st.json(bc["common_location_parameter_changes"], expanded=False)

with tabs[7]:
    st.markdown("### 2D hydraulic comparison")
    for area, h in report["hydraulic_comparison"]["two_d"].items():
        with st.expander(area, expanded=True):
            depth_source = h.get("depth_data_source", {})
            if depth_source.get("uses_proxy"):
                st.warning(
                    "Native HEC-RAS Depth output is not available for one or both plans. "
                    f"Depth/wet-dry screening uses a derived proxy. Existing: {depth_source.get('existing')}; "
                    f"Revised: {depth_source.get('revised')}."
                )
            st.markdown("**Primary ΔMaximum WSEL — both models wet**")
            st.json(h.get("max_wse_delta_both_wet", {}), expanded=False)
            st.markdown("**Wet/dry transitions**")
            st.json(h.get("wet_dry_transitions", {}), expanded=False)
            st.markdown("**Raw ΔMaximum WSEL diagnostic**")
            st.json(h.get("raw_max_wse_delta", {}), expanded=False)
            st.markdown("**ΔMaximum Depth**")
            st.json(h.get("max_depth_delta", {}), expanded=False)
            st.markdown("**ΔMaximum Face Velocity**")
            st.json(h.get("max_face_velocity_delta", {}), expanded=False)
            st.markdown("**Same-time WSEL diagnostics — both wet at timestamp**")
            st.json(h.get("same_time_wse_both_wet", {}), expanded=False)
            if h.get("top_same_time_wse_events"):
                st.markdown("**Largest same-time WSEL events**")
                _show_dataframe(pd.DataFrame(h["top_same_time_wse_events"]), use_container_width=True)
            top = h.get("max_wse_top_changes_both_wet")
            if top:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("Largest wet-to-wet increases")
                    _show_dataframe(pd.DataFrame(top["largest_increases"]), use_container_width=True)
                with c2:
                    st.markdown("Largest wet-to-wet decreases")
                    _show_dataframe(pd.DataFrame(top["largest_decreases"]), use_container_width=True)

    st.markdown("### 1D cross-section maximum WSEL")
    xs = report["hydraulic_comparison"]["cross_section_max_wse"]
    if not xs.get("available", False):
        st.info(xs.get("reason", "1D Maximum Water Surface summary output is not available for both selected plans."))
    summary = {k: v for k, v in xs.items() if k != "top_10_absolute_changes"}
    st.json(summary, expanded=False)
    if xs.get("top_10_absolute_changes"):
        _show_dataframe(pd.DataFrame(xs["top_10_absolute_changes"]), use_container_width=True)

    st.divider()
    st.markdown("### 1D longitudinal result profiles")
    st.caption("Choose a 1D summary result and River/Reach. Existing and Revised are plotted together; the second plot is Revised − Existing.")
    try:
        one_d_catalog = available_1d_results(active_model_path, existing, revised)
        summary_catalog = {
            k: v for k, v in one_d_catalog.get("summary", {}).items()
            if "Flow Distribution" not in k
        }
        if summary_catalog:
            preferred = ["Maximum Water Surface", "Maximum Channel Velocity", "Maximum Flow"]
            summary_names = list(summary_catalog)
            default_summary = next((x for x in preferred if x in summary_names), summary_names[0])
            param = st.selectbox(
                "1D profile parameter", summary_names, index=summary_names.index(default_summary),
                format_func=lambda x: f"{summary_catalog[x].get('primary_variable', x)} ({summary_catalog[x].get('primary_unit','')}) — {x}",
                key="one_d_profile_parameter",
            )
            profile = one_d_profile_review(active_model_path, existing, revised, param)
            rr_options = profile.get("river_reaches", [])
            if rr_options:
                rr = st.selectbox("River / Reach", rr_options, format_func=lambda x: f"{x[0]} / {x[1]}", key="one_d_rr")
                pdf = pd.DataFrame([x for x in profile["rows"] if (x["river"], x["reach"]) == tuple(rr)])
                pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["rs_numeric", "existing", "revised"])
                pdf = pdf.sort_values("rs_numeric", ascending=False)
                unit = profile.get("unit", "")
                label = profile.get("label", param)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pdf["rs_numeric"], y=pdf["existing"], mode="lines+markers", name=f"Existing — {existing}", customdata=pdf[["rs"]], hovertemplate="RS %{customdata[0]}<br>Existing %{y:.3f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=pdf["rs_numeric"], y=pdf["revised"], mode="lines+markers", name=f"Revised — {revised}", customdata=pdf[["rs"]], hovertemplate="RS %{customdata[0]}<br>Revised %{y:.3f}<extra></extra>"))
                fig.update_layout(title=f"{rr[0]} / {rr[1]} — {label}", xaxis_title="River Station (upstream → downstream)", yaxis_title=f"{label} ({unit})" if unit else label, height=520, hovermode="x unified")
                fig.update_xaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

                dfig = go.Figure()
                dfig.add_trace(go.Scatter(x=pdf["rs_numeric"], y=pdf["delta"], mode="lines+markers", name="Revised − Existing", customdata=pdf[["rs"]], hovertemplate="RS %{customdata[0]}<br>Δ %{y:.3f}<extra></extra>"))
                dfig.add_hline(y=0, line_dash="dash")
                dfig.update_layout(title=f"{rr[0]} / {rr[1]} — Δ{label}", xaxis_title="River Station (upstream → downstream)", yaxis_title=f"Δ {unit}" if unit else "Difference", height=380)
                dfig.update_xaxes(autorange="reversed")
                st.plotly_chart(dfig, use_container_width=True)

                st.markdown("#### 1D cross-section time series")
                ts_preferred = [x for x in ["Water Surface", "Velocity Channel", "Velocity Total", "Flow", "Flow Lateral"] if x in one_d_catalog.get("time_series", [])]
                if ts_preferred and not pdf.empty:
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        ts_param = st.selectbox("Time-series parameter", ts_preferred, key="one_d_ts_parameter")
                    with tc2:
                        xs_options = pdf[["rs", "rs_numeric"]].to_dict("records")
                        selected_xs = st.selectbox("Cross section", xs_options, format_func=lambda x: f"RS {x['rs']}", key="one_d_ts_xs")
                    ts_data = one_d_timeseries_review(active_model_path, existing, revised, rr[0], rr[1], selected_xs["rs"], ts_param)
                    tdf = pd.DataFrame(ts_data["rows"])
                    tfig = go.Figure()
                    tfig.add_trace(go.Scatter(x=tdf["time"], y=tdf["existing"], mode="lines+markers", name=f"Existing — {existing}"))
                    tfig.add_trace(go.Scatter(x=tdf["time"], y=tdf["revised"], mode="lines+markers", name=f"Revised — {revised}"))
                    tfig.update_layout(title=f"RS {selected_xs['rs']} — {ts_param}", xaxis_title="Simulation time", yaxis_title=f"{ts_param} ({ts_data.get('unit','')})", height=460, hovermode="x unified")
                    st.plotly_chart(tfig, use_container_width=True)
                    tdfig = go.Figure(go.Scatter(x=tdf["time"], y=tdf["delta"], mode="lines+markers", name="Revised − Existing"))
                    tdfig.add_hline(y=0, line_dash="dash")
                    tdfig.update_layout(title=f"RS {selected_xs['rs']} — Δ{ts_param}", xaxis_title="Simulation time", yaxis_title=f"Δ {ts_data.get('unit','')}", height=340)
                    st.plotly_chart(tdfig, use_container_width=True)
                else:
                    st.info("No common 1D time-series datasets are available for the selected plans.")
        else:
            st.info("No common 1D cross-section summary datasets are available for the selected plans.")
    except Exception as exc:
        st.warning(f"1D plot data could not be prepared: {exc}")

with tabs[8]:
    st.markdown("### Word review summary")
    st.caption("The Word report summarizes the selected plans, model changes, reviewer flags, 2D hydraulic results, and 1D WSEL comparison. It is a screening report, not an automatic regulatory determination.")
    try:
        word_bytes = build_review_docx(report, run_history=st.session_state.get("run_history", []))
        st.download_button(
            "Download Word review summary (.docx)",
            data=word_bytes,
            file_name=f"{existing}_vs_{revised}_HECRAS_Review_Summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )
        if st.session_state.get("run_history"):
            st.markdown("#### HEC-RAS run history")
            _show_dataframe(pd.DataFrame(st.session_state["run_history"]), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Could not build Word report: {exc}")

with tabs[9]:
    st.download_button(
        "Download JSON review report",
        data=json.dumps(report, indent=2),
        file_name=f"{existing}_vs_{revised}_review_v10.json",
        mime="application/json",
    )
    st.json(report, expanded=False)
