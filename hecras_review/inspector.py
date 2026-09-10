from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .compare import (
    compare_2d_mesh,
    compare_common_cell_wse,
    compare_cross_sections,
    compare_cross_section_details,
    compare_dict_settings,
    compare_structures,
)
from .hdf_reader import (
    read_2d_cell_geometry,
    read_2d_cell_basics,
    read_2d_area_perimeters,
    read_2d_max_depth_lightweight,
    read_2d_max_depth,
    read_2d_max_face_velocity,
    read_2d_max_wse,
    read_2d_timeseries,
    read_2d_timeseries_optional,
    read_2d_summary_result_optional,
    read_2d_depth_with_fallback,
    read_hdf_inventory,
    read_map_reference_geometry,
    read_cross_section_details,
    read_time_axis,
    read_xs_max_wse,
    read_xs_summary_catalog,
    read_xs_summary_result,
    read_xs_timeseries_result,
    read_xs_identities,
)
from .source import open_model_source
from .review_flags import build_impact_summary, build_reviewer_flags
from .spatial import (
    direct_cell_delta,
    direct_face_delta,
    normalized_polygon_cell_delta,
    synchronized_cell_wse,
    top_change_points,
    top_synchronized_events,
    wet_dry_classification,
    wet_wet_delta_summary,
)
from .text_parser import important_plan_settings, parse_plan, parse_project, parse_unsteady_boundaries, boundary_identity
from .utils import sha256_bytes


def _ras_basename(project_basename: str, code: str) -> str:
    stem = os.path.splitext(project_basename)[0]
    return f"{stem}.{code}"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _same_order_delta(coordinates: np.ndarray, existing: np.ndarray, revised: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=float)
    existing = np.asarray(existing, dtype=float)
    revised = np.asarray(revised, dtype=float)
    if len(coordinates) != len(existing) or len(existing) != len(revised):
        raise ValueError("Coordinate/result length mismatch")
    return {
        "coordinates": coordinates,
        "existing": existing,
        "revised": revised,
        "delta": revised - existing,
        "existing_only_coordinates": np.empty((0, 2), dtype=float),
        "revised_only_coordinates": np.empty((0, 2), dtype=float),
        "common_count": int(len(coordinates)),
    }


def _large_mesh_compare_area(e_area: dict, r_area: dict, e_centers: np.ndarray, r_centers: np.ndarray) -> dict[str, Any]:
    """Memory-conscious mesh identity check for very large 2D areas.

    For huge changed meshes, exact common-center set construction is deferred because
    millions of Python tuple keys can consume substantially more RAM than the HDF arrays.
    """
    ec = np.asarray(e_centers, dtype=float)
    rc = np.asarray(r_centers, dtype=float)
    same_order = ec.shape == rc.shape and np.array_equal(ec, rc)
    if same_order:
        return {
            "status": "same_mesh",
            "existing_cell_count": int(len(ec)), "revised_cell_count": int(len(rc)),
            "cell_count_change": 0, "common_cell_centers": int(len(ec)),
            "existing_only_centers": 0, "revised_only_centers": 0,
            "same_cell_center_order": True, "common_center_counts_deferred": False,
        }
    # For moderately-sized changed meshes retain the exact common-center counts.
    if max(len(ec), len(rc)) <= 500_000:
        return compare_2d_mesh(
            {"_": {"cell_count": len(ec), "cell_centers": ec}},
            {"_": {"cell_count": len(rc), "cell_centers": rc}},
        )["_"]
    return {
        "status": "changed_mesh",
        "existing_cell_count": int(len(ec)), "revised_cell_count": int(len(rc)),
        "cell_count_change": int(len(rc) - len(ec)),
        "common_cell_centers": None, "existing_only_centers": None, "revised_only_centers": None,
        "same_cell_center_order": False, "common_center_counts_deferred": True,
    }


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    with open_model_source(model_path) as src:
        project_files = src.find_project_files()
        if not project_files:
            raise RuntimeError("No HEC-RAS project .prj file detected")
        project_archive = project_files[0]
        project_basename = os.path.basename(project_archive)
        project_text = src.read_text(project_archive)
        project = parse_project(project_text)
        project["archive_member"] = project_archive
        project["sha256"] = sha256_bytes(src.read_bytes(project_archive))

        plans = {}
        for code in project.get("plan_files", []):
            basename = _ras_basename(project_basename, code)
            try:
                member = src.find_sibling(project_archive, basename)
            except FileNotFoundError:
                continue
            text = src.read_text(member)
            info = parse_plan(text, code)
            info["archive_member"] = member
            info["sha256"] = sha256_bytes(src.read_bytes(member))
            info["important_settings"] = important_plan_settings(text)
            hdf_name = basename + ".hdf"
            try:
                info["hdf_archive_member"] = src.find_sibling(project_archive, hdf_name)
            except FileNotFoundError:
                info["hdf_archive_member"] = None
            plans[code] = info

        return {
            "source_model": str(model_path),
            "source_kind": "directory" if model_path.is_dir() else "zip",
            "source_zip": str(model_path) if model_path.is_file() else None,
            "source_zip_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
            "project": project,
            "plans": plans,
            "archive_file_count": len(src.names),
        }


def compare_plans(model_path: str | Path, existing_plan: str, revised_plan: str, large_model_mode: bool | None = None) -> dict[str, Any]:
    base = inspect_model(model_path)
    if existing_plan not in base["plans"]:
        raise KeyError(f"Existing plan {existing_plan} not found")
    if revised_plan not in base["plans"]:
        raise KeyError(f"Revised plan {revised_plan} not found")

    ep = base["plans"][existing_plan]
    rp = base["plans"][revised_plan]
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        raise RuntimeError("Both plans require plan HDF output for this prototype")

    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_review_") as td:
        e_hdf = src.materialize(ep["hdf_archive_member"], td)
        r_hdf = src.materialize(rp["hdf_archive_member"], td)

        project_basename = os.path.basename(base["project"]["archive_member"])
        e_flow_member = src.find_sibling(base["project"]["archive_member"], _ras_basename(project_basename, ep["flow_file"]))
        r_flow_member = src.find_sibling(base["project"]["archive_member"], _ras_basename(project_basename, rp["flow_file"]))
        e_boundaries = parse_unsteady_boundaries(src.read_text(e_flow_member))
        r_boundaries = parse_unsteady_boundaries(src.read_text(r_flow_member))
        e_bmap = {boundary_identity(x): x for x in e_boundaries}
        r_bmap = {boundary_identity(x): x for x in r_boundaries}

        combined_hdf_size_bytes = int(Path(e_hdf).stat().st_size + Path(r_hdf).stat().st_size)
        # Start with metadata-only inventory so automatic mode selection never requires
        # loading all cell-center arrays just to decide whether the model is large.
        e_inv_light = read_hdf_inventory(e_hdf, include_geometry_arrays=False)
        r_inv_light = read_hdf_inventory(r_hdf, include_geometry_arrays=False)
        e_total_cells = int(sum(int(x.get("cell_count", 0) or 0) for x in e_inv_light.get("two_d_areas", {}).values()))
        r_total_cells = int(sum(int(x.get("cell_count", 0) or 0) for x in r_inv_light.get("two_d_areas", {}).values()))
        auto_large_model_mode = bool(
            combined_hdf_size_bytes >= 750 * 1024 * 1024
            or max(e_total_cells, r_total_cells) >= 500_000
        )
        large_model_mode = auto_large_model_mode if large_model_mode is None else bool(large_model_mode)
        if large_model_mode:
            e_inv, r_inv = e_inv_light, r_inv_light
        else:
            e_inv = read_hdf_inventory(e_hdf, include_geometry_arrays=True)
            r_inv = read_hdf_inventory(r_hdf, include_geometry_arrays=True)

        if large_model_mode:
            mesh = {}
            all_areas = sorted(set(e_inv["two_d_areas"]) | set(r_inv["two_d_areas"]))
            for area in all_areas:
                if area not in e_inv["two_d_areas"]:
                    mesh[area] = {"status": "added_area", "revised_cell_count": r_inv["two_d_areas"][area]["cell_count"]}
                    continue
                if area not in r_inv["two_d_areas"]:
                    mesh[area] = {"status": "removed_area", "existing_cell_count": e_inv["two_d_areas"][area]["cell_count"]}
                    continue
                eb = read_2d_cell_basics(e_hdf, area)
                rb = read_2d_cell_basics(r_hdf, area)
                mesh[area] = _large_mesh_compare_area(e_inv["two_d_areas"][area], r_inv["two_d_areas"][area], eb["cell_centers"], rb["cell_centers"])
                del eb, rb
        else:
            mesh = compare_2d_mesh(e_inv["two_d_areas"], r_inv["two_d_areas"])
        hydraulic = {}
        spatial_summary = {}
        e_time = read_time_axis(e_hdf) if not large_model_mode else {"time_stamps": [], "time_days": np.array([])}
        r_time = read_time_axis(r_hdf) if not large_model_mode else {"time_stamps": [], "time_days": np.array([])}
        wet_depth_threshold = 0.01
        for area in sorted(set(e_inv["two_d_areas"]) & set(r_inv["two_d_areas"])):
            if large_model_mode:
                egeom = read_2d_cell_basics(e_hdf, area)
                rgeom = read_2d_cell_basics(r_hdf, area)
            else:
                egeom = read_2d_cell_geometry(e_hdf, area)
                rgeom = read_2d_cell_geometry(r_hdf, area)
            ec = egeom["cell_centers"]
            rc = rgeom["cell_centers"]

            ew = read_2d_max_wse(e_hdf, area)
            rw = read_2d_max_wse(r_hdf, area)
            huge_changed_area = bool(large_model_mode and not mesh[area].get("same_cell_center_order") and max(len(ec), len(rc)) > 500_000)
            if mesh[area].get("same_cell_center_order"):
                direct_wse = _same_order_delta(ec, ew, rw)
            elif huge_changed_area:
                direct_wse = None
            else:
                direct_wse = direct_cell_delta(ec, rc, ew, rw)

            # Large-model mode intentionally avoids full time-series reads during the
            # initial comparison. Detailed same-time analysis is deferred to a selected area.
            e_wse_ts = None if large_model_mode else read_2d_timeseries_optional(e_hdf, area, "Water Surface")
            r_wse_ts = None if large_model_mode else read_2d_timeseries_optional(r_hdf, area, "Water Surface")
            if large_model_mode:
                e_depth_info = read_2d_max_depth_lightweight(e_hdf, area, egeom["cell_min_elevation"], ew)
                r_depth_info = read_2d_max_depth_lightweight(r_hdf, area, rgeom["cell_min_elevation"], rw)
                e_depth_info["time_series"] = None
                r_depth_info["time_series"] = None
            else:
                e_depth_info = read_2d_depth_with_fallback(
                    e_hdf, area, egeom["cell_min_elevation"],
                    water_surface_ts=e_wse_ts, maximum_water_surface=ew,
                )
                r_depth_info = read_2d_depth_with_fallback(
                    r_hdf, area, rgeom["cell_min_elevation"],
                    water_surface_ts=r_wse_ts, maximum_water_surface=rw,
                )
            ed = e_depth_info["maximum"]
            rd = r_depth_info["maximum"]
            if direct_wse is None:
                direct_depth = None
                wet_mask = np.array([], dtype=bool)
                wet = {
                    "both_wet_count": 0, "became_dry_count": 0, "became_wet_count": 0, "both_dry_count": 0,
                    "classification_available": False, "deferred": True,
                    "reason": "Changed mesh is very large; exact common-cell hydraulic comparison is deferred to detailed spatial/raster review.",
                }
            elif ed is not None and rd is not None:
                if mesh[area].get("same_cell_center_order"):
                    direct_depth = _same_order_delta(ec, ed, rd)
                else:
                    direct_depth = direct_cell_delta(ec, rc, ed, rd)
                wet = wet_dry_classification(direct_depth["existing"], direct_depth["revised"], wet_depth_threshold)
                wet_mask = wet["both_wet_mask"]
            else:
                # Last-resort compatibility mode: preserve WSEL comparison but do not
                # pretend that wet/dry filtering was possible.
                direct_depth = {
                    "coordinates": direct_wse["coordinates"],
                    "existing": np.full(len(direct_wse["coordinates"]), np.nan),
                    "revised": np.full(len(direct_wse["coordinates"]), np.nan),
                    "delta": np.full(len(direct_wse["coordinates"]), np.nan),
                    "existing_only_coordinates": direct_wse.get("existing_only_coordinates", np.empty((0,2))),
                    "revised_only_coordinates": direct_wse.get("revised_only_coordinates", np.empty((0,2))),
                    "common_count": direct_wse.get("common_count", 0),
                }
                wet_mask = np.isfinite(direct_wse["delta"])
                wet = {
                    "both_wet_mask": wet_mask, "became_dry_mask": np.zeros_like(wet_mask, dtype=bool),
                    "became_wet_mask": np.zeros_like(wet_mask, dtype=bool), "both_dry_mask": np.zeros_like(wet_mask, dtype=bool),
                    "both_wet_count": int(wet_mask.sum()), "became_dry_count": 0, "became_wet_count": 0,
                    "both_dry_count": 0, "classification_available": False,
                }

            if direct_wse is None:
                hydraulic[area] = {
                    "comparison_method": "deferred_large_changed_mesh",
                    "raw_max_wse_delta": {"available": False, "deferred": True},
                    "max_wse_delta_both_wet": {"available": False, "deferred": True},
                    "wet_dry_transitions": {k:v for k,v in wet.items() if not k.endswith("_mask")},
                    "max_wse_top_changes_both_wet": {"largest_increases": [], "largest_decreases": []},
                    "max_depth_delta": {"available": False, "deferred": True},
                    "depth_data_source": {
                        "existing": e_depth_info["source"], "revised": r_depth_info["source"],
                        "uses_proxy": bool(e_depth_info.get("is_proxy") or r_depth_info.get("is_proxy")),
                    },
                }
            else:
                hydraulic[area] = {
                    "comparison_method": (
                        "direct_same_order_cell" if mesh[area].get("same_cell_center_order") else "exact_common_cell_center_only"
                    ),
                    "raw_max_wse_delta": {k:v for k,v in direct_wse.items() if k not in {
                        "coordinates", "existing", "revised", "delta", "existing_only_coordinates", "revised_only_coordinates"
                    }},
                    "max_wse_delta_both_wet": wet_wet_delta_summary(
                        direct_wse["delta"], direct_depth["existing"], direct_depth["revised"], wet_depth_threshold
                    ),
                    "wet_dry_transitions": {k:v for k,v in wet.items() if not k.endswith("_mask")},
                    "max_wse_top_changes_both_wet": top_change_points(
                        direct_wse["coordinates"][wet_mask], direct_wse["delta"][wet_mask], n=10
                    ),
                    "max_depth_delta": {k:v for k,v in direct_depth.items() if k not in {
                        "coordinates", "existing", "revised", "delta", "existing_only_coordinates", "revised_only_coordinates"
                    }},
                    "depth_data_source": {
                        "existing": e_depth_info["source"], "revised": r_depth_info["source"],
                        "uses_proxy": bool(e_depth_info.get("is_proxy") or r_depth_info.get("is_proxy")),
                    },
                }

            # Velocity is face-based and optional in some HEC-RAS output configurations.
            if large_model_mode:
                hydraulic[area]["max_face_velocity_delta"] = {
                    "available": False, "deferred": True,
                    "reason": "Face-level velocity geometry comparison is deferred in large-model mode; use the selected-area Spatial Review or Raster Map.",
                }
            else:
                ev = read_2d_summary_result_optional(e_hdf, area, "Maximum Face Velocity", value_row=0)
                rv = read_2d_summary_result_optional(r_hdf, area, "Maximum Face Velocity", value_row=0)
                if ev is not None and rv is not None:
                    face_vel = direct_face_delta(
                        egeom["facepoint_coordinates"], egeom["faces_facepoint_indexes"],
                        rgeom["facepoint_coordinates"], rgeom["faces_facepoint_indexes"], ev, rv,
                    )
                    hydraulic[area]["max_face_velocity_delta"] = {k:v for k,v in face_vel.items() if k not in {"coordinates", "delta"}}
                else:
                    hydraulic[area]["max_face_velocity_delta"] = {
                        "available": False,
                        "reason": "Maximum Face Velocity was not written for both selected plans.",
                    }

            # Same-time WSEL is optional. When available, use native or derived depth
            # time series to mask timestamps where both models are wet.
            e_depth_ts = e_depth_info.get("time_series")
            r_depth_ts = r_depth_info.get("time_series")
            if large_model_mode:
                hydraulic[area]["same_time_wse_both_wet"] = {
                    "compatible": False, "deferred": True,
                    "reason": "Full Water Surface time-series comparison is deferred in large-model mode; open a selected area for detailed analysis.",
                }
                hydraulic[area]["top_same_time_wse_events"] = []
            elif e_wse_ts is not None and r_wse_ts is not None:
                sync = synchronized_cell_wse(
                    ec, rc, e_wse_ts, r_wse_ts, e_time["time_stamps"], r_time["time_stamps"],
                    existing_depth_ts=e_depth_ts, revised_depth_ts=r_depth_ts,
                    wet_depth_threshold=wet_depth_threshold,
                )
                hydraulic[area]["same_time_wse_both_wet"] = {k:v for k,v in sync.items() if k not in {
                    "coordinates", "max_same_time_increase", "max_same_time_decrease", "peak_absolute_same_time_delta", "peak_time_index", "time_stamps"
                }}
                hydraulic[area]["top_same_time_wse_events"] = top_synchronized_events(sync, n=10)
            else:
                hydraulic[area]["same_time_wse_both_wet"] = {
                    "compatible": False,
                    "reason": "Water Surface time-series output was not written for both selected plans.",
                }
                hydraulic[area]["top_same_time_wse_events"] = []

            spatial_summary[area] = {
                "existing_only_center_count": int(len(direct_wse.get("existing_only_coordinates", []))) if direct_wse is not None else None,
                "revised_only_center_count": int(len(direct_wse.get("revised_only_coordinates", []))) if direct_wse is not None else None,
                "direct_wet_dry_transitions": {k:v for k,v in wet.items() if not k.endswith("_mask")},
            }
            if (not large_model_mode) and mesh[area].get("status") == "changed_mesh":
                spacing = float(e_inv["two_d_areas"][area].get("spacing_dx") or 100.0)
                normalized = normalized_polygon_cell_delta(
                    ew, rw,
                    egeom["cell_facepoint_indexes"], egeom["facepoint_coordinates"],
                    rgeom["cell_facepoint_indexes"], rgeom["facepoint_coordinates"],
                    egeom["perimeter"], rgeom["perimeter"], grid_spacing=spacing
                )
                if normalized.get("point_count", 0):
                    n_ed = ed[np.asarray(normalized["existing_cell_indexes"], dtype=int)]
                    n_rd = rd[np.asarray(normalized["revised_cell_indexes"], dtype=int)]
                    n_wet = wet_dry_classification(n_ed, n_rd, wet_depth_threshold)
                    nmask = n_wet["both_wet_mask"]
                    spatial_summary[area]["normalized_max_wse_both_wet"] = {
                        "method": normalized["method"],
                        "point_count": normalized["point_count"],
                        "grid_spacing_ft": normalized["grid_spacing_ft"],
                        **wet_wet_delta_summary(normalized["delta"], n_ed, n_rd, wet_depth_threshold),
                    }
                    spatial_summary[area]["normalized_wet_dry_transitions"] = {
                        k:v for k,v in n_wet.items() if not k.endswith("_mask")
                    }
                    spatial_summary[area]["normalized_top_changes_both_wet"] = top_change_points(
                        normalized["coordinates"][nmask], normalized["delta"][nmask], n=10
                    )

        xs_compare = compare_cross_sections(e_inv["cross_sections"], r_inv["cross_sections"])
        xs_detailed = compare_cross_section_details(read_cross_section_details(e_hdf), read_cross_section_details(r_hdf))
        e_xs_wse = read_xs_max_wse(e_hdf)
        r_xs_wse = read_xs_max_wse(r_hdf)
        # Geometry order is checked via river/reach/RS. Only calculate index-wise if identities match in order.
        e_ids = [(str(x.get("River","")).strip(), str(x.get("Reach","")).strip(), str(x.get("RS","")).strip()) for x in e_inv["cross_sections"].get("items",[])]
        r_ids = [(str(x.get("River","")).strip(), str(x.get("Reach","")).strip(), str(x.get("RS","")).strip()) for x in r_inv["cross_sections"].get("items",[])]
        e_xs_available = e_xs_wse is not None
        r_xs_available = r_xs_wse is not None
        xs_hyd = {
            "dataset": "Maximum Water Surface",
            "index_order_identical": e_ids == r_ids,
            "existing_available": e_xs_available,
            "revised_available": r_xs_available,
            "available": bool(e_xs_available and r_xs_available),
        }
        if not e_xs_available or not r_xs_available:
            missing = []
            if not e_xs_available:
                missing.append("Existing")
            if not r_xs_available:
                missing.append("Revised")
            xs_hyd.update({
                "count": 0,
                "reason": (
                    f"1D Maximum Water Surface summary output is not available for {' and '.join(missing)} plan(s). "
                    "The comparison continued using the available geometry, structures, 2D results, and other saved outputs."
                ),
                "top_10_absolute_changes": [],
            })
        elif e_ids != r_ids:
            xs_hyd.update({
                "count": 0,
                "reason": "1D cross-section order/identity differs between plans; index-wise Maximum WSEL comparison was skipped.",
                "top_10_absolute_changes": [],
            })
        elif len(e_xs_wse) != len(r_xs_wse) or len(e_xs_wse) != len(e_ids):
            xs_hyd.update({
                "count": 0,
                "reason": (
                    "1D Maximum Water Surface array length does not match the common cross-section geometry count; "
                    "index-wise hydraulic comparison was skipped."
                ),
                "top_10_absolute_changes": [],
            })
        elif len(e_xs_wse):
            d = r_xs_wse - e_xs_wse
            ad = np.abs(d)
            xs_hyd.update({
                "count": int(d.size),
                "min_delta_ft": float(np.nanmin(d)),
                "max_delta_ft": float(np.nanmax(d)),
                "count_abs_gt_0_001_ft": int(np.sum(ad > 0.001)),
                "count_abs_gt_0_01_ft": int(np.sum(ad > 0.01)),
                "count_abs_gt_0_1_ft": int(np.sum(ad > 0.1)),
            })
            order = np.argsort(np.abs(d))[::-1][:10]
            xs_hyd["top_10_absolute_changes"] = [
                {
                    "river": e_ids[i][0], "reach": e_ids[i][1], "rs": e_ids[i][2],
                    "existing_max_wse_ft": float(e_xs_wse[i]),
                    "revised_max_wse_ft": float(r_xs_wse[i]),
                    "delta_ft": float(d[i]),
                } for i in order
            ]

        report = {
            "review_pair": {
                "existing": {k: ep.get(k) for k in ["plan_code","plan_title","program_version","geometry_file","flow_file","simulation_date","computation_interval","output_interval","mapping_interval"]},
                "revised": {k: rp.get(k) for k in ["plan_code","plan_title","program_version","geometry_file","flow_file","simulation_date","computation_interval","output_interval","mapping_interval"]},
            },
            "project": base["project"],
            "plan_setting_changes": compare_dict_settings(ep.get("important_settings", {}), rp.get("important_settings", {})),
            "hdf_metadata": {
                "existing": {k: e_inv.get(k) for k in ["file_version","units","projection_wkt"]},
                "revised": {k: r_inv.get(k) for k in ["file_version","units","projection_wkt"]},
            },
            "performance": {
                "large_model_mode": bool(large_model_mode),
                "combined_plan_hdf_size_bytes": combined_hdf_size_bytes,
                "existing_total_2d_cells": e_total_cells,
                "revised_total_2d_cells": r_total_cells,
                "initial_comparison_strategy": "summary_first_deferred_heavy_spatial" if large_model_mode else "full_small_model",
            },
            "mesh_comparison": mesh,
            "cross_section_comparison": xs_compare,
            "cross_section_detailed_comparison": xs_detailed,
            "structure_comparison": compare_structures(e_inv["structures"], r_inv["structures"]),
            "boundary_condition_lines": {
                "existing": e_inv["boundary_condition_lines"],
                "revised": r_inv["boundary_condition_lines"],
            },
            "unsteady_boundary_comparison": {
                "existing_count": len(e_boundaries),
                "revised_count": len(r_boundaries),
                "added_locations": [r_bmap[k] for k in sorted(set(r_bmap) - set(e_bmap))],
                "removed_locations": [e_bmap[k] for k in sorted(set(e_bmap) - set(r_bmap))],
                "common_location_parameter_changes": [
                    {
                        "identity": list(k),
                        "existing_parameters": e_bmap[k]["parameters"],
                        "revised_parameters": r_bmap[k]["parameters"],
                    }
                    for k in sorted(set(e_bmap) & set(r_bmap))
                    if e_bmap[k]["parameters"] != r_bmap[k]["parameters"]
                ],
            },
            "breaklines": {
                "existing_count": len(e_inv["breaklines"]),
                "revised_count": len(r_inv["breaklines"]),
            },
            "available_results": {
                "existing": e_inv["results"],
                "revised": r_inv["results"],
            },
            "hydraulic_comparison": {
                "two_d": hydraulic,
                "cross_section_max_wse": xs_hyd,
            },
            "spatial_comparison": spatial_summary,
            "limitations_v0_5": [
                "Exact common-cell comparisons are direct. Changed-mesh normalization uses common analysis points assigned to the containing HEC-RAS cell polygon in each model and is explicitly a spatial comparison, not cell equivalence.",
                "Face velocity is compared only where face endpoint geometry matches exactly; full face-field normalization is a later refinement.",
                "Detailed 1D cross-section comparison now covers station/elevation geometry, bank/length/contraction-expansion attributes, Manning's n, ineffective blocks, and mapped XS polylines; additional specialized 1D options remain for later versions.",
                "Maximum-vs-maximum WSEL and same-time WSEL are reported separately because they answer different review questions.",
            ],
        }
        report["impact_summary"] = build_impact_summary(report)
        report["reviewer_flags"] = build_reviewer_flags(report)
        return _json_safe(report)



def available_1d_results(model_path: str | Path, existing_plan: str, revised_plan: str) -> dict[str, Any]:
    """List common 1D cross-section summary and time-series datasets for two plans."""
    base = inspect_model(model_path)
    ep = base["plans"][existing_plan]
    rp = base["plans"][revised_plan]
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        return {"summary": {}, "time_series": []}
    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_1dcat_") as td:
        e_hdf = src.materialize(ep["hdf_archive_member"], td)
        r_hdf = src.materialize(rp["hdf_archive_member"], td)
        ec = read_xs_summary_catalog(e_hdf)
        rc = read_xs_summary_catalog(r_hdf)
        common_summary = {k: ec[k] for k in ec.keys() & rc.keys()}
        ei = read_hdf_inventory(e_hdf)
        ri = read_hdf_inventory(r_hdf)
        e_ts = set(ei.get("results", {}).get("cross_section_time_series", []))
        r_ts = set(ri.get("results", {}).get("cross_section_time_series", []))
        return {"summary": dict(sorted(common_summary.items())), "time_series": sorted(e_ts & r_ts)}


def one_d_profile_review(model_path: str | Path, existing_plan: str, revised_plan: str,
                         dataset: str = "Maximum Water Surface") -> dict[str, Any]:
    """Return aligned Existing/Revised 1D longitudinal profile data for one summary result."""
    base = inspect_model(model_path)
    ep = base["plans"][existing_plan]
    rp = base["plans"][revised_plan]
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        raise RuntimeError("Both selected plans require HDF output for 1D profile comparison")
    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_1dprof_") as td:
        e_hdf = src.materialize(ep["hdf_archive_member"], td)
        r_hdf = src.materialize(rp["hdf_archive_member"], td)
        e_ids = read_xs_identities(e_hdf)
        r_ids = read_xs_identities(r_hdf)
        ev = read_xs_summary_result(e_hdf, dataset)
        rv = read_xs_summary_result(r_hdf, dataset)
        e_map = {k: float(ev["values"][i]) for i, k in enumerate(e_ids) if i < len(ev["values"])}
        r_map = {k: float(rv["values"][i]) for i, k in enumerate(r_ids) if i < len(rv["values"])}
        common = sorted(set(e_map) & set(r_map), key=lambda k: (k[0], k[1], -_rs_number(k[2])))
        rows = []
        for river, reach, rs in common:
            e = e_map[(river, reach, rs)]
            r = r_map[(river, reach, rs)]
            rows.append({
                "river": river, "reach": reach, "rs": rs, "rs_numeric": _rs_number(rs),
                "existing": e, "revised": r, "delta": r - e,
            })
        return {
            "dataset": dataset,
            "label": ev.get("label") or dataset,
            "unit": ev.get("unit") or rv.get("unit") or "",
            "existing_plan": existing_plan,
            "revised_plan": revised_plan,
            "rows": rows,
            "river_reaches": sorted({(r["river"], r["reach"]) for r in rows}),
        }


def one_d_timeseries_review(model_path: str | Path, existing_plan: str, revised_plan: str,
                            river: str, reach: str, rs: str,
                            dataset: str = "Water Surface") -> dict[str, Any]:
    """Return synchronized 1D time-series values at a selected cross section."""
    base = inspect_model(model_path)
    ep = base["plans"][existing_plan]
    rp = base["plans"][revised_plan]
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        raise RuntimeError("Both selected plans require HDF output for 1D time-series comparison")
    key = (river, reach, rs)
    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_1dts_") as td:
        e_hdf = src.materialize(ep["hdf_archive_member"], td)
        r_hdf = src.materialize(rp["hdf_archive_member"], td)
        e_ids = read_xs_identities(e_hdf)
        r_ids = read_xs_identities(r_hdf)
        if key not in e_ids or key not in r_ids:
            raise KeyError(f"Cross section {key} is not common to both plans")
        e_idx, r_idx = e_ids.index(key), r_ids.index(key)
        e_ts = read_xs_timeseries_result(e_hdf, dataset)
        r_ts = read_xs_timeseries_result(r_hdf, dataset)
        et = read_time_axis(e_hdf)
        rt = read_time_axis(r_hdf)
        n = min(len(et["time_stamps"]), len(rt["time_stamps"]), e_ts.shape[0], r_ts.shape[0])
        stamps_match = et["time_stamps"][:n] == rt["time_stamps"][:n]
        rows = []
        for i in range(n):
            e = float(e_ts[i, e_idx])
            r = float(r_ts[i, r_idx])
            rows.append({"time": et["time_stamps"][i], "existing": e, "revised": r, "delta": r - e})
        units = {"Water Surface": "ft", "Velocity Channel": "ft/s", "Velocity Total": "ft/s", "Flow": "cfs", "Flow Lateral": "cfs"}
        return {
            "dataset": dataset, "unit": units.get(dataset, ""), "river": river, "reach": reach, "rs": rs,
            "timestamps_identical": bool(stamps_match), "rows": rows,
        }


def _rs_number(value: str) -> float:
    try:
        return float(str(value).replace("*", "").strip())
    except Exception:
        return float("-inf")

def save_json(report: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return output_path



def map_reference_geometry(model_path: str | Path, plan_code: str) -> dict[str, Any]:
    """Return georeference-ready line features for a selected plan HDF."""
    base = inspect_model(model_path)
    if plan_code not in base["plans"]:
        raise KeyError(f"Unknown plan {plan_code!r}")
    plan = base["plans"][plan_code]
    member = plan.get("hdf_archive_member")
    if not member:
        raise RuntimeError(f"Plan {plan_code} has no HDF output")
    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_mapref_") as td:
        hdf = src.materialize(member, td)
        return read_map_reference_geometry(hdf)



def map_2d_area_perimeters(model_path: str | Path, plan_code: str) -> dict[str, Any]:
    """Return projection and all 2D flow-area perimeters for one plan."""
    base = inspect_model(model_path)
    plan = base["plans"][plan_code]
    member = plan.get("hdf_archive_member")
    if not member:
        return {"projection_wkt": "", "areas": {}}
    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_2dper_" ) as td:
        hdf = src.materialize(member, td)
        return read_2d_area_perimeters(hdf)

def spatial_area_review(model_path: str | Path, existing_plan: str, revised_plan: str,
                        area_name: str, grid_spacing_ft: float | None = None,
                        include_time_series: bool = True,
                        include_normalized: bool = True,
                        include_velocity: bool = True) -> dict[str, Any]:
    """Return in-memory arrays for an interactive spatial review of one 2D area.

    Heavy time-series, changed-mesh normalization, and face-velocity reads can be
    disabled for combined maps and very large models.
    """
    base = inspect_model(model_path)
    ep = base["plans"][existing_plan]
    rp = base["plans"][revised_plan]
    if not ep.get("hdf_archive_member") or not rp.get("hdf_archive_member"):
        raise RuntimeError("Both plans require plan HDF output")

    with open_model_source(Path(model_path)) as src, tempfile.TemporaryDirectory(prefix="hecras_spatial_") as td:
        e_hdf = src.materialize(ep["hdf_archive_member"], td)
        r_hdf = src.materialize(rp["hdf_archive_member"], td)
        e_inv = read_hdf_inventory(e_hdf)
        r_inv = read_hdf_inventory(r_hdf)
        if area_name not in e_inv["two_d_areas"] or area_name not in r_inv["two_d_areas"]:
            raise KeyError(f"2D area {area_name!r} is not present in both plans")

        need_full_geometry = bool(include_velocity or include_normalized)
        eg = read_2d_cell_geometry(e_hdf, area_name) if need_full_geometry else read_2d_cell_basics(e_hdf, area_name, include_perimeter=True)
        rg = read_2d_cell_geometry(r_hdf, area_name) if need_full_geometry else read_2d_cell_basics(r_hdf, area_name, include_perimeter=True)
        ew = read_2d_max_wse(e_hdf, area_name)
        rw = read_2d_max_wse(r_hdf, area_name)
        e_wse_ts = read_2d_timeseries_optional(e_hdf, area_name, "Water Surface") if include_time_series else None
        r_wse_ts = read_2d_timeseries_optional(r_hdf, area_name, "Water Surface") if include_time_series else None
        if include_time_series:
            e_depth_info = read_2d_depth_with_fallback(
                e_hdf, area_name, eg["cell_min_elevation"], water_surface_ts=e_wse_ts, maximum_water_surface=ew,
            )
            r_depth_info = read_2d_depth_with_fallback(
                r_hdf, area_name, rg["cell_min_elevation"], water_surface_ts=r_wse_ts, maximum_water_surface=rw,
            )
        else:
            e_depth_info = read_2d_max_depth_lightweight(e_hdf, area_name, eg["cell_min_elevation"], ew)
            r_depth_info = read_2d_max_depth_lightweight(r_hdf, area_name, rg["cell_min_elevation"], rw)
            e_depth_info["time_series"] = None
            r_depth_info["time_series"] = None
        ed = e_depth_info["maximum"]
        rd = r_depth_info["maximum"]
        wet_depth_threshold = 0.01

        same_order = (
            np.asarray(eg["cell_centers"]).shape == np.asarray(rg["cell_centers"]).shape
            and np.array_equal(np.asarray(eg["cell_centers"]), np.asarray(rg["cell_centers"]))
        )
        huge_changed = bool((not same_order) and max(len(eg["cell_centers"]), len(rg["cell_centers"])) > 500_000)
        if same_order:
            wse = _same_order_delta(eg["cell_centers"], ew, rw)
        elif huge_changed and not include_normalized:
            wse = {
                "coordinates": np.empty((0,2)), "existing": np.array([]), "revised": np.array([]), "delta": np.array([]),
                "existing_only_coordinates": np.empty((0,2)), "revised_only_coordinates": np.empty((0,2)),
                "common_count": 0, "deferred": True,
                "reason": "Exact common-cell matching deferred for very large changed mesh in lightweight view.",
            }
        else:
            wse = direct_cell_delta(eg["cell_centers"], rg["cell_centers"], ew, rw)
        if ed is not None and rd is not None and not wse.get("deferred"):
            depth = _same_order_delta(eg["cell_centers"], ed, rd) if same_order else direct_cell_delta(eg["cell_centers"], rg["cell_centers"], ed, rd)
            wet = wet_dry_classification(depth["existing"], depth["revised"], wet_depth_threshold)
        elif wse.get("deferred"):
            depth = {"coordinates": np.empty((0,2)), "existing": np.array([]), "revised": np.array([]), "delta": np.array([]), "common_count": 0, "deferred": True}
            wet = {
                "both_wet_mask": np.array([], dtype=bool), "became_dry_mask": np.array([], dtype=bool),
                "became_wet_mask": np.array([], dtype=bool), "both_dry_mask": np.array([], dtype=bool),
                "both_wet_count": 0, "became_dry_count": 0, "became_wet_count": 0, "both_dry_count": 0,
                "classification_available": False, "deferred": True,
            }
        else:
            depth = {
                "coordinates": wse["coordinates"],
                "existing": np.full(len(wse["coordinates"]), np.nan),
                "revised": np.full(len(wse["coordinates"]), np.nan),
                "delta": np.full(len(wse["coordinates"]), np.nan),
                "existing_only_coordinates": wse.get("existing_only_coordinates", np.empty((0,2))),
                "revised_only_coordinates": wse.get("revised_only_coordinates", np.empty((0,2))),
                "common_count": wse.get("common_count", 0),
            }
            finite = np.isfinite(wse["delta"])
            wet = {
                "both_wet_mask": finite, "became_dry_mask": np.zeros_like(finite, dtype=bool),
                "became_wet_mask": np.zeros_like(finite, dtype=bool), "both_dry_mask": np.zeros_like(finite, dtype=bool),
                "both_wet_count": int(finite.sum()), "became_dry_count": 0,
                "became_wet_count": 0, "both_dry_count": 0, "classification_available": False,
            }

        if include_velocity:
            ev = read_2d_summary_result_optional(e_hdf, area_name, "Maximum Face Velocity", value_row=0)
            rv = read_2d_summary_result_optional(r_hdf, area_name, "Maximum Face Velocity", value_row=0)
            if ev is not None and rv is not None:
                velocity = direct_face_delta(
                    eg["facepoint_coordinates"], eg["faces_facepoint_indexes"],
                    rg["facepoint_coordinates"], rg["faces_facepoint_indexes"], ev, rv,
                )
            else:
                velocity = {"coordinates": np.empty((0,2)), "delta": np.array([]), "available": False}
        else:
            velocity = {"coordinates": np.empty((0,2)), "delta": np.array([]), "available": False, "deferred": True}

        e_time = read_time_axis(e_hdf) if include_time_series else {"time_stamps": []}
        r_time = read_time_axis(r_hdf) if include_time_series else {"time_stamps": []}
        if include_time_series and e_wse_ts is not None and r_wse_ts is not None:
            sync = synchronized_cell_wse(
                eg["cell_centers"], rg["cell_centers"], e_wse_ts, r_wse_ts,
                e_time["time_stamps"], r_time["time_stamps"],
                existing_depth_ts=e_depth_info.get("time_series"), revised_depth_ts=r_depth_info.get("time_series"),
                wet_depth_threshold=wet_depth_threshold,
            )
        else:
            sync = {
                "coordinates": np.empty((0,2)), "peak_absolute_same_time_delta": np.array([]),
                "peak_time_index": np.array([], dtype=int), "time_stamps": [],
                "compatible": False, "reason": (
                    "Time-series analysis was deferred for this view." if not include_time_series
                    else "Water Surface time-series output is unavailable for one or both plans."
                ),
            }
        spacing = float(grid_spacing_ft or e_inv["two_d_areas"][area_name].get("spacing_dx") or 100.0)
        if include_normalized:
            normalized = normalized_polygon_cell_delta(
                ew, rw,
                eg["cell_facepoint_indexes"], eg["facepoint_coordinates"],
                rg["cell_facepoint_indexes"], rg["facepoint_coordinates"],
                eg["perimeter"], rg["perimeter"], grid_spacing=spacing,
            )
        else:
            normalized = {"coordinates": np.empty((0,2)), "delta": np.array([]), "point_count": 0, "deferred": True}
        nwet = None
        if include_normalized and ed is not None and rd is not None and normalized.get("point_count", 0):
            n_ed = ed[np.asarray(normalized["existing_cell_indexes"], dtype=int)]
            n_rd = rd[np.asarray(normalized["revised_cell_indexes"], dtype=int)]
            nwet = wet_dry_classification(n_ed, n_rd, wet_depth_threshold)

        return {
            "area_name": area_name,
            "grid_spacing_ft": spacing,
            "projection_wkt": e_inv.get("projection_wkt") or r_inv.get("projection_wkt"),
            "existing_perimeter": eg["perimeter"],
            "revised_perimeter": rg["perimeter"],
            "direct_max_wse": wse,
            "direct_max_depth": depth,
            "depth_data_source": {
                "existing": e_depth_info["source"], "revised": r_depth_info["source"],
                "uses_proxy": bool(e_depth_info.get("is_proxy") or r_depth_info.get("is_proxy")),
            },
            "direct_wet_dry": wet,
            "direct_max_face_velocity": velocity,
            "same_time_wse": sync,
            "normalized_max_wse": normalized,
            "normalized_wet_dry": nwet,
        }


def inspect_zip(zip_path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for ZIP-only callers."""
    return inspect_model(zip_path)
