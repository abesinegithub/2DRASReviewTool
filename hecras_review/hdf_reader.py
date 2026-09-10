from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .utils import decode, records_to_dicts

SUMMARY_2D = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas"
SUMMARY_XS = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/Cross Sections"
TS_ROOT = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
TS_2D = f"{TS_ROOT}/2D Flow Areas"
TS_XS = f"{TS_ROOT}/Cross Sections"


def _dataset_names(group: h5py.Group) -> list[str]:
    return [k for k, v in group.items() if isinstance(v, h5py.Dataset)]


def read_hdf_inventory(path: str | Path, include_geometry_arrays: bool = True) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as h:
        root_attrs = {k: decode(v) for k, v in h.attrs.items()}
        inv: dict[str, Any] = {
            "file_version": root_attrs.get("File Version"),
            "units": root_attrs.get("Units System"),
            "projection_wkt": root_attrs.get("Projection"),
            "two_d_areas": {},
            "cross_sections": {},
            "structures": {},
            "boundary_condition_lines": [],
            "breaklines": [],
            "results": {"two_d_areas": {}, "cross_sections": []},
        }

        area_root = h.get("Geometry/2D Flow Areas")
        if area_root is not None:
            attrs_ds = area_root.get("Attributes")
            area_attrs = records_to_dicts(attrs_ds[()]) if attrs_ds is not None else []
            for a in area_attrs:
                name = str(a.get("Name", "")).strip()
                if not name:
                    continue
                group = area_root.get(name)
                item = {
                    "name": name,
                    "declared_cell_count": int(a.get("Cell Count", 0)),
                    "spacing_dx": a.get("Spacing dx"),
                    "spacing_dy": a.get("Spacing dy"),
                    "mann": a.get("Mann"),
                }
                if group is not None:
                    for ds_name, out_name in [
                        ("Cells Center Coordinate", "cell_centers"),
                        ("Cells Minimum Elevation", "cell_min_elevation"),
                        ("Cells Surface Area", "cell_surface_area"),
                        ("Perimeter", "perimeter"),
                    ]:
                        ds = group.get(ds_name)
                        if ds is not None:
                            item[out_name + "_shape"] = list(ds.shape)
                            if include_geometry_arrays and ds_name in {"Cells Center Coordinate", "Perimeter"}:
                                item[out_name] = ds[()]
                    centers = group.get("Cells Center Coordinate")
                    item["cell_count"] = int(centers.shape[0]) if centers is not None else int(a.get("Cell Count", 0))
                    faces = group.get("Faces FacePoint Indexes")
                    item["face_count"] = int(faces.shape[0]) if faces is not None else 0
                inv["two_d_areas"][name] = item

        xs_attrs = h.get("Geometry/Cross Sections/Attributes")
        if xs_attrs is not None:
            xs = records_to_dicts(xs_attrs[()])
            inv["cross_sections"] = {"count": len(xs), "items": xs}

        structs = h.get("Geometry/Structures/Attributes")
        culverts = h.get("Geometry/Structures/Culvert Groups/Attributes")
        inv["structures"] = {
            "count": int(structs.shape[0]) if structs is not None else 0,
            "items": records_to_dicts(structs[()]) if structs is not None else [],
            "culvert_group_count": int(culverts.shape[0]) if culverts is not None else 0,
            "culvert_groups": records_to_dicts(culverts[()]) if culverts is not None else [],
        }

        bc = h.get("Geometry/Boundary Condition Lines/Attributes")
        if bc is not None:
            inv["boundary_condition_lines"] = records_to_dicts(bc[()])

        bl = h.get("Geometry/2D Flow Area Break Lines/Attributes")
        if bl is not None:
            inv["breaklines"] = records_to_dicts(bl[()])

        if SUMMARY_2D in h:
            for area_name, area_group in h[SUMMARY_2D].items():
                inv["results"]["two_d_areas"][area_name] = {
                    "summary_datasets": _dataset_names(area_group),
                    "time_series_datasets": _dataset_names(h[f"{TS_2D}/{area_name}"]) if f"{TS_2D}/{area_name}" in h else [],
                }
        if SUMMARY_XS in h:
            inv["results"]["cross_sections"] = _dataset_names(h[SUMMARY_XS])
        inv["results"]["cross_section_time_series"] = _dataset_names(h[TS_XS]) if TS_XS in h else []

        return inv


def read_2d_area_perimeters(path: str | Path) -> dict[str, Any]:
    """Read only 2D flow-area perimeters and projection for lightweight overview maps."""
    with h5py.File(path, "r") as h:
        projection = decode(h.attrs.get("Projection", ""))
        root = h.get("Geometry/2D Flow Areas")
        areas: dict[str, np.ndarray] = {}
        if root is not None:
            for name, obj in root.items():
                if not isinstance(obj, h5py.Group):
                    continue
                ds = obj.get("Perimeter")
                if ds is not None:
                    areas[str(name)] = np.asarray(ds[()], dtype=float)
        return {"projection_wkt": projection, "areas": areas}


def read_2d_cell_basics(path: str | Path, area_name: str, include_perimeter: bool = False) -> dict[str, np.ndarray]:
    """Read only the cell arrays needed for lightweight large-model comparison.

    Avoids loading face connectivity and face-point geometry, which can be very large
    for MAAPnext-scale 2D models.
    """
    root = f"Geometry/2D Flow Areas/{area_name}"
    with h5py.File(path, "r") as h:
        g = h[root]
        out = {
            "cell_centers": np.asarray(g["Cells Center Coordinate"][()], dtype=float),
            "cell_min_elevation": np.asarray(g["Cells Minimum Elevation"][()], dtype=float),
        }
        if include_perimeter and "Perimeter" in g:
            out["perimeter"] = np.asarray(g["Perimeter"][()], dtype=float)
        return out


def read_2d_max_depth_lightweight(
    path: str | Path,
    area_name: str,
    cell_min_elevation: np.ndarray,
    maximum_water_surface: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return maximum depth without reading full unsteady time-series arrays.

    Priority is native Maximum Depth summary output, then a clearly-labelled proxy
    derived from Maximum Water Surface minus Cell Minimum Elevation.
    """
    native_max = read_2d_summary_result_optional(path, area_name, "Maximum Depth")
    if native_max is not None:
        return {"maximum": native_max, "source": "native_maximum_depth_summary", "is_proxy": False}
    if maximum_water_surface is None:
        maximum_water_surface = read_2d_summary_result_optional(path, area_name, "Maximum Water Surface")
    cell_min = np.asarray(cell_min_elevation, dtype=float).reshape(-1)
    if maximum_water_surface is not None and len(maximum_water_surface) == len(cell_min):
        proxy = np.maximum(np.asarray(maximum_water_surface, dtype=float) - cell_min, 0.0)
        return {
            "maximum": proxy,
            "source": "derived_maximum_water_surface_minus_cell_minimum_elevation",
            "is_proxy": True,
        }
    return {"maximum": None, "source": "unavailable", "is_proxy": True}


def read_2d_cell_geometry(path: str | Path, area_name: str) -> dict[str, np.ndarray]:
    root = f"Geometry/2D Flow Areas/{area_name}"
    with h5py.File(path, "r") as h:
        g = h[root]
        return {
            "cell_centers": np.asarray(g["Cells Center Coordinate"][()], dtype=float),
            "cell_min_elevation": np.asarray(g["Cells Minimum Elevation"][()], dtype=float),
            "cell_surface_area": np.asarray(g["Cells Surface Area"][()], dtype=float),
            "cell_facepoint_indexes": np.asarray(g["Cells FacePoint Indexes"][()], dtype=int),
            "facepoint_coordinates": np.asarray(g["FacePoints Coordinate"][()], dtype=float),
            "faces_facepoint_indexes": np.asarray(g["Faces FacePoint Indexes"][()], dtype=int),
            "faces_cell_indexes": np.asarray(g["Faces Cell Indexes"][()], dtype=int),
            "perimeter": np.asarray(g["Perimeter"][()], dtype=float),
        }


def read_2d_max_wse(path: str | Path, area_name: str) -> np.ndarray:
    return read_2d_summary_result(path, area_name, "Maximum Water Surface", value_row=0)


def read_2d_summary_result(path: str | Path, area_name: str, dataset: str, value_row: int = 0) -> np.ndarray:
    ds_path = f"{SUMMARY_2D}/{area_name}/{dataset}"
    with h5py.File(path, "r") as h:
        data = np.asarray(h[ds_path][()])
        if data.ndim == 2:
            return np.asarray(data[value_row], dtype=float)
        return np.asarray(data, dtype=float)




def hdf_dataset_exists(path: str | Path, dataset_path: str) -> bool:
    """Return True when an HDF dataset/group path exists.

    HEC-RAS output datasets vary by version and by the output options used for a plan.
    Callers should use this for optional outputs rather than assuming every dataset exists.
    """
    with h5py.File(path, "r") as h:
        return dataset_path in h


def read_2d_timeseries_optional(path: str | Path, area_name: str, dataset: str) -> np.ndarray | None:
    """Read an optional 2D time-series dataset; return None when it was not written."""
    ds_path = f"{TS_2D}/{area_name}/{dataset}"
    with h5py.File(path, "r") as h:
        ds = h.get(ds_path)
        if ds is None:
            return None
        return np.asarray(ds[()], dtype=float)


def read_2d_summary_result_optional(
    path: str | Path, area_name: str, dataset: str, value_row: int = 0
) -> np.ndarray | None:
    """Read an optional 2D summary dataset; return None when it was not written."""
    ds_path = f"{SUMMARY_2D}/{area_name}/{dataset}"
    with h5py.File(path, "r") as h:
        ds = h.get(ds_path)
        if ds is None:
            return None
        data = np.asarray(ds[()])
        if data.ndim == 2:
            return np.asarray(data[value_row], dtype=float)
        return np.asarray(data, dtype=float)


def read_2d_depth_with_fallback(
    path: str | Path,
    area_name: str,
    cell_min_elevation: np.ndarray,
    *,
    water_surface_ts: np.ndarray | None = None,
    maximum_water_surface: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return depth data without requiring HEC-RAS to have written a ``Depth`` dataset.

    Priority:
      1. Native ``Depth`` time series.
      2. Native ``Maximum Depth`` summary output, when present.
      3. Derived depth proxy = max(Water Surface - Cell Minimum Elevation, 0).

    The derived value is explicitly labelled as a proxy so reviewers can distinguish it
    from a native HEC-RAS depth output. It is primarily used to classify wet/dry cells
    and keep dry-cell elevations from being reported as WSEL impacts.
    """
    cell_min = np.asarray(cell_min_elevation, dtype=float).reshape(-1)
    native_ts = read_2d_timeseries_optional(path, area_name, "Depth")
    if native_ts is not None:
        if native_ts.ndim != 2:
            raise ValueError(f"Unexpected Depth shape {native_ts.shape} for {area_name}")
        return {
            "time_series": native_ts,
            "maximum": np.nanmax(native_ts, axis=0),
            "source": "native_depth_time_series",
            "is_proxy": False,
        }

    native_max = read_2d_summary_result_optional(path, area_name, "Maximum Depth")
    if native_max is not None:
        return {
            "time_series": None,
            "maximum": native_max,
            "source": "native_maximum_depth_summary",
            "is_proxy": False,
        }

    if water_surface_ts is None:
        water_surface_ts = read_2d_timeseries_optional(path, area_name, "Water Surface")
    if water_surface_ts is not None and water_surface_ts.ndim == 2 and water_surface_ts.shape[1] == len(cell_min):
        proxy_ts = np.maximum(np.asarray(water_surface_ts, dtype=float) - cell_min[None, :], 0.0)
        safe_proxy = np.where(np.isfinite(proxy_ts), proxy_ts, -np.inf)
        proxy_max = np.max(safe_proxy, axis=0)
        proxy_max[~np.isfinite(proxy_max)] = 0.0
        return {
            "time_series": proxy_ts,
            "maximum": proxy_max,
            "source": "derived_water_surface_minus_cell_minimum_elevation",
            "is_proxy": True,
        }

    if maximum_water_surface is None:
        maximum_water_surface = read_2d_summary_result_optional(path, area_name, "Maximum Water Surface")
    if maximum_water_surface is not None and len(maximum_water_surface) == len(cell_min):
        proxy_max = np.maximum(np.asarray(maximum_water_surface, dtype=float) - cell_min, 0.0)
        return {
            "time_series": None,
            "maximum": proxy_max,
            "source": "derived_maximum_water_surface_minus_cell_minimum_elevation",
            "is_proxy": True,
        }

    return {
        "time_series": None,
        "maximum": None,
        "source": "unavailable",
        "is_proxy": True,
    }

def read_2d_timeseries(path: str | Path, area_name: str, dataset: str) -> np.ndarray:
    ds_path = f"{TS_2D}/{area_name}/{dataset}"
    with h5py.File(path, "r") as h:
        return np.asarray(h[ds_path][()], dtype=float)


def read_time_axis(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h:
        numeric = np.asarray(h[f"{TS_ROOT}/Time"][()], dtype=float) if f"{TS_ROOT}/Time" in h else np.array([])
        stamps_raw = h[f"{TS_ROOT}/Time Date Stamp"][()] if f"{TS_ROOT}/Time Date Stamp" in h else []
        stamps = [decode(x) for x in stamps_raw]
        return {"time_days": numeric, "time_stamps": stamps}


def read_2d_max_depth(path: str | Path, area_name: str) -> np.ndarray:
    depth = read_2d_timeseries(path, area_name, "Depth")
    if depth.ndim != 2:
        raise ValueError(f"Unexpected Depth shape {depth.shape} for {area_name}")
    return np.nanmax(depth, axis=0)


def read_2d_max_face_velocity(path: str | Path, area_name: str) -> np.ndarray:
    return read_2d_summary_result(path, area_name, "Maximum Face Velocity", value_row=0)


def read_xs_max_wse(path: str | Path) -> np.ndarray | None:
    """Read 1D cross-section Maximum Water Surface when available.

    HEC-RAS plan HDF content varies by model type, version, and output settings.
    A valid plan can contain cross-section geometry without a saved 1D Summary
    Output/Maximum Water Surface dataset. Missing optional hydraulic output must
    not abort the overall model comparison.
    """
    ds_path = f"{SUMMARY_XS}/Maximum Water Surface"
    with h5py.File(path, "r") as h:
        ds = h.get(ds_path)
        if ds is None:
            return None
        data = np.asarray(ds[()])
        if data.ndim == 2:
            if data.shape[0] == 0:
                return None
            return np.asarray(data[0], dtype=float)
        return np.asarray(data, dtype=float)


def _polyline_entities(h: h5py.File, root: str, attributes_name: str = "Attributes") -> list[dict[str, Any]]:
    """Read HEC-RAS HDF polyline entities from Attributes + Polyline/Centerline Info/Points."""
    if root not in h:
        return []
    g = h[root]
    attrs_ds = g.get(attributes_name)
    if attrs_ds is None:
        return []
    attrs = records_to_dicts(attrs_ds[()])
    candidates = [
        ("Polyline Info", "Polyline Points"),
        ("Centerline Info", "Centerline Points"),
    ]
    info = points = None
    for info_name, points_name in candidates:
        if info_name in g and points_name in g:
            info = np.asarray(g[info_name][()], dtype=int)
            points = np.asarray(g[points_name][()], dtype=float)
            break
    if info is None or points is None:
        return attrs
    out = []
    for i, row in enumerate(attrs):
        item = dict(row)
        if i < len(info):
            start = int(info[i, 0])
            count = int(info[i, 1])
            item["points"] = points[start:start + count]
        out.append(item)
    return out


def read_map_reference_geometry(path: str | Path) -> dict[str, Any]:
    """Read georeference-ready 1D/2D line features from a plan HDF."""
    with h5py.File(path, "r") as h:
        projection = decode(h.attrs.get("Projection", ""))
        return {
            "projection_wkt": projection,
            "cross_sections": _polyline_entities(h, "Geometry/Cross Sections"),
            "structures": _polyline_entities(h, "Geometry/Structures"),
            "boundary_condition_lines": _polyline_entities(h, "Geometry/Boundary Condition Lines"),
            "breaklines": _polyline_entities(h, "Geometry/2D Flow Area Break Lines"),
        }


def read_cross_section_details(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read detailed 1D cross-section geometry and hydraulic parameters keyed by River/Reach/RS."""
    with h5py.File(path, "r") as h:
        root = h.get("Geometry/Cross Sections")
        if root is None or "Attributes" not in root:
            return {}
        attrs = records_to_dicts(root["Attributes"][()])
        data = {
            "station_elevation": (root.get("Station Elevation Info"), root.get("Station Elevation Values")),
            "mannings_n": (root.get("Manning's n Info"), root.get("Manning's n Values")),
            "ineffective": (root.get("Ineffective Info"), root.get("Ineffective Blocks")),
            "polyline": (root.get("Polyline Info"), root.get("Polyline Points")),
        }

        out: dict[tuple[str, str, str], dict[str, Any]] = {}
        for i, row in enumerate(attrs):
            key = (str(row.get("River", "")).strip(), str(row.get("Reach", "")).strip(), str(row.get("RS", "")).strip())
            item: dict[str, Any] = {"attributes": row}
            for name, (info_ds, values_ds) in data.items():
                if info_ds is None or values_ds is None or i >= len(info_ds):
                    item[name] = np.empty((0, 2))
                    continue
                info = np.asarray(info_ds[i], dtype=int)
                start, count = int(info[0]), int(info[1])
                item[name] = np.asarray(values_ds[start:start + count])
            out[key] = item
        return out


def read_xs_summary_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return 1D cross-section summary datasets and their documented row variables."""
    out: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as h:
        root = h.get(SUMMARY_XS)
        if root is None:
            return out
        for name, ds in root.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            variables = []
            raw = ds.attrs.get("Variables")
            if raw is not None:
                arr = np.asarray(raw)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    variables = [
                        {"name": str(decode(row[0])), "unit": str(decode(row[1]))}
                        for row in arr
                    ]
            out[name] = {
                "shape": list(ds.shape),
                "variables": variables,
                "primary_variable": variables[0]["name"] if variables else name,
                "primary_unit": variables[0]["unit"] if variables else "",
            }
    return out


def read_xs_summary_result(path: str | Path, dataset: str, value_row: int = 0) -> dict[str, Any]:
    ds_path = f"{SUMMARY_XS}/{dataset}"
    with h5py.File(path, "r") as h:
        if ds_path not in h:
            raise KeyError(f"1D summary result {dataset!r} not available")
        ds = h[ds_path]
        data = np.asarray(ds[()])
        if data.ndim == 2:
            if value_row >= data.shape[0]:
                raise IndexError(value_row)
            values = np.asarray(data[value_row], dtype=float)
        else:
            values = np.asarray(data, dtype=float)
        raw = ds.attrs.get("Variables")
        variables = []
        if raw is not None:
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                variables = [
                    {"name": str(decode(row[0])), "unit": str(decode(row[1]))}
                    for row in arr
                ]
        label = variables[value_row]["name"] if value_row < len(variables) else dataset
        unit = variables[value_row]["unit"] if value_row < len(variables) else ""
        return {"values": values, "label": label, "unit": unit, "dataset": dataset, "value_row": value_row}


def read_xs_timeseries_result(path: str | Path, dataset: str) -> np.ndarray:
    ds_path = f"{TS_XS}/{dataset}"
    with h5py.File(path, "r") as h:
        if ds_path not in h:
            raise KeyError(f"1D time-series result {dataset!r} not available")
        return np.asarray(h[ds_path][()], dtype=float)


def read_xs_identities(path: str | Path) -> list[tuple[str, str, str]]:
    with h5py.File(path, "r") as h:
        attrs = h.get("Geometry/Cross Sections/Attributes")
        if attrs is None:
            return []
        rows = records_to_dicts(attrs[()])
        return [
            (str(r.get("River", "")).strip(), str(r.get("Reach", "")).strip(), str(r.get("RS", "")).strip())
            for r in rows
        ]
