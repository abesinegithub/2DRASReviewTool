from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely import contains_xy, points

from .utils import coord_key


def _finite_stats(delta: np.ndarray, suffix: str = "ft") -> dict[str, Any]:
    d = np.asarray(delta, dtype=float)
    finite = np.isfinite(d)
    d = d[finite]
    if d.size == 0:
        return {"finite_count": 0}
    ad = np.abs(d)
    return {
        "finite_count": int(d.size),
        f"min_delta_{suffix}": float(np.min(d)),
        f"max_delta_{suffix}": float(np.max(d)),
        f"mean_delta_{suffix}": float(np.mean(d)),
        f"median_delta_{suffix}": float(np.median(d)),
        f"count_abs_gt_0_001_{suffix}": int(np.sum(ad > 0.001)),
        f"count_abs_gt_0_01_{suffix}": int(np.sum(ad > 0.01)),
        f"count_abs_gt_0_1_{suffix}": int(np.sum(ad > 0.1)),
        f"count_abs_gt_0_5_{suffix}": int(np.sum(ad > 0.5)),
    }


def wet_dry_classification(existing_depth: np.ndarray, revised_depth: np.ndarray,
                           wet_depth_threshold: float = 0.01) -> dict[str, Any]:
    """Classify paired locations by maximum (or same-time) depth."""
    ed = np.asarray(existing_depth, dtype=float)
    rd = np.asarray(revised_depth, dtype=float)
    if ed.shape != rd.shape:
        raise ValueError(f"Depth arrays must have same shape, got {ed.shape} vs {rd.shape}")
    ewet = np.isfinite(ed) & (ed > wet_depth_threshold)
    rwet = np.isfinite(rd) & (rd > wet_depth_threshold)
    both_wet = ewet & rwet
    became_dry = ewet & ~rwet
    became_wet = ~ewet & rwet
    both_dry = ~ewet & ~rwet
    return {
        "wet_depth_threshold_ft": float(wet_depth_threshold),
        "both_wet_mask": both_wet,
        "became_dry_mask": became_dry,
        "became_wet_mask": became_wet,
        "both_dry_mask": both_dry,
        "both_wet_count": int(np.sum(both_wet)),
        "became_dry_count": int(np.sum(became_dry)),
        "became_wet_count": int(np.sum(became_wet)),
        "both_dry_count": int(np.sum(both_dry)),
    }


def wet_wet_delta_summary(delta: np.ndarray, existing_depth: np.ndarray, revised_depth: np.ndarray,
                          wet_depth_threshold: float = 0.01, suffix: str = "ft") -> dict[str, Any]:
    cls = wet_dry_classification(existing_depth, revised_depth, wet_depth_threshold)
    d = np.asarray(delta, dtype=float)
    stats = _finite_stats(d[cls["both_wet_mask"]], suffix=suffix)
    return {
        **{k:v for k,v in cls.items() if not k.endswith("_mask")},
        **stats,
    }


def match_exact_centers(existing_centers: np.ndarray, revised_centers: np.ndarray, decimals: int | None = None):
    e_map = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(existing_centers)}
    r_map = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(revised_centers)}
    common_keys = sorted(set(e_map) & set(r_map))
    e_only_keys = sorted(set(e_map) - set(r_map))
    r_only_keys = sorted(set(r_map) - set(e_map))
    return {
        "common_keys": common_keys,
        "existing_indices": np.asarray([e_map[k] for k in common_keys], dtype=int),
        "revised_indices": np.asarray([r_map[k] for k in common_keys], dtype=int),
        "common_coordinates": np.asarray(common_keys, dtype=float),
        "existing_only_indices": np.asarray([e_map[k] for k in e_only_keys], dtype=int),
        "revised_only_indices": np.asarray([r_map[k] for k in r_only_keys], dtype=int),
        "existing_only_coordinates": np.asarray(e_only_keys, dtype=float) if e_only_keys else np.empty((0, 2)),
        "revised_only_coordinates": np.asarray(r_only_keys, dtype=float) if r_only_keys else np.empty((0, 2)),
    }


def direct_cell_delta(existing_centers: np.ndarray, revised_centers: np.ndarray,
                      existing_values: np.ndarray, revised_values: np.ndarray,
                      decimals: int | None = None) -> dict[str, Any]:
    m = match_exact_centers(existing_centers, revised_centers, decimals)
    if len(m["existing_indices"]) == 0:
        return {"common_count": 0, "coordinates": np.empty((0,2)), "delta": np.array([])}
    e = np.asarray(existing_values, dtype=float)[m["existing_indices"]]
    r = np.asarray(revised_values, dtype=float)[m["revised_indices"]]
    delta = r - e
    result = {
        "common_count": int(len(delta)),
        "coordinates": m["common_coordinates"],
        "existing": e,
        "revised": r,
        "delta": delta,
        "existing_only_coordinates": m["existing_only_coordinates"],
        "revised_only_coordinates": m["revised_only_coordinates"],
    }
    result.update(_finite_stats(delta))
    return result


def synchronized_cell_wse(existing_centers: np.ndarray, revised_centers: np.ndarray,
                          existing_ts: np.ndarray, revised_ts: np.ndarray,
                          existing_stamps: list[str], revised_stamps: list[str],
                          existing_depth_ts: np.ndarray | None = None,
                          revised_depth_ts: np.ndarray | None = None,
                          wet_depth_threshold: float = 0.01,
                          decimals: int | None = None) -> dict[str, Any]:
    m = match_exact_centers(existing_centers, revised_centers, decimals)
    if existing_stamps != revised_stamps:
        return {"timestamps_identical": False, "common_count": int(len(m["existing_indices"]))}
    ei, ri = m["existing_indices"], m["revised_indices"]
    if existing_ts.shape[0] != revised_ts.shape[0] or not len(ei):
        return {"timestamps_identical": True, "common_count": int(len(ei)), "compatible": False}
    delta = revised_ts[:, ri] - existing_ts[:, ei]

    wet_mask = np.isfinite(delta)
    transition = {}
    if existing_depth_ts is not None and revised_depth_ts is not None:
        e_depth = np.asarray(existing_depth_ts, dtype=float)[:, ei]
        r_depth = np.asarray(revised_depth_ts, dtype=float)[:, ri]
        if e_depth.shape != delta.shape or r_depth.shape != delta.shape:
            raise ValueError("Depth time-series shape is incompatible with WSEL time series")
        ewet = e_depth > wet_depth_threshold
        rwet = r_depth > wet_depth_threshold
        wet_mask &= ewet & rwet
        transition = {
            "wet_depth_threshold_ft": float(wet_depth_threshold),
            "cells_ever_became_dry": int(np.sum(np.any(ewet & ~rwet, axis=0))),
            "cells_ever_became_wet": int(np.sum(np.any(~ewet & rwet, axis=0))),
            "cells_with_any_both_wet_timestamp": int(np.sum(np.any(ewet & rwet, axis=0))),
        }

    masked = np.where(wet_mask, delta, np.nan)
    any_valid = np.any(np.isfinite(masked), axis=0)
    max_inc = np.full(masked.shape[1], np.nan, dtype=float)
    max_dec = np.full(masked.shape[1], np.nan, dtype=float)
    peak_abs_delta = np.full(masked.shape[1], np.nan, dtype=float)
    abs_idx = np.full(masked.shape[1], -1, dtype=int)
    if np.any(any_valid):
        cols_valid = np.where(any_valid)[0]
        valid_block = masked[:, cols_valid]
        max_inc[cols_valid] = np.nanmax(valid_block, axis=0)
        max_dec[cols_valid] = np.nanmin(valid_block, axis=0)
        abs_local = np.nanargmax(np.abs(valid_block), axis=0)
        peak_abs_delta[cols_valid] = valid_block[abs_local, np.arange(len(cols_valid))]
        abs_idx[cols_valid] = abs_local

    result = {
        "timestamps_identical": True,
        "compatible": True,
        "common_count": int(len(ei)),
        "coordinates": m["common_coordinates"],
        "max_same_time_increase": max_inc,
        "max_same_time_decrease": max_dec,
        "peak_absolute_same_time_delta": peak_abs_delta,
        "peak_time_index": abs_idx,
        "time_stamps": existing_stamps,
        "summary_peak_absolute": _finite_stats(peak_abs_delta),
        "summary_max_increase": _finite_stats(max_inc),
        "summary_max_decrease": _finite_stats(max_dec),
        **transition,
    }
    return result


def face_centers(facepoint_coordinates: np.ndarray, faces_facepoint_indexes: np.ndarray) -> np.ndarray:
    idx = np.asarray(faces_facepoint_indexes, dtype=int)
    pts = np.asarray(facepoint_coordinates, dtype=float)
    return (pts[idx[:, 0]] + pts[idx[:, 1]]) / 2.0


def face_keys(facepoint_coordinates: np.ndarray, faces_facepoint_indexes: np.ndarray, decimals: int | None = None):
    pts = np.asarray(facepoint_coordinates, dtype=float)
    out = []
    for a, b in np.asarray(faces_facepoint_indexes, dtype=int):
        p1 = coord_key(*pts[a], decimals)
        p2 = coord_key(*pts[b], decimals)
        out.append(tuple(sorted((p1, p2))))
    return out


def direct_face_delta(existing_facepoints: np.ndarray, existing_faces: np.ndarray,
                      revised_facepoints: np.ndarray, revised_faces: np.ndarray,
                      existing_values: np.ndarray, revised_values: np.ndarray,
                      decimals: int | None = None) -> dict[str, Any]:
    ek = face_keys(existing_facepoints, existing_faces, decimals)
    rk = face_keys(revised_facepoints, revised_faces, decimals)
    em = {k:i for i,k in enumerate(ek)}
    rm = {k:i for i,k in enumerate(rk)}
    common = sorted(set(em) & set(rm))
    if not common:
        return {"common_face_count": 0}
    ei = np.asarray([em[k] for k in common], dtype=int)
    ri = np.asarray([rm[k] for k in common], dtype=int)
    delta = np.asarray(revised_values, dtype=float)[ri] - np.asarray(existing_values, dtype=float)[ei]
    centers = face_centers(existing_facepoints, existing_faces)[ei]
    result = {"common_face_count": int(len(common)), "coordinates": centers, "delta": delta}
    result.update(_finite_stats(delta, suffix="ft_per_s"))
    return result


def _polygon_from_perimeter(perimeter: np.ndarray) -> Polygon:
    poly = Polygon(np.asarray(perimeter, dtype=float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _cell_polygons(cell_facepoint_indexes: np.ndarray, facepoint_coordinates: np.ndarray):
    polygons = []
    original_indices = []
    for i, row in enumerate(np.asarray(cell_facepoint_indexes, dtype=int)):
        ids = row[row >= 0]
        # HEC-RAS 5.x may include 2-point boundary/ghost records after the polygon cells.
        if len(ids) < 3:
            continue
        poly = Polygon(np.asarray(facepoint_coordinates, dtype=float)[ids])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        polygons.append(poly)
        original_indices.append(i)
    return polygons, np.asarray(original_indices, dtype=int)


def _assign_points_to_cells(sample_coordinates: np.ndarray,
                            cell_facepoint_indexes: np.ndarray,
                            facepoint_coordinates: np.ndarray) -> np.ndarray:
    polygons, original_indices = _cell_polygons(cell_facepoint_indexes, facepoint_coordinates)
    assigned = np.full(len(sample_coordinates), -1, dtype=int)
    if not polygons or not len(sample_coordinates):
        return assigned
    tree = STRtree(polygons)
    sample_points = points(sample_coordinates[:, 0], sample_coordinates[:, 1])
    pairs = tree.query(sample_points, predicate="within")
    if pairs.size:
        assigned[pairs[0]] = original_indices[pairs[1]]
    return assigned


def normalized_polygon_cell_delta(existing_values: np.ndarray, revised_values: np.ndarray,
                                  existing_cell_facepoint_indexes: np.ndarray,
                                  existing_facepoint_coordinates: np.ndarray,
                                  revised_cell_facepoint_indexes: np.ndarray,
                                  revised_facepoint_coordinates: np.ndarray,
                                  existing_perimeter: np.ndarray, revised_perimeter: np.ndarray,
                                  grid_spacing: float = 100.0,
                                  max_points: int = 250_000) -> dict[str, Any]:
    """Compare cell-centered results on common points using the containing HEC-RAS cell polygon.

    This is a spatially normalized comparison for changed meshes. It is not cell equivalence:
    each common analysis point is assigned independently to the cell polygon that contains it
    in the Existing and Revised meshes, then the two cell result values are differenced.
    """
    ep = _polygon_from_perimeter(existing_perimeter)
    rp = _polygon_from_perimeter(revised_perimeter)
    intersection = ep.intersection(rp)
    if intersection.is_empty:
        return {"method": "common_grid_containing_cell", "point_count": 0, "reason": "No overlapping 2D perimeter"}

    minx, miny, maxx, maxy = intersection.bounds
    spacing = float(grid_spacing)
    nx = max(1, int(np.floor((maxx-minx)/spacing)))
    ny = max(1, int(np.floor((maxy-miny)/spacing)))
    if nx * ny > max_points:
        spacing *= np.sqrt((nx * ny) / max_points)
        nx = max(1, int(np.floor((maxx-minx)/spacing)))
        ny = max(1, int(np.floor((maxy-miny)/spacing)))

    # Offset by half a cell to avoid deliberately sampling on regular 100-ft mesh lines.
    xs = minx + spacing / 2.0 + np.arange(nx) * spacing
    ys = miny + spacing / 2.0 + np.arange(ny) * spacing
    xx, yy = np.meshgrid(xs, ys)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    inside = contains_xy(intersection, flat[:, 0], flat[:, 1])
    sample = flat[inside]
    if not len(sample):
        return {"method": "common_grid_containing_cell", "point_count": 0, "grid_spacing_ft": spacing}

    e_idx = _assign_points_to_cells(sample, existing_cell_facepoint_indexes, existing_facepoint_coordinates)
    r_idx = _assign_points_to_cells(sample, revised_cell_facepoint_indexes, revised_facepoint_coordinates)
    keep = (e_idx >= 0) & (r_idx >= 0)
    sample = sample[keep]
    e_idx = e_idx[keep]
    r_idx = r_idx[keep]
    e = np.asarray(existing_values, dtype=float)[e_idx]
    r = np.asarray(revised_values, dtype=float)[r_idx]
    delta = r - e
    result = {
        "method": "common_grid_containing_cell",
        "point_count": int(len(sample)),
        "grid_spacing_ft": float(spacing),
        "coordinates": sample,
        "existing_cell_indexes": e_idx,
        "revised_cell_indexes": r_idx,
        "existing": e,
        "revised": r,
        "delta": delta,
    }
    result.update(_finite_stats(delta))
    return result


def top_change_points(coordinates: np.ndarray, delta: np.ndarray, n: int = 10) -> dict[str, list[dict[str, float]]]:
    c = np.asarray(coordinates, dtype=float)
    d = np.asarray(delta, dtype=float)
    finite = np.isfinite(d)
    c, d = c[finite], d[finite]
    if not len(d):
        return {"largest_increases": [], "largest_decreases": []}
    inc = np.argsort(d)[::-1][:n]
    dec = np.argsort(d)[:n]
    def rows(indices):
        return [{"x": float(c[i,0]), "y": float(c[i,1]), "delta_ft": float(d[i])} for i in indices]
    return {"largest_increases": rows(inc), "largest_decreases": rows(dec)}


def top_synchronized_events(sync_result: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
    coords = np.asarray(sync_result.get("coordinates", []), dtype=float)
    delta = np.asarray(sync_result.get("peak_absolute_same_time_delta", []), dtype=float)
    idx = np.asarray(sync_result.get("peak_time_index", []), dtype=int)
    stamps = sync_result.get("time_stamps", [])
    if not len(delta):
        return []
    finite = np.isfinite(delta) & (idx >= 0)
    candidates = np.where(finite)[0]
    if not len(candidates):
        return []
    order = candidates[np.argsort(np.abs(delta[candidates]))[::-1][:n]]
    rows = []
    for i in order:
        ti = int(idx[i])
        rows.append({
            "x": float(coords[i,0]),
            "y": float(coords[i,1]),
            "delta_ft": float(delta[i]),
            "time_index": ti,
            "time_stamp": stamps[ti] if 0 <= ti < len(stamps) else None,
        })
    return rows
