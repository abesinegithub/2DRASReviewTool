from __future__ import annotations

from typing import Any

import numpy as np

from .utils import coord_key


def _entity_key(row: dict[str, Any], include_name: bool = False) -> tuple:
    key = (
        str(row.get("Type", "")).strip(),
        str(row.get("River", "")).strip(),
        str(row.get("Reach", "")).strip(),
        str(row.get("RS", "")).strip(),
        str(row.get("Connection", "")).strip(),
    )
    if include_name:
        key += (str(row.get("Name", "")).strip(),)
    return key


def compare_dict_settings(a: dict, b: dict) -> list[dict]:
    changes = []
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key), b.get(key)
        if av != bv:
            changes.append({"setting": key, "existing": av, "revised": bv})
    return changes


def compare_cross_sections(existing: dict, revised: dict) -> dict:
    def key(r):
        return (str(r.get("River", "")).strip(), str(r.get("Reach", "")).strip(), str(r.get("RS", "")).strip())
    e = {key(r): r for r in existing.get("items", [])}
    r = {key(x): x for x in revised.get("items", [])}
    common = sorted(set(e) & set(r))
    return {
        "existing_count": len(e),
        "revised_count": len(r),
        "common_count": len(common),
        "added": [list(k) for k in sorted(set(r) - set(e))],
        "removed": [list(k) for k in sorted(set(e) - set(r))],
    }


def compare_structures(existing: dict, revised: dict) -> dict:
    e_struct = {_entity_key(x): x for x in existing.get("items", [])}
    r_struct = {_entity_key(x): x for x in revised.get("items", [])}
    e_culv = {_entity_key(x, include_name=True): x for x in existing.get("culvert_groups", [])}
    r_culv = {_entity_key(x, include_name=True): x for x in revised.get("culvert_groups", [])}

    added_keys = sorted(set(r_culv) - set(e_culv))
    removed_keys = sorted(set(e_culv) - set(r_culv))

    modified = []
    for key in sorted(set(e_culv) & set(r_culv)):
        a, b = e_culv[key], r_culv[key]
        fields = ["Shape Name", "Rise", "Span", "Length", "Top Mann", "Bot Mann", "Entr Loss", "Exit Loss", "Barrels"]
        diffs = {f: {"existing": a.get(f), "revised": b.get(f)} for f in fields if a.get(f) != b.get(f)}
        if diffs:
            modified.append({"key": list(key), "differences": diffs})

    modified_structures = []
    struct_fields = [
        "Weir Width", "Weir US Slope", "Weir DS Slope", "Culverts", "Gates",
        "US Type", "US River", "US Reach", "US RS", "US SA/2D",
        "DS Type", "DS River", "DS Reach", "DS RS", "DS SA/2D",
        "Cell Spacing Near", "Cell Spacing Far", "Near Repeats",
    ]
    for key in sorted(set(e_struct) & set(r_struct)):
        a, b = e_struct[key], r_struct[key]
        diffs = {}
        for f in struct_fields:
            av, bv = a.get(f), b.get(f)
            if isinstance(av, float) and isinstance(bv, float) and np.isnan(av) and np.isnan(bv):
                continue
            if av != bv:
                diffs[f] = {"existing": av, "revised": bv}
        if diffs:
            modified_structures.append({"key": list(key), "differences": diffs})

    return {
        "structure_count": {"existing": len(e_struct), "revised": len(r_struct)},
        "culvert_group_count": {"existing": len(e_culv), "revised": len(r_culv)},
        "added_culvert_groups": [r_culv[k] for k in added_keys],
        "removed_culvert_groups": [e_culv[k] for k in removed_keys],
        "modified_culvert_groups": modified,
        "modified_structures": modified_structures,
    }


def compare_2d_mesh(existing_areas: dict, revised_areas: dict, decimals: int | None = None) -> dict:
    result = {}
    for area in sorted(set(existing_areas) | set(revised_areas)):
        if area not in existing_areas:
            result[area] = {"status": "added_area", "revised_cell_count": revised_areas[area]["cell_count"]}
            continue
        if area not in revised_areas:
            result[area] = {"status": "removed_area", "existing_cell_count": existing_areas[area]["cell_count"]}
            continue
        ea = existing_areas[area]
        ra = revised_areas[area]
        ec = np.asarray(ea.get("cell_centers", []), dtype=float)
        rc = np.asarray(ra.get("cell_centers", []), dtype=float)
        ekeys = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(ec)}
        rkeys = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(rc)}
        common = set(ekeys) & set(rkeys)
        same_order = ec.shape == rc.shape and np.array_equal(ec, rc)
        result[area] = {
            "status": "same_mesh" if same_order else "changed_mesh",
            "existing_cell_count": int(ec.shape[0]),
            "revised_cell_count": int(rc.shape[0]),
            "cell_count_change": int(rc.shape[0] - ec.shape[0]),
            "common_cell_centers": len(common),
            "existing_only_centers": len(set(ekeys) - set(rkeys)),
            "revised_only_centers": len(set(rkeys) - set(ekeys)),
            "same_cell_center_order": bool(same_order),
        }
    return result


def compare_common_cell_wse(existing_centers: np.ndarray, revised_centers: np.ndarray,
                            existing_wse: np.ndarray, revised_wse: np.ndarray,
                            decimals: int | None = None) -> dict:
    e_map = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(existing_centers)}
    r_map = {coord_key(x, y, decimals): i for i, (x, y) in enumerate(revised_centers)}
    keys = sorted(set(e_map) & set(r_map))
    if not keys:
        return {"common_count": 0}
    diffs = np.array([revised_wse[r_map[k]] - existing_wse[e_map[k]] for k in keys], dtype=float)
    finite = np.isfinite(diffs)
    diffs = diffs[finite]
    if diffs.size == 0:
        return {"common_count": len(keys), "finite_count": 0}
    absd = np.abs(diffs)
    return {
        "common_count": len(keys),
        "finite_count": int(diffs.size),
        "min_delta_ft": float(np.min(diffs)),
        "max_delta_ft": float(np.max(diffs)),
        "mean_delta_ft": float(np.mean(diffs)),
        "count_abs_gt_0_001_ft": int(np.sum(absd > 0.001)),
        "count_abs_gt_0_01_ft": int(np.sum(absd > 0.01)),
        "count_abs_gt_0_1_ft": int(np.sum(absd > 0.1)),
    }


def _array_equal_nan_safe(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype.names != b.dtype.names:
        return False
    if a.dtype.names:
        return a.tobytes() == b.tobytes()
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def compare_cross_section_details(existing: dict, revised: dict) -> dict[str, Any]:
    """Semantic comparison of 1D cross-section parameters and station/elevation geometry."""
    ekeys, rkeys = set(existing), set(revised)
    attr_fields = [
        "Len Left", "Len Channel", "Len Right", "Left Bank", "Right Bank", "Friction Mode",
        "Contr", "Expan", "Skew", "PC Invert", "PC Width", "PC Mann",
    ]
    modified = []
    category_counts = {"attributes": 0, "station_elevation": 0, "mannings_n": 0, "ineffective": 0, "polyline": 0}
    for key in sorted(ekeys & rkeys):
        e, r = existing[key], revised[key]
        diffs: dict[str, Any] = {}
        ea, ra = e.get("attributes", {}), r.get("attributes", {})
        ad = {}
        for field in attr_fields:
            av, bv = ea.get(field), ra.get(field)
            if isinstance(av, float) and isinstance(bv, float) and np.isnan(av) and np.isnan(bv):
                continue
            if av != bv:
                ad[field] = {"existing": av, "revised": bv}
        if ad:
            diffs["attributes"] = ad
            category_counts["attributes"] += 1

        for name in ["station_elevation", "mannings_n", "ineffective", "polyline"]:
            av = np.asarray(e.get(name, []))
            bv = np.asarray(r.get(name, []))
            if not _array_equal_nan_safe(av, bv):
                entry: dict[str, Any] = {"existing_shape": list(av.shape), "revised_shape": list(bv.shape)}
                if name == "station_elevation" and av.ndim == 2 and bv.ndim == 2 and av.shape == bv.shape and av.shape[1] >= 2:
                    if np.array_equal(av[:, 0], bv[:, 0]):
                        dz = np.asarray(bv[:, 1], dtype=float) - np.asarray(av[:, 1], dtype=float)
                        finite = dz[np.isfinite(dz)]
                        if finite.size:
                            entry.update({"max_abs_elevation_change_ft": float(np.max(np.abs(finite))), "mean_elevation_change_ft": float(np.mean(finite))})
                diffs[name] = entry
                category_counts[name] += 1
        if diffs:
            modified.append({"river": key[0], "reach": key[1], "rs": key[2], "differences": diffs})

    return {
        "existing_count": len(existing),
        "revised_count": len(revised),
        "common_count": len(ekeys & rkeys),
        "added": [list(k) for k in sorted(rkeys - ekeys)],
        "removed": [list(k) for k in sorted(ekeys - rkeys)],
        "modified_count": len(modified),
        "modified": modified,
        "category_counts": category_counts,
    }
