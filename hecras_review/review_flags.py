from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ReviewThresholds:
    """Screening thresholds used to prioritize review, not regulatory acceptance criteria."""

    delta_wse_ft: float = 0.01
    delta_depth_ft: float = 0.10
    delta_velocity_ft_per_s: float = 0.50
    same_time_delta_wse_ft: float = 0.10
    xs_delta_wse_ft: float = 0.01

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _flag(code: str, category: str, title: str, detail: str, *, severity: str = "review", area: str | None = None,
          value: float | int | None = None, threshold: float | None = None, unit: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "code": code,
        "category": category,
        "severity": severity,
        "title": title,
        "detail": detail,
    }
    if area is not None:
        out["area"] = area
    if value is not None:
        out["value"] = value
    if threshold is not None:
        out["threshold"] = threshold
    if unit is not None:
        out["unit"] = unit
    return out


def build_reviewer_flags(report: dict[str, Any], thresholds: ReviewThresholds | None = None) -> dict[str, Any]:
    """Create reviewer-priority flags without making a regulatory pass/fail decision."""
    t = thresholds or ReviewThresholds()
    flags: list[dict[str, Any]] = []

    # Plan/computation settings.
    settings = report.get("plan_setting_changes", [])
    if settings:
        flags.append(_flag(
            "PLAN_SETTINGS_CHANGED", "model_inputs", "Plan/computation settings changed",
            f"{len(settings)} tracked plan/computation setting(s) differ between the selected plans.",
        ))

    # 2D mesh changes.
    for area, m in report.get("mesh_comparison", {}).items():
        status = m.get("status")
        if status == "changed_mesh":
            added = int(m.get("revised_only_centers", 0) or 0)
            removed = int(m.get("existing_only_centers", 0) or 0)
            change = int(m.get("cell_count_change", 0) or 0)
            flags.append(_flag(
                "MESH_CHANGED", "geometry", f"2D mesh changed — {area}",
                f"Cell count change {change:+,}; {removed:,} Existing-only and {added:,} Revised-only cell centers.",
                area=area,
            ))
        elif status in {"added_area", "removed_area"}:
            flags.append(_flag(
                "2D_AREA_CHANGED", "geometry", f"2D flow area {status.replace('_', ' ')} — {area}",
                "The 2D flow-area inventory differs between the selected plans.", area=area,
            ))

    # 1D XS geometry changes.
    xs = report.get("cross_section_detailed_comparison", {})
    if xs.get("added") or xs.get("removed") or int(xs.get("modified_count", 0) or 0) > 0:
        flags.append(_flag(
            "XS_GEOMETRY_CHANGED", "geometry", "1D cross-section geometry changed",
            f"Added: {len(xs.get('added', []))}; removed: {len(xs.get('removed', []))}; modified: {int(xs.get('modified_count', 0) or 0)}.",
        ))

    # Structures / culverts.
    structures = report.get("structure_comparison", {})
    added_c = structures.get("added_culvert_groups", [])
    removed_c = structures.get("removed_culvert_groups", [])
    modified_c = structures.get("modified_culvert_groups", [])
    modified_s = structures.get("modified_structures", [])
    if added_c or removed_c or modified_c or modified_s:
        flags.append(_flag(
            "STRUCTURE_CHANGED", "geometry", "Hydraulic structure/culvert changed",
            f"Added culvert groups: {len(added_c)}; removed: {len(removed_c)}; modified culvert groups: {len(modified_c)}; modified structures: {len(modified_s)}.",
        ))

    # Breaklines / BC lines.
    bl = report.get("breaklines", {})
    if bl.get("existing_count") != bl.get("revised_count"):
        flags.append(_flag(
            "BREAKLINES_CHANGED", "geometry", "2D breakline inventory changed",
            f"Existing breaklines: {bl.get('existing_count', 0)}; Revised breaklines: {bl.get('revised_count', 0)}.",
        ))

    geom_bc = report.get("boundary_condition_lines", {})
    if geom_bc.get("existing", []) != geom_bc.get("revised", []):
        flags.append(_flag(
            "BC_LINES_CHANGED", "boundary_conditions", "2D boundary-condition line geometry changed",
            f"Existing BC lines: {len(geom_bc.get('existing', []))}; Revised BC lines: {len(geom_bc.get('revised', []))}.",
        ))

    ubc = report.get("unsteady_boundary_comparison", {})
    if ubc.get("added_locations") or ubc.get("removed_locations") or ubc.get("common_location_parameter_changes"):
        flags.append(_flag(
            "UNSTEADY_BC_CHANGED", "boundary_conditions", "Unsteady-flow boundary condition changed",
            f"Added locations: {len(ubc.get('added_locations', []))}; removed locations: {len(ubc.get('removed_locations', []))}; parameter changes at common locations: {len(ubc.get('common_location_parameter_changes', []))}.",
        ))

    # 2D hydraulic response.
    for area, h in report.get("hydraulic_comparison", {}).get("two_d", {}).items():
        w = h.get("max_wse_delta_both_wet", {})
        if w:
            max_inc = float(w.get("max_delta_ft", 0.0) or 0.0)
            min_delta = float(w.get("min_delta_ft", 0.0) or 0.0)
            count = int(w.get("count_abs_gt_0_01_ft", 0) or 0)
            if max_inc > t.delta_wse_ft or min_delta < -t.delta_wse_ft:
                flags.append(_flag(
                    "DELTA_MAX_WSE", "hydraulics", f"Wet-to-wet ΔMaximum WSEL — {area}",
                    f"Range {min_delta:+.3f} to {max_inc:+.3f} ft; {count:,} common wet cells have |ΔWSEL| > 0.01 ft.",
                    area=area, value=max(max_inc, abs(min_delta)), threshold=t.delta_wse_ft, unit="ft",
                ))
            bd = int(w.get("became_dry_count", 0) or 0)
            bw = int(w.get("became_wet_count", 0) or 0)
            if bd or bw:
                flags.append(_flag(
                    "WET_DRY_TRANSITION", "hydraulics", f"Wet/dry transitions — {area}",
                    f"Became dry: {bd:,}; became wet: {bw:,}. Review separately from WSEL difference.", area=area,
                ))

        depth = h.get("max_depth_delta", {})
        if depth:
            max_abs = max(abs(float(depth.get("min_delta_ft", 0.0) or 0.0)), abs(float(depth.get("max_delta_ft", 0.0) or 0.0)))
            if max_abs > t.delta_depth_ft:
                flags.append(_flag(
                    "DELTA_MAX_DEPTH", "hydraulics", f"ΔMaximum depth — {area}",
                    f"Maximum absolute depth change is {max_abs:.3f} ft.",
                    area=area, value=max_abs, threshold=t.delta_depth_ft, unit="ft",
                ))

        vel = h.get("max_face_velocity_delta", {})
        if vel:
            max_abs = max(abs(float(vel.get("min_delta_ft_per_s", 0.0) or 0.0)), abs(float(vel.get("max_delta_ft_per_s", 0.0) or 0.0)))
            if max_abs > t.delta_velocity_ft_per_s:
                flags.append(_flag(
                    "DELTA_MAX_VELOCITY", "hydraulics", f"ΔMaximum face velocity — {area}",
                    f"Maximum absolute face-velocity change is {max_abs:.3f} ft/s at exact geometry-matched faces.",
                    area=area, value=max_abs, threshold=t.delta_velocity_ft_per_s, unit="ft/s",
                ))

        sync = h.get("same_time_wse_both_wet", {}).get("summary_peak_absolute", {})
        if sync:
            max_abs = max(abs(float(sync.get("min_delta_ft", 0.0) or 0.0)), abs(float(sync.get("max_delta_ft", 0.0) or 0.0)))
            if max_abs > t.same_time_delta_wse_ft:
                flags.append(_flag(
                    "SAME_TIME_WSE", "hydraulics", f"Same-time WSEL separation — {area}",
                    f"Largest signed same-time WSEL separation magnitude is {max_abs:.3f} ft where both models are wet at that timestamp.",
                    area=area, value=max_abs, threshold=t.same_time_delta_wse_ft, unit="ft",
                ))

    # 1D hydraulic response.
    xs_h = report.get("hydraulic_comparison", {}).get("cross_section_max_wse", {})
    if xs_h.get("index_order_identical") and xs_h.get("count"):
        max_abs = max(abs(float(xs_h.get("min_delta_ft", 0.0) or 0.0)), abs(float(xs_h.get("max_delta_ft", 0.0) or 0.0)))
        if max_abs > t.xs_delta_wse_ft:
            flags.append(_flag(
                "XS_DELTA_MAX_WSE", "hydraulics", "1D cross-section ΔMaximum WSEL",
                f"Maximum absolute cross-section ΔWSEL is {max_abs:.3f} ft; {int(xs_h.get('count_abs_gt_0_01_ft', 0) or 0)} XS have |ΔWSEL| > 0.01 ft.",
                value=max_abs, threshold=t.xs_delta_wse_ft, unit="ft",
            ))

    categories: dict[str, int] = {}
    for f in flags:
        categories[f["category"]] = categories.get(f["category"], 0) + 1

    return {
        "screening_only": True,
        "regulatory_determination": "No automatic pass/fail determination is made.",
        "thresholds": t.to_dict(),
        "flag_count": len(flags),
        "category_counts": categories,
        "flags": flags,
    }


def build_impact_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Compact summary for a reviewer dashboard."""
    mesh_changed = [a for a, m in report.get("mesh_comparison", {}).items() if m.get("status") != "same_mesh"]
    xs = report.get("cross_section_detailed_comparison", {})
    structs = report.get("structure_comparison", {})
    ubc = report.get("unsteady_boundary_comparison", {})

    two_d: dict[str, Any] = {}
    for area, h in report.get("hydraulic_comparison", {}).get("two_d", {}).items():
        w = h.get("max_wse_delta_both_wet", {})
        d = h.get("max_depth_delta", {})
        v = h.get("max_face_velocity_delta", {})
        sync = h.get("same_time_wse_both_wet", {}).get("summary_peak_absolute", {})
        two_d[area] = {
            "wet_to_wet_max_wse_range_ft": [w.get("min_delta_ft"), w.get("max_delta_ft")],
            "wet_to_wet_cells_abs_gt_0_01_ft": w.get("count_abs_gt_0_01_ft", 0),
            "became_dry_count": w.get("became_dry_count", 0),
            "became_wet_count": w.get("became_wet_count", 0),
            "max_depth_delta_range_ft": [d.get("min_delta_ft"), d.get("max_delta_ft")],
            "max_face_velocity_delta_range_ft_per_s": [v.get("min_delta_ft_per_s"), v.get("max_delta_ft_per_s")],
            "peak_same_time_wse_range_ft": [sync.get("min_delta_ft"), sync.get("max_delta_ft")],
        }

    xs_h = report.get("hydraulic_comparison", {}).get("cross_section_max_wse", {})
    return {
        "changed_2d_areas": mesh_changed,
        "modified_cross_sections": int(xs.get("modified_count", 0) or 0),
        "added_cross_sections": len(xs.get("added", [])),
        "removed_cross_sections": len(xs.get("removed", [])),
        "added_culvert_groups": len(structs.get("added_culvert_groups", [])),
        "removed_culvert_groups": len(structs.get("removed_culvert_groups", [])),
        "modified_structures": len(structs.get("modified_structures", [])),
        "added_boundary_locations": len(ubc.get("added_locations", [])),
        "removed_boundary_locations": len(ubc.get("removed_locations", [])),
        "two_d": two_d,
        "one_d": {
            "cross_section_count": xs_h.get("count", 0),
            "max_wse_delta_range_ft": [xs_h.get("min_delta_ft"), xs_h.get("max_delta_ft")],
            "xs_abs_gt_0_01_ft": xs_h.get("count_abs_gt_0_01_ft", 0),
        },
    }
