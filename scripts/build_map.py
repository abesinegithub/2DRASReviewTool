from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hecras_review.georef_map import build_review_map, save_map
from hecras_review.inspector import compare_plans, map_reference_geometry, spatial_area_review


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a georeferenced HEC-RAS Existing-vs-Revised review map")
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("existing_plan")
    ap.add_argument("revised_plan")
    ap.add_argument("area_name")
    ap.add_argument("--threshold", type=float, default=0.01, help="Minimum absolute wet-to-wet Delta WSEL shown, ft")
    ap.add_argument("--grid-spacing", type=float, default=100.0, help="Changed-mesh normalization spacing, ft")
    ap.add_argument("--output", type=Path, default=Path("hecras_review_map.html"))
    args = ap.parse_args()

    report = compare_plans(args.zip_path, args.existing_plan, args.revised_plan)
    spatial = spatial_area_review(args.zip_path, args.existing_plan, args.revised_plan, args.area_name, args.grid_spacing)
    e_ref = map_reference_geometry(args.zip_path, args.existing_plan)
    r_ref = map_reference_geometry(args.zip_path, args.revised_plan)
    keys = {
        (str(x.get("Type", "")), str(x.get("River", "")), str(x.get("Reach", "")), str(x.get("RS", "")))
        for x in report["structure_comparison"].get("added_culvert_groups", [])
    }
    m, meta = build_review_map(
        spatial, spatial["projection_wkt"], delta_threshold_ft=args.threshold,
        cross_sections=e_ref.get("cross_sections", []), structures=r_ref.get("structures", []),
        bc_lines=r_ref.get("boundary_condition_lines", []), breaklines=r_ref.get("breaklines", []),
        added_culvert_keys=keys,
    )
    out = save_map(m, args.output)
    print(f"Saved {out}")
    print(meta)


if __name__ == "__main__":
    main()
