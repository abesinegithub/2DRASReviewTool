"""Core reader/comparison package for the HEC-RAS Review Tool prototype."""

from .inspector import (
    compare_plans, inspect_model, inspect_zip, spatial_area_review,
    available_1d_results, one_d_profile_review, one_d_timeseries_review,
)
from .georef_map import build_review_map, map_html, save_map, CoordinateTransformer

__all__ = [
    "inspect_model", "inspect_zip", "compare_plans", "spatial_area_review",
    "available_1d_results", "one_d_profile_review", "one_d_timeseries_review",
]
__version__ = "1.0.0"
