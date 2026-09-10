from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import folium
import numpy as np
from branca.colormap import LinearColormap


from .compare import compare_2d_mesh
from .georef_map import CoordinateTransformer
from .hdf_reader import (
    read_2d_cell_geometry,
    read_2d_max_wse,
    read_2d_summary_result_optional,
    read_2d_depth_with_fallback,
    read_hdf_inventory,
)
from .spatial import direct_cell_delta, direct_face_delta


def _classify_external_raster(path: str | Path) -> str:
    text = str(path).lower().replace("_", " ")
    name = Path(path).name.lower().replace("_", " ")
    if "terrain" in text or name.startswith("dem") or "elevation" in text:
        return "Terrain"
    if "land" in text and ("class" in text or "cover" in text):
        return "Land Classification"
    if "impervious" in text:
        return "Impervious"
    return classify_raster(path)


def classify_raster(path: str | Path) -> str:
    text = str(path).lower().replace("_", " ")
    name = Path(path).name.lower().replace("_", " ")
    if "delta" in text or "difference" in text or " diff" in text:
        if "wse" in text or "water surface" in text:
            return "Delta WSE"
        if "depth" in text:
            return "Delta Depth"
        if "vel" in text:
            return "Delta Velocity"
        return "Difference"
    if "wse" in text or "water surface" in text:
        return "WSE"
    if "depth" in text:
        return "Depth"
    if "velocity" in text or " vel" in text or name.startswith("vel"):
        return "Velocity"
    if "arrival" in text:
        return "Arrival Time"
    if "duration" in text:
        return "Duration"
    return "Other"


def discover_result_rasters(project_dir: str | Path, max_results: int = 5000) -> list[dict[str, Any]]:
    """Discover raster/VRT outputs below the HEC-RAS project directory.

    The project directory is intentionally used instead of the common Terrain/HMS workspace
    root so large terrain repositories do not flood the result list.
    """
    root = Path(project_dir).resolve()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if len(rows) >= max_results:
            break
        if not p.is_file() or p.suffix.lower() not in {".tif", ".tiff", ".vrt"}:
            continue
        rel = p.relative_to(root).as_posix()
        # HEC-RAS terrain/source grids can live in the model folder, but the reviewer
        # normally wants result rasters. Keep everything discoverable and classify it;
        # the UI can filter to likely result types.
        rows.append({
            "path": str(p),
            "relative_path": rel,
            "name": p.name,
            "category": _classify_external_raster(rel),
            "size_bytes": p.stat().st_size,
            "extension": p.suffix.lower(),
        })
    priority = {"Delta WSE": 0, "WSE": 1, "Depth": 2, "Velocity": 3, "Delta Depth": 4, "Delta Velocity": 5, "Other": 9}
    rows.sort(key=lambda x: (priority.get(x["category"], 8), x["relative_path"].lower()))
    return rows


def _delta_colormap(values: np.ndarray, caption: str) -> LinearColormap:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        limit = 0.1
    else:
        limit = float(np.nanpercentile(np.abs(finite), 98))
        limit = max(limit, 0.01)
    return LinearColormap(
        colors=["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        vmin=-limit,
        vmax=limit,
        caption=caption,
    )


def _ramp(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Map 0..1 float values to RGB without requiring matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    out = np.zeros(v.shape + (3,), dtype=np.uint8)
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        mask = (v >= x0) & (v <= x1 if i == len(stops) - 2 else v < x1)
        if not np.any(mask):
            continue
        t = (v[mask] - x0) / max(x1 - x0, 1e-12)
        c0a = np.asarray(c0, dtype=float)
        c1a = np.asarray(c1, dtype=float)
        out[mask] = np.round(c0a + (c1a - c0a) * t[:, None]).astype(np.uint8)
    return out


def _colorize(data: np.ma.MaskedArray, vmin: float, vmax: float, diverging: bool) -> np.ndarray:
    raw = np.asarray(data.filled(np.nan), dtype=float)
    valid = np.isfinite(raw) & ~np.ma.getmaskarray(data)
    rgba = np.zeros(raw.shape + (4,), dtype=np.uint8)
    if not np.any(valid):
        return rgba
    if diverging:
        lim = max(abs(vmin), abs(vmax), 1e-12)
        norm = (raw + lim) / (2.0 * lim)
        rgb = _ramp(norm, [
            (0.0, (49, 54, 149)), (0.45, (191, 211, 230)),
            (0.5, (247, 247, 247)), (0.55, (253, 219, 199)), (1.0, (165, 0, 38)),
        ])
    else:
        norm = (raw - vmin) / max(vmax - vmin, 1e-12)
        rgb = _ramp(norm, [
            (0.0, (68, 1, 84)), (0.33, (49, 104, 142)),
            (0.66, (53, 183, 121)), (1.0, (253, 231, 37)),
        ])
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(valid, 210, 0).astype(np.uint8)
    return rgba


def _approx_valid_source_bounds(src, max_mask_dim: int = 1024):
    """Approximate the bounding box of valid raster cells without reading the full raster.

    HEC-RAS result rasters can contain large NoData margins. Fitting the web map to the
    full file extent can make the initial view look global. A decimated mask is enough
    to focus the browser map on the actual result footprint.
    """
    from rasterio.enums import Resampling
    from rasterio.windows import Window, bounds as window_bounds

    oh = max(1, min(int(src.height), int(max_mask_dim)))
    ow = max(1, min(int(src.width), int(max_mask_dim)))
    sample = src.read(1, out_shape=(oh, ow), masked=True, resampling=Resampling.nearest)
    valid = (~np.ma.getmaskarray(sample)) & np.isfinite(np.asarray(sample.filled(np.nan), dtype=float))
    if not np.any(valid):
        return src.bounds, False
    rows, cols = np.where(valid)
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1
    sx = src.width / float(ow)
    sy = src.height / float(oh)
    src_col0 = max(0, int(math.floor(col0 * sx)) - 1)
    src_col1 = min(src.width, int(math.ceil(col1 * sx)) + 1)
    src_row0 = max(0, int(math.floor(row0 * sy)) - 1)
    src_row1 = min(src.height, int(math.ceil(row1 * sy)) + 1)
    win = Window(src_col0, src_row0, max(1, src_col1 - src_col0), max(1, src_row1 - src_row0))
    return window_bounds(win, src.transform), True


def _validate_wgs84_bounds(west: float, south: float, east: float, north: float) -> None:
    vals = [west, south, east, north]
    if not all(np.isfinite(vals)):
        raise ValueError("Raster transformed bounds contain non-finite coordinates")
    if west < -180.001 or east > 180.001 or south < -90.001 or north > 90.001 or east <= west or north <= south:
        raise ValueError(
            f"Raster CRS/extent does not transform to a valid WGS84 footprint: "
            f"west={west:.4f}, south={south:.4f}, east={east:.4f}, north={north:.4f}"
        )


def build_raster_review_map(
    raster_path: str | Path,
    max_display_pixels: int = 1_500_000,
    opacity: float = 0.72,
    percentile_clip: float = 2.0,
) -> tuple[folium.Map, dict[str, Any]]:
    """Build a web map for one georeferenced HEC-RAS raster.

    The raster is reprojected/downsampled for browser display only. The source file is not
    modified and full-resolution values remain on disk.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    path = Path(raster_path).resolve()
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS/georeferencing metadata: {path.name}")
        src_width, src_height = src.width, src.height
        focus_bounds, focused_on_valid_data = _approx_valid_source_bounds(src)
        bounds_ll = transform_bounds(src.crs, "EPSG:4326", *focus_bounds, densify_pts=21)
        west, south, east, north = bounds_ll
        _validate_wgs84_bounds(west, south, east, north)
        aspect = max((east - west) / max(north - south, 1e-12), 1e-6)
        # Pick dimensions bounded by max_display_pixels and source dimensions.
        height = int(math.sqrt(max_display_pixels / aspect))
        width = int(height * aspect)
        width = max(64, min(width, src_width, 2200))
        height = max(64, min(height, src_height, 2200))
        if width * height > max_display_pixels:
            scale = math.sqrt(max_display_pixels / (width * height))
            width = max(64, int(width * scale))
            height = max(64, int(height * scale))

        dst = np.full((height, width), np.nan, dtype=np.float32)
        dst_transform = from_bounds(west, south, east, north, width, height)
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        mask = ~np.isfinite(dst)
        data = np.ma.array(dst, mask=mask)
        vals = data.compressed()
        if vals.size == 0:
            raise ValueError(f"Raster contains no finite display values: {path.name}")
        clip = min(max(float(percentile_clip), 0.0), 20.0)
        lo = float(np.nanpercentile(vals, clip)) if clip else float(np.nanmin(vals))
        hi = float(np.nanpercentile(vals, 100.0 - clip)) if clip else float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        category = classify_raster(path)
        diverging = category.startswith("Delta") or category == "Difference" or (lo < 0 < hi)
        if diverging:
            lim = max(abs(lo), abs(hi), 1e-9)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = lo, hi
        rgba = _colorize(data, vmin, vmax, diverging)

        center = [(south + north) / 2.0, (west + east) / 2.0]
        m = folium.Map(
            location=center, zoom_start=11, min_zoom=2, max_zoom=20, tiles=None,
            control_scale=True, max_bounds=True, world_copy_jump=False,
        )
        folium.TileLayer("OpenStreetMap", name="Street Map", show=True, no_wrap=True).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="Aerial Imagery", overlay=False, control=True, show=False, no_wrap=True,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="OpenTopoMap (CC-BY-SA)", name="Topographic", overlay=False, control=True, show=False, no_wrap=True,
        ).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[south, west], [north, east]],
            opacity=float(opacity),
            name=f"{category}: {path.name}",
            interactive=True,
            cross_origin=False,
            zindex=3,
        ).add_to(m)
        if diverging:
            legend = LinearColormap(["#313695", "#f7f7f7", "#a50026"], vmin=vmin, vmax=vmax, caption=f"{category} — display scale")
        else:
            legend = LinearColormap(["#440154", "#31688e", "#35b779", "#fde725"], vmin=vmin, vmax=vmax, caption=f"{category} — display scale")
        legend.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        lat_pad = max((north - south) * 0.08, 0.0005)
        lon_pad = max((east - west) * 0.08, 0.0005)
        view_south = max(-90.0, south - lat_pad)
        view_north = min(90.0, north + lat_pad)
        view_west = max(-180.0, west - lon_pad)
        view_east = min(180.0, east + lon_pad)
        m.fit_bounds([[view_south, view_west], [view_north, view_east]], padding=(30, 30), max_zoom=15)

        meta = {
            "path": str(path), "category": category,
            "source_crs": str(src.crs), "source_width": src_width, "source_height": src_height,
            "display_width": width, "display_height": height,
            "source_bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "display_source_bounds": [float(focus_bounds.left), float(focus_bounds.bottom), float(focus_bounds.right), float(focus_bounds.top)] if hasattr(focus_bounds, "left") else [float(x) for x in focus_bounds],
            "wgs84_bounds": [west, south, east, north],
            "focused_on_valid_data_bounds": bool(focused_on_valid_data),
            "nodata": src.nodata, "source_dtype": src.dtypes[0],
            "display_vmin": vmin, "display_vmax": vmax,
            "source_sample_min": float(np.nanmin(vals)), "source_sample_max": float(np.nanmax(vals)),
            "downsampled_for_display": bool(width < src_width or height < src_height),
        }
        return m, meta


def raster_map_html(m: folium.Map) -> str:
    return m.get_root().render()




def _face_to_cell_max(face_values: np.ndarray, faces_cell_indexes: np.ndarray, n_cells: int) -> np.ndarray:
    """Convert face-based Maximum Face Velocity to a cell display value.

    Each cell receives the maximum finite value from its bounding faces. This is used
    only to create a continuous review raster; raw face-based values remain authoritative.
    """
    vals = np.asarray(face_values, dtype=float).reshape(-1)
    cells = np.asarray(faces_cell_indexes, dtype=int)
    out = np.full(int(n_cells), np.nan, dtype=float)
    for side in range(cells.shape[1]):
        ci = cells[:, side]
        valid = (ci >= 0) & (ci < n_cells) & np.isfinite(vals)
        for c, v in zip(ci[valid], vals[valid]):
            if not np.isfinite(out[c]) or v > out[c]:
                out[c] = v
    return out


def _cell_shapes(geom: dict[str, np.ndarray], values: np.ndarray):
    """Return rasterio-ready polygon/value pairs for HEC-RAS 2D cells."""
    fp = np.asarray(geom['facepoint_coordinates'], dtype=float)
    idx = np.asarray(geom['cell_facepoint_indexes'], dtype=int)
    vals = np.asarray(values, dtype=float).reshape(-1)
    shapes = []
    xs = []
    ys = []
    n = min(len(idx), len(vals))
    for i in range(n):
        v = float(vals[i])
        if not np.isfinite(v):
            continue
        ids = idx[i]
        ids = ids[ids >= 0]
        if len(ids) < 3:
            continue
        ring = fp[ids, :2]
        finite = np.all(np.isfinite(ring), axis=1)
        ring = ring[finite]
        if len(ring) < 3:
            continue
        coords = [[float(x), float(y)] for x, y in ring]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        shapes.append(({"type": "Polygon", "coordinates": [coords]}, v))
        xs.extend(ring[:, 0].tolist())
        ys.extend(ring[:, 1].tolist())
    if not shapes:
        return [], None
    return shapes, (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _merge_bounds(bounds_list):
    valid = [b for b in bounds_list if b is not None]
    if not valid:
        raise ValueError('No valid 2D cell polygon bounds were found')
    return (
        min(b[0] for b in valid), min(b[1] for b in valid),
        max(b[2] for b in valid), max(b[3] for b in valid),
    )


def _grid_spec(bounds, max_pixels: int):
    from rasterio.transform import from_bounds
    west, south, east, north = bounds
    xspan = max(east - west, 1e-6)
    yspan = max(north - south, 1e-6)
    aspect = xspan / yspan
    h = int(math.sqrt(max_pixels / max(aspect, 1e-6)))
    w = int(h * aspect)
    h = max(128, min(h, 2200))
    w = max(128, min(w, 2200))
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / float(w * h))
        w = max(128, int(w * scale))
        h = max(128, int(h * scale))
    return h, w, from_bounds(west, south, east, north, w, h)


def _rasterize_shapes(shapes, out_shape, transform):
    from rasterio.features import rasterize
    if not shapes:
        return np.full(out_shape, np.nan, dtype=np.float32)
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        all_touched=False,
        dtype='float32',
    )


def _to_wgs84_grid(src_grid, src_transform, src_crs, max_pixels: int):
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    src_h, src_w = src_grid.shape
    src_bounds = (
        src_transform.c,
        src_transform.f + src_transform.e * src_h,
        src_transform.c + src_transform.a * src_w,
        src_transform.f,
    )
    west, south, east, north = transform_bounds(src_crs, 'EPSG:4326', *src_bounds, densify_pts=21)
    _validate_wgs84_bounds(west, south, east, north)
    aspect = max((east - west) / max(north - south, 1e-12), 1e-6)
    h = int(math.sqrt(max_pixels / aspect))
    w = int(h * aspect)
    h = max(128, min(h, 2200))
    w = max(128, min(w, 2200))
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / float(w * h))
        w = max(128, int(w * scale))
        h = max(128, int(h * scale))
    dst = np.full((h, w), np.nan, dtype=np.float32)
    dst_transform = from_bounds(west, south, east, north, w, h)
    reproject(
        source=src_grid,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return np.ma.array(dst, mask=~np.isfinite(dst)), (south, west, north, east)


def _value_summary(values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'count': 0, 'min': None, 'max': None}
    return {'count': int(vals.size), 'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals))}


def _prepare_area_cell_values(hdf_path, area_name: str, result_parameter: str, wet_depth_threshold: float = 0.01):
    geom = read_2d_cell_geometry(hdf_path, area_name)
    max_wse = read_2d_summary_result_optional(hdf_path, area_name, 'Maximum Water Surface', value_row=0)
    if max_wse is None:
        raise ValueError(f'Maximum Water Surface is not available for {area_name}')
    depth_info = read_2d_depth_with_fallback(
        hdf_path, area_name, geom['cell_min_elevation'], maximum_water_surface=max_wse,
    )
    max_depth = depth_info.get('maximum')
    wet = np.asarray(max_depth, dtype=float) > float(wet_depth_threshold) if max_depth is not None else np.ones(len(max_wse), dtype=bool)

    if result_parameter == 'Maximum WSE':
        values = np.asarray(max_wse, dtype=float).copy()
        values[~wet] = np.nan
        return geom, values, 'ft', f"WSE; depth source={depth_info.get('source')}"
    if result_parameter == 'Maximum Depth':
        if max_depth is None:
            raise ValueError(f'Depth is unavailable for {area_name}')
        values = np.asarray(max_depth, dtype=float).copy()
        # Existing/Revised depth surfaces show inundated cells only. Delta handling
        # uses the unmasked original values so wetting/drying transitions are retained.
        return geom, values, 'ft', str(depth_info.get('source'))
    if result_parameter == 'Maximum Face Velocity':
        face_values = read_2d_summary_result_optional(hdf_path, area_name, 'Maximum Face Velocity', value_row=0)
        if face_values is None:
            raise ValueError(f'Maximum Face Velocity is not available for {area_name}')
        values = _face_to_cell_max(face_values, geom['faces_cell_indexes'], len(geom['cell_centers']))
        values[~wet] = np.nan
        return geom, values, 'ft/s', 'cell display = maximum of bounding face Maximum Face Velocity values'
    raise ValueError(f'Unsupported HEC-RAS result parameter: {result_parameter}')


def build_plan_result_surface_map(
    existing_hdf: str | Path,
    revised_hdf: str | Path,
    area_name: str,
    result_parameter: str,
    display_mode: str = 'Revised - Existing',
    max_display_pixels: int = 1_500_000,
    opacity: float = 0.74,
    percentile_clip: float = 2.0,
    wet_depth_threshold: float = 0.01,
) -> tuple[folium.Map, dict[str, Any]]:
    """Build a true continuous raster map from HEC-RAS plan HDF 2D cell polygons.

    WSE and Depth are rasterized directly from cell polygons. Face-based velocity is
    converted to a cell display value using the maximum of bounding faces, then rasterized.
    Delta maps rasterize Existing and Revised independently onto the same model-space
    grid and subtract pixel-by-pixel, which supports changed meshes without index matching.
    """
    e_inv = read_hdf_inventory(existing_hdf)
    r_inv = read_hdf_inventory(revised_hdf)
    projection = e_inv.get('projection_wkt') or r_inv.get('projection_wkt')
    if not projection:
        raise ValueError('Plan HDF does not contain projection information')
    source_crs = projection
    common_areas = sorted(set(e_inv.get('two_d_areas', {})) & set(r_inv.get('two_d_areas', {})))
    if area_name == 'All 2D Flow Areas':
        areas = common_areas
    else:
        if area_name not in common_areas:
            raise ValueError(f'2D flow area {area_name!r} is not common to both selected plans')
        areas = [area_name]
    if not areas:
        raise ValueError('No common 2D flow areas were found')

    e_shapes_all, r_shapes_all = [], []
    e_bounds_all, r_bounds_all = [], []
    units = 'ft'
    sources = []
    e_raw_stats, r_raw_stats = [], []

    for area in areas:
        egeom, e_values, units, e_source = _prepare_area_cell_values(existing_hdf, area, result_parameter, wet_depth_threshold)
        rgeom, r_values, units, r_source = _prepare_area_cell_values(revised_hdf, area, result_parameter, wet_depth_threshold)

        # For standalone depth maps, suppress dry cells. For delta depth, retain zero-depth
        # cells so wetting/drying transitions can appear after pixel subtraction.
        if result_parameter == 'Maximum Depth' and display_mode != 'Revised - Existing':
            e_values = np.where(np.asarray(e_values) > wet_depth_threshold, e_values, np.nan)
            r_values = np.where(np.asarray(r_values) > wet_depth_threshold, r_values, np.nan)

        es, eb = _cell_shapes(egeom, e_values)
        rs, rb = _cell_shapes(rgeom, r_values)
        e_shapes_all.extend(es); r_shapes_all.extend(rs)
        e_bounds_all.append(eb); r_bounds_all.append(rb)
        e_raw_stats.append(np.asarray(e_values, dtype=float)); r_raw_stats.append(np.asarray(r_values, dtype=float))
        sources.append(f'{area}: Existing={e_source}; Revised={r_source}')

    bounds = _merge_bounds(e_bounds_all + r_bounds_all)
    h, w, transform = _grid_spec(bounds, int(max_display_pixels))
    e_grid = _rasterize_shapes(e_shapes_all, (h, w), transform)
    r_grid = _rasterize_shapes(r_shapes_all, (h, w), transform)
    mode = str(display_mode)
    if mode == 'Existing':
        src_grid = e_grid
    elif mode == 'Revised':
        src_grid = r_grid
    else:
        both = np.isfinite(e_grid) & np.isfinite(r_grid)
        src_grid = np.full_like(e_grid, np.nan, dtype=np.float32)
        src_grid[both] = r_grid[both] - e_grid[both]
        if result_parameter == 'Maximum Depth':
            # Keep only places where there is an actual depth difference; both-dry cells
            # are not a hydraulic result and should remain transparent.
            same_dry = both & (e_grid <= wet_depth_threshold) & (r_grid <= wet_depth_threshold)
            src_grid[same_dry] = np.nan

    grid, bounds_ll = _to_wgs84_grid(src_grid, transform, source_crs, int(max_display_pixels))
    vals = grid.compressed()
    if vals.size == 0:
        raise ValueError('The selected result produced no finite raster pixels')
    vinfo = _value_summary(vals)
    clip = min(max(float(percentile_clip), 0.0), 20.0)
    lo = float(np.nanpercentile(vals, clip)) if clip else float(np.nanmin(vals))
    hi = float(np.nanpercentile(vals, 100.0 - clip)) if clip else float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    diverging = mode == 'Revised - Existing' or (lo < 0 < hi)
    if diverging:
        lim = max(abs(lo), abs(hi), 1e-9)
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = lo, hi
    rgba = _colorize(grid, vmin, vmax, diverging)

    south, west, north, east = bounds_ll
    center = [(south + north) / 2.0, (west + east) / 2.0]
    m = folium.Map(location=center, zoom_start=10, min_zoom=3, max_zoom=20, tiles=None, control_scale=True, max_bounds=True, world_copy_jump=False)
    folium.TileLayer('OpenStreetMap', name='Street Map', show=True, no_wrap=True).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        name='Aerial Imagery', overlay=False, control=True, show=False, no_wrap=True,
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap (CC-BY-SA)', name='Topographic', overlay=False, control=True, show=False, no_wrap=True,
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=[[south, west], [north, east]], opacity=float(opacity),
        name=f'{mode} {result_parameter}', interactive=False, cross_origin=False, zindex=5,
    ).add_to(m)
    legend_caption = f'{mode} {result_parameter} ({units})'
    if diverging:
        cmap = _delta_colormap(np.array([vmin, vmax]), legend_caption)
    else:
        cmap = LinearColormap(colors=['#440154', '#31688e', '#35b779', '#fde725'], vmin=vmin, vmax=vmax, caption=legend_caption)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[south, west], [north, east]], padding=(18, 18))

    xres = (bounds[2] - bounds[0]) / max(w, 1)
    yres = (bounds[3] - bounds[1]) / max(h, 1)
    meta = {
        'source': 'HEC-RAS Plan Results — polygon rasterization',
        'area_name': area_name,
        'areas_included': areas,
        'parameter': result_parameter,
        'display_mode': mode,
        'value_units': units,
        'result_value_summary': vinfo,
        'source_crs': str(source_crs),
        'model_grid_shape': [h, w],
        'model_grid_resolution': [float(xres), float(yres)],
        'display_grid_shape': list(grid.shape),
        'bounds_wgs84': [south, west, north, east],
        'display_vmin': float(vmin),
        'display_vmax': float(vmax),
        'native_result_source': sources,
        'continuous_surface_method': 'HEC-RAS 2D cell polygon rasterization',
        'delta_method': 'Existing and Revised independently rasterized to common model-space grid, then Revised - Existing pixel-by-pixel',
        'velocity_display_method': 'Maximum of bounding face Maximum Face Velocity values assigned to each cell' if result_parameter == 'Maximum Face Velocity' else None,
        'wet_depth_threshold_ft': float(wet_depth_threshold),
    }
    return m, meta
