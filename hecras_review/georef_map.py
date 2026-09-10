from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import folium
import numpy as np
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen, MousePosition
from pyproj import CRS, Transformer


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass(frozen=True)
class CoordinateTransformer:
    source_wkt: str

    def __post_init__(self):
        source = CRS.from_wkt(self.source_wkt)
        object.__setattr__(self, "source_crs", source)
        object.__setattr__(self, "to_wgs84", Transformer.from_crs(source, "EPSG:4326", always_xy=True))

    @property
    def epsg(self) -> int | None:
        return self.source_crs.to_epsg()

    @property
    def source_name(self) -> str:
        return self.source_crs.name

    def xy(self, coordinates: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        arr = np.asarray(coordinates, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        arr = np.atleast_2d(arr)
        lon, lat = self.to_wgs84.transform(arr[:, 0].tolist(), arr[:, 1].tolist())
        return np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)])

    def latlon(self, coordinates: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        ll = self.xy(coordinates)
        return ll[:, [1, 0]] if len(ll) else ll


def _finite_bounds(*arrays: np.ndarray) -> tuple[float, float, float, float] | None:
    valid = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        a = np.atleast_2d(a)
        finite = np.all(np.isfinite(a[:, :2]), axis=1)
        if np.any(finite):
            valid.append(a[finite, :2])
    if not valid:
        return None
    pts = np.vstack(valid)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _expand_bounds(bounds: tuple[float, float, float, float] | None, fraction: float = 0.10, minimum_span: float = 0.002) -> tuple[float, float, float, float] | None:
    """Expand lat/lon bounds to provide useful surrounding map context."""
    if bounds is None:
        return None
    south, west, north, east = bounds
    lat_span = max(north - south, minimum_span)
    lon_span = max(east - west, minimum_span)
    return (
        max(-90.0, south - lat_span * fraction),
        max(-180.0, west - lon_span * fraction),
        min(90.0, north + lat_span * fraction),
        min(180.0, east + lon_span * fraction),
    )


def _point_feature(lat: float, lon: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": properties,
    }


def _line_feature(latlon: np.ndarray, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lat, lon in latlon],
        },
        "properties": properties,
    }


def _point_layer(features: list[dict[str, Any]], name: str, *, show: bool = False,
                 radius: int = 4, fill_opacity: float = 0.8,
                 tooltip_fields: list[str] | None = None,
                 tooltip_aliases: list[str] | None = None) -> folium.GeoJson:
    tooltip = None
    popup = None
    if tooltip_fields:
        aliases = tooltip_aliases or tooltip_fields
        tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, sticky=False)
        popup = folium.GeoJsonPopup(fields=tooltip_fields, aliases=aliases, localize=True, labels=True)
    layer = folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=name,
        show=show,
        marker=folium.CircleMarker(radius=radius, weight=1, fill=True, fill_opacity=fill_opacity),
        style_function=lambda feature: {
            "color": feature["properties"].get("color", "#555555"),
            "fillColor": feature["properties"].get("color", "#555555"),
            "weight": feature["properties"].get("weight", 1),
            "fillOpacity": feature["properties"].get("fillOpacity", fill_opacity),
        },
        tooltip=tooltip,
        popup=popup,
    )
    return layer


def _line_layer(features: list[dict[str, Any]], name: str, *, show: bool = False,
                tooltip_fields: list[str] | None = None) -> folium.GeoJson:
    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, sticky=False) if tooltip_fields else None
    popup = folium.GeoJsonPopup(fields=tooltip_fields, localize=True, labels=True) if tooltip_fields else None
    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=name,
        show=show,
        style_function=lambda feature: {
            "color": feature["properties"].get("color", "#444444"),
            "weight": feature["properties"].get("weight", 2),
            "opacity": feature["properties"].get("opacity", 0.8),
            "dashArray": feature["properties"].get("dashArray"),
        },
        tooltip=tooltip,
        popup=popup,
    )


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


def _build_single_review_map(
    spatial: dict[str, Any],
    projection_wkt: str,
    *,
    delta_threshold_ft: float = 0.01,
    cross_sections: list[dict[str, Any]] | None = None,
    structures: list[dict[str, Any]] | None = None,
    bc_lines: list[dict[str, Any]] | None = None,
    breaklines: list[dict[str, Any]] | None = None,
    added_culvert_keys: set[tuple[str, str, str, str]] | None = None,
    additional_perimeters: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[folium.Map, dict[str, Any]]:
    """Build a georeferenced Leaflet/Folium review map for one 2D flow area.

    The hydraulic WSEL layer contains wet-to-wet exact common-cell comparisons only.
    Wet/dry transitions and mesh changes are deliberately separate layers.
    """
    transformer = CoordinateTransformer(projection_wkt)
    area = spatial["area_name"]
    direct = spatial["direct_max_wse"]
    wet = spatial["direct_wet_dry"]

    e_per = transformer.latlon(spatial.get("existing_perimeter", []))
    r_per = transformer.latlon(spatial.get("revised_perimeter", []))
    coords_xy = np.asarray(direct.get("coordinates", []), dtype=float)
    coords_ll = transformer.latlon(coords_xy)
    delta = np.asarray(direct.get("delta", []), dtype=float)
    existing = np.asarray(direct.get("existing", []), dtype=float)
    revised = np.asarray(direct.get("revised", []), dtype=float)
    both_wet = np.asarray(wet.get("both_wet_mask", []), dtype=bool)

    bounds = _finite_bounds(e_per, r_per, coords_ll)
    center = [29.75, -95.5] if bounds is None else [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    m = folium.Map(location=center, zoom_start=11, min_zoom=3, max_zoom=20, tiles=None, control_scale=True, prefer_canvas=True, max_bounds=True, world_copy_jump=False)

    folium.TileLayer("OpenStreetMap", name="Street Map", show=True, no_wrap=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Aerial Imagery",
        overlay=False,
        control=True,
        show=False,
        no_wrap=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap (CC-BY-SA)",
        name="Topographic",
        overlay=False,
        control=True,
        show=False,
        no_wrap=True,
    ).add_to(m)

    # 2D flow-area perimeters.
    if len(e_per):
        fg = folium.FeatureGroup(name=f"{area} — Existing perimeter", show=True)
        folium.PolyLine(e_per.tolist(), color="#5b5b5b", weight=3, dash_array="8 5", opacity=0.9).add_to(fg)
        fg.add_to(m)
    if len(r_per):
        fg = folium.FeatureGroup(name=f"{area} — Revised perimeter", show=True)
        folium.PolyLine(r_per.tolist(), color="#111111", weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)

    # Wet-to-wet direct Delta Max WSEL.
    wse_mask = both_wet & np.isfinite(delta) & (np.abs(delta) >= float(delta_threshold_ft))
    cmap = _delta_colormap(delta[wse_mask], "Revised − Existing Maximum WSEL (ft)")
    wse_features: list[dict[str, Any]] = []
    for i in np.where(wse_mask)[0]:
        lat, lon = coords_ll[i]
        d = float(delta[i])
        wse_features.append(_point_feature(lat, lon, {
            "Layer": "Wet-to-wet ΔMax WSEL",
            "Existing WSEL (ft)": round(float(existing[i]), 3),
            "Revised WSEL (ft)": round(float(revised[i]), 3),
            "Delta WSEL (ft)": round(d, 3),
            "Model X (ft)": round(float(coords_xy[i, 0]), 2),
            "Model Y (ft)": round(float(coords_xy[i, 1]), 2),
            "color": cmap(d),
        }))
    if wse_features:
        _point_layer(
            wse_features, f"{area} — Wet-to-wet ΔMax WSEL |Δ| ≥ {delta_threshold_ft:g} ft", show=True, radius=4,
            tooltip_fields=["Delta WSEL (ft)", "Existing WSEL (ft)", "Revised WSEL (ft)", "Model X (ft)", "Model Y (ft)"],
        ).add_to(m)
        cmap.add_to(m)

    # Spatially normalized wet-to-wet Delta Max WSEL for changed meshes.
    norm = spatial.get("normalized_max_wse", {})
    nwet = spatial.get("normalized_wet_dry") or {}
    n_xy = np.asarray(norm.get("coordinates", []), dtype=float)
    n_ll = transformer.latlon(n_xy)
    n_delta = np.asarray(norm.get("delta", []), dtype=float)
    n_existing = np.asarray(norm.get("existing", []), dtype=float)
    n_revised = np.asarray(norm.get("revised", []), dtype=float)
    n_both_wet = np.asarray(nwet.get("both_wet_mask", []), dtype=bool)
    n_mask = n_both_wet & np.isfinite(n_delta) & (np.abs(n_delta) >= float(delta_threshold_ft)) if len(n_delta) else np.array([], dtype=bool)
    normalized_features = []
    if len(n_delta):
        for i in np.where(n_mask)[0]:
            lat, lon = n_ll[i]
            d = float(n_delta[i])
            normalized_features.append(_point_feature(lat, lon, {
                "Layer": "Normalized wet-to-wet ΔMax WSEL",
                "Existing WSEL (ft)": round(float(n_existing[i]), 3),
                "Revised WSEL (ft)": round(float(n_revised[i]), 3),
                "Delta WSEL (ft)": round(d, 3),
                "Model X (ft)": round(float(n_xy[i,0]), 2),
                "Model Y (ft)": round(float(n_xy[i,1]), 2),
                "color": cmap(d),
            }))
    if normalized_features:
        _point_layer(
            normalized_features, f"{area} — Normalized wet-to-wet ΔMax WSEL ({len(normalized_features):,})", show=False, radius=4,
            tooltip_fields=["Layer", "Delta WSEL (ft)", "Existing WSEL (ft)", "Revised WSEL (ft)", "Model X (ft)", "Model Y (ft)"],
        ).add_to(m)

    # Delta maximum depth on exact common cell centers.
    depth = spatial.get("direct_max_depth", {})
    d_xy = np.asarray(depth.get("coordinates", []), dtype=float)
    d_ll = transformer.latlon(d_xy)
    d_delta = np.asarray(depth.get("delta", []), dtype=float)
    d_existing = np.asarray(depth.get("existing", []), dtype=float)
    d_revised = np.asarray(depth.get("revised", []), dtype=float)
    d_cmap = _delta_colormap(d_delta[np.isfinite(d_delta)], "Revised − Existing Maximum Depth (ft)")
    depth_features = []
    d_mask = np.isfinite(d_delta) & (np.abs(d_delta) >= float(delta_threshold_ft)) if len(d_delta) else np.array([], dtype=bool)
    for i in np.where(d_mask)[0]:
        lat, lon = d_ll[i]
        dd = float(d_delta[i])
        depth_features.append(_point_feature(lat, lon, {
            "Existing Max Depth (ft)": round(float(d_existing[i]), 3),
            "Revised Max Depth (ft)": round(float(d_revised[i]), 3),
            "Delta Max Depth (ft)": round(dd, 3),
            "Model X (ft)": round(float(d_xy[i,0]),2), "Model Y (ft)": round(float(d_xy[i,1]),2),
            "color": d_cmap(dd),
        }))
    if depth_features:
        _point_layer(depth_features, f"{area} — ΔMaximum Depth |Δ| ≥ {delta_threshold_ft:g} ft", show=False, radius=3,
                     tooltip_fields=["Delta Max Depth (ft)", "Existing Max Depth (ft)", "Revised Max Depth (ft)", "Model X (ft)", "Model Y (ft)"]).add_to(m)

    # Delta maximum face velocity where HEC-RAS face endpoint geometry matches exactly.
    vel = spatial.get("direct_max_face_velocity", {})
    v_xy = np.asarray(vel.get("coordinates", []), dtype=float)
    v_ll = transformer.latlon(v_xy)
    v_delta = np.asarray(vel.get("delta", []), dtype=float)
    v_cmap = _delta_colormap(v_delta[np.isfinite(v_delta)], "Revised − Existing Maximum Face Velocity (ft/s)")
    velocity_features = []
    v_mask = np.isfinite(v_delta) & (np.abs(v_delta) >= 0.05) if len(v_delta) else np.array([], dtype=bool)
    for i in np.where(v_mask)[0]:
        lat, lon = v_ll[i]
        dv = float(v_delta[i])
        velocity_features.append(_point_feature(lat, lon, {
            "Delta Max Velocity (ft/s)": round(dv, 3),
            "Model X (ft)": round(float(v_xy[i,0]),2), "Model Y (ft)": round(float(v_xy[i,1]),2),
            "color": v_cmap(dv),
        }))
    if velocity_features:
        _point_layer(velocity_features, f"{area} — ΔMaximum Face Velocity |Δ| ≥ 0.05 ft/s", show=False, radius=3,
                     tooltip_fields=["Delta Max Velocity (ft/s)", "Model X (ft)", "Model Y (ft)"]).add_to(m)

    # Largest signed same-time WSEL separation per exact common cell.
    sync = spatial.get("same_time_wse", {})
    s_xy = np.asarray(sync.get("coordinates", []), dtype=float)
    s_ll = transformer.latlon(s_xy)
    s_delta = np.asarray(sync.get("peak_absolute_same_time_delta", []), dtype=float)
    s_ti = np.asarray(sync.get("peak_time_index", []), dtype=int)
    s_stamps = sync.get("time_stamps", [])
    same_time_features = []
    s_mask = np.isfinite(s_delta) & (np.abs(s_delta) >= float(delta_threshold_ft)) if len(s_delta) else np.array([], dtype=bool)
    s_cmap = _delta_colormap(s_delta[s_mask], "Largest Same-Time WSEL Separation (ft)")
    for i in np.where(s_mask)[0]:
        lat, lon = s_ll[i]
        dv = float(s_delta[i])
        ti = int(s_ti[i]) if i < len(s_ti) else -1
        stamp = s_stamps[ti] if 0 <= ti < len(s_stamps) else ""
        same_time_features.append(_point_feature(lat, lon, {
            "Peak Same-Time Delta WSEL (ft)": round(dv, 3), "Timestamp": stamp,
            "Model X (ft)": round(float(s_xy[i,0]),2), "Model Y (ft)": round(float(s_xy[i,1]),2),
            "color": s_cmap(dv),
        }))
    if same_time_features:
        _point_layer(same_time_features, f"{area} — Largest Same-Time WSEL Separation", show=False, radius=3,
                     tooltip_fields=["Peak Same-Time Delta WSEL (ft)", "Timestamp", "Model X (ft)", "Model Y (ft)"]).add_to(m)

    # Wet/dry transitions.
    became_dry = np.asarray(wet.get("became_dry_mask", []), dtype=bool)
    became_wet = np.asarray(wet.get("became_wet_mask", []), dtype=bool)
    dry_features = []
    wet_features = []
    for i in np.where(became_dry)[0]:
        lat, lon = coords_ll[i]
        dry_features.append(_point_feature(lat, lon, {"Status": "Became dry", "color": "#7b3294", "Model X (ft)": round(float(coords_xy[i,0]),2), "Model Y (ft)": round(float(coords_xy[i,1]),2)}))
    for i in np.where(became_wet)[0]:
        lat, lon = coords_ll[i]
        wet_features.append(_point_feature(lat, lon, {"Status": "Became wet", "color": "#008837", "Model X (ft)": round(float(coords_xy[i,0]),2), "Model Y (ft)": round(float(coords_xy[i,1]),2)}))
    if dry_features:
        _point_layer(dry_features, f"{area} — Became dry ({len(dry_features):,})", radius=4, tooltip_fields=["Status", "Model X (ft)", "Model Y (ft)"]).add_to(m)
    if wet_features:
        _point_layer(wet_features, f"{area} — Became wet ({len(wet_features):,})", radius=6, tooltip_fields=["Status", "Model X (ft)", "Model Y (ft)"]).add_to(m)

    # Mesh modification locations (center identity only, not hydraulic equivalence).
    eo_xy = np.asarray(direct.get("existing_only_coordinates", []), dtype=float)
    ro_xy = np.asarray(direct.get("revised_only_coordinates", []), dtype=float)
    for raw, label, color in [(eo_xy, "Existing-only cell centers", "#8c510a"), (ro_xy, "Revised-only cell centers", "#01665e")]:
        ll = transformer.latlon(raw)
        feats = [_point_feature(lat, lon, {"Status": label, "color": color, "Model X (ft)": round(float(x),2), "Model Y (ft)": round(float(y),2)}) for (lat,lon),(x,y) in zip(ll, raw)]
        if feats:
            _point_layer(feats, f"{area} — {label} ({len(feats):,})", radius=3, tooltip_fields=["Status", "Model X (ft)", "Model Y (ft)"]).add_to(m)

    # Cross-section reference layer (off by default).
    xs_features = []
    for xs in cross_sections or []:
        pts = np.asarray(xs.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        ll = transformer.latlon(pts)
        xs_features.append(_line_feature(ll, {
            "River": xs.get("River", ""), "Reach": xs.get("Reach", ""), "RS": xs.get("RS", ""),
            "color": "#636363", "weight": 1, "opacity": 0.45,
        }))
    if xs_features:
        _line_layer(xs_features, f"1D Cross Sections ({len(xs_features):,})", show=False, tooltip_fields=["River", "Reach", "RS"]).add_to(m)

    # Structure centerlines. Emphasize structures associated with added culvert groups.
    struct_features = []
    added_culvert_keys = added_culvert_keys or set()
    for s in structures or []:
        pts = np.asarray(s.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        ll = transformer.latlon(pts)
        key = (str(s.get("Type","")), str(s.get("River","")), str(s.get("Reach","")), str(s.get("RS","")))
        changed = key in added_culvert_keys
        struct_features.append(_line_feature(ll, {
            "Type": s.get("Type", ""), "River": s.get("River", ""), "Reach": s.get("Reach", ""), "RS": s.get("RS", ""),
            "Culverts": s.get("Culverts", ""), "Review": "Added culvert group" if changed else "Reference structure",
            "color": "#d73027" if changed else "#252525", "weight": 5 if changed else 2, "opacity": 0.95 if changed else 0.55,
        }))
    if struct_features:
        _line_layer(struct_features, "1D/2D Structures", show=True, tooltip_fields=["Review", "Type", "River", "Reach", "RS", "Culverts"]).add_to(m)

    # Proposed BC lines / breaklines.
    bc_features = []
    for b in bc_lines or []:
        pts = np.asarray(b.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        ll = transformer.latlon(pts)
        bc_features.append(_line_feature(ll, {
            "Name": b.get("Name", ""), "SA-2D": b.get("SA-2D", ""), "Type": b.get("Type", ""),
            "color": "#e31a1c", "weight": 5, "opacity": 0.95,
        }))
    if bc_features:
        _line_layer(bc_features, f"Revised BC Lines ({len(bc_features):,})", show=True, tooltip_fields=["Name", "SA-2D", "Type"]).add_to(m)

    bl_features = []
    for b in breaklines or []:
        pts = np.asarray(b.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        ll = transformer.latlon(pts)
        bl_features.append(_line_feature(ll, {
            "Name": b.get("Name", ""), "Cell Spacing Near": b.get("Cell Spacing Near"), "Cell Spacing Far": b.get("Cell Spacing Far"),
            "color": "#ff7f00", "weight": 4, "opacity": 0.9,
        }))
    if bl_features:
        _line_layer(bl_features, f"Revised Breaklines ({len(bl_features):,})", show=True, tooltip_fields=["Name", "Cell Spacing Near", "Cell Spacing Far"]).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon:", num_digits=6).add_to(m)
    folium.LatLngPopup().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    display_bounds = _expand_bounds(bounds, fraction=0.12)
    if display_bounds is not None:
        m.fit_bounds(
            [[display_bounds[0], display_bounds[1]], [display_bounds[2], display_bounds[3]]],
            padding=(35, 35), max_zoom=14,
        )

    metadata = {
        "area_name": area,
        "source_crs_name": transformer.source_name,
        "source_epsg": transformer.epsg,
        "wse_feature_count": len(wse_features),
        "normalized_wse_feature_count": len(normalized_features),
        "depth_feature_count": len(depth_features),
        "velocity_feature_count": len(velocity_features),
        "same_time_wse_feature_count": len(same_time_features),
        "became_dry_count": len(dry_features),
        "became_wet_count": len(wet_features),
        "existing_only_center_count": int(len(eo_xy)),
        "revised_only_center_count": int(len(ro_xy)),
        "cross_section_feature_count": len(xs_features),
        "structure_feature_count": len(struct_features),
        "bc_line_feature_count": len(bc_features),
        "breakline_feature_count": len(bl_features),
    }
    return m, metadata


def build_review_map(
    spatial: dict[str, Any] | list[dict[str, Any]],
    projection_wkt: str,
    *,
    delta_threshold_ft: float = 0.01,
    cross_sections: list[dict[str, Any]] | None = None,
    structures: list[dict[str, Any]] | None = None,
    bc_lines: list[dict[str, Any]] | None = None,
    breaklines: list[dict[str, Any]] | None = None,
    added_culvert_keys: set[tuple[str, str, str, str]] | None = None,
    additional_perimeters: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[folium.Map, dict[str, Any]]:
    """Build one map for a single 2D area or a combined map for several areas."""
    if isinstance(spatial, dict):
        return _build_single_review_map(
            spatial, projection_wkt, delta_threshold_ft=delta_threshold_ft,
            cross_sections=cross_sections, structures=structures, bc_lines=bc_lines,
            breaklines=breaklines, added_culvert_keys=added_culvert_keys,
        )

    spatials = [x for x in spatial if isinstance(x, dict)]
    if not spatials:
        raise ValueError("At least one 2D spatial review is required")

    base, first_meta = _build_single_review_map(
        spatials[0], projection_wkt, delta_threshold_ft=delta_threshold_ft,
        cross_sections=cross_sections, structures=structures, bc_lines=bc_lines,
        breaklines=breaklines, added_culvert_keys=added_culvert_keys,
    )
    # Rebuild LayerControl only after all child-area layers are attached.  Reusing the
    # first area's already-rendered control can omit layers added later on some Folium/
    # Leaflet versions, making "All 2D Flow Areas" appear to contain only one area.
    for key, child in list(base._children.items()):
        if child.__class__.__name__ == "LayerControl":
            del base._children[key]
    metas = [first_meta]
    for item in spatials[1:]:
        child_map, child_meta = _build_single_review_map(
            item, projection_wkt, delta_threshold_ft=delta_threshold_ft,
            cross_sections=None, structures=None, bc_lines=None, breaklines=None,
            added_culvert_keys=None,
        )
        metas.append(child_meta)
        for child in list(child_map._children.values()):
            if isinstance(child, (folium.GeoJson, folium.FeatureGroup, folium.vector_layers.PolyLine)):
                child.add_to(base)

    transformer = CoordinateTransformer(projection_wkt)
    spatial_area_names = {str(x.get("area_name", "")) for x in spatials}
    extra = additional_perimeters or {}
    for side, color, dash in [("existing", "#777777", "8 5"), ("revised", "#111111", None)]:
        for area_name, xy in (extra.get(side, {}) or {}).items():
            if str(area_name) in spatial_area_names:
                continue
            ll = transformer.latlon(np.asarray(xy, dtype=float))
            if not len(ll):
                continue
            fg = folium.FeatureGroup(name=f"{area_name} — {side.title()} perimeter (geometry only)", show=True)
            folium.PolyLine(ll.tolist(), color=color, weight=3, dash_array=dash, opacity=0.9).add_to(fg)
            fg.add_to(base)

    bound_arrays = []
    for item in spatials:
        for key in ("existing_perimeter", "revised_perimeter"):
            arr = np.asarray(item.get(key, []), dtype=float)
            if arr.size:
                bound_arrays.append(transformer.latlon(arr))
        coords = np.asarray(item.get("direct_max_wse", {}).get("coordinates", []), dtype=float)
        if coords.size:
            bound_arrays.append(transformer.latlon(coords))
    for side in ("existing", "revised"):
        for xy in (extra.get(side, {}) or {}).values():
            arr = np.asarray(xy, dtype=float)
            if arr.size:
                bound_arrays.append(transformer.latlon(arr))
    combined_bounds = _expand_bounds(_finite_bounds(*bound_arrays), fraction=0.10)
    if combined_bounds is not None:
        base.fit_bounds(
            [[combined_bounds[0], combined_bounds[1]], [combined_bounds[2], combined_bounds[3]]],
            padding=(35, 35), max_zoom=13,
        )
    folium.LayerControl(collapsed=False).add_to(base)

    all_area_names = set(spatial_area_names)
    all_area_names.update(str(x) for x in (extra.get("existing", {}) or {}))
    all_area_names.update(str(x) for x in (extra.get("revised", {}) or {}))
    meta = {
        "area_name": "All 2D Flow Areas",
        "area_names": sorted(all_area_names),
        "area_count": len(all_area_names),
        "source_crs_name": first_meta.get("source_crs_name"),
        "source_epsg": first_meta.get("source_epsg"),
    }
    for key in [
        "wse_feature_count", "normalized_wse_feature_count", "depth_feature_count",
        "velocity_feature_count", "same_time_wse_feature_count", "became_dry_count",
        "became_wet_count", "existing_only_center_count", "revised_only_center_count",
    ]:
        meta[key] = int(sum(int(m.get(key, 0) or 0) for m in metas))
    for key in ["cross_section_feature_count", "structure_feature_count", "bc_line_feature_count", "breakline_feature_count"]:
        meta[key] = int(first_meta.get(key, 0) or 0)
    return base, meta


def map_html(map_obj: folium.Map) -> str:
    return map_obj.get_root().render()


def save_map(map_obj: folium.Map, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    return output_path
