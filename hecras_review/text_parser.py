from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class KVText:
    text: str

    def values(self, key: str) -> list[str]:
        prefix = key.lower() + "="
        found = []
        for raw in self.text.splitlines():
            line = raw.strip("\ufeff\r\n")
            if line.lower().startswith(prefix):
                found.append(line.split("=", 1)[1].strip())
        return found

    def first(self, key: str, default=None):
        vals = self.values(key)
        return vals[0] if vals else default


def parse_project(text: str) -> dict:
    kv = KVText(text)
    units = "US Customary" if re.search(r"^English Units\s*$", text, re.M | re.I) else (
        "SI" if re.search(r"^SI Units\s*$", text, re.M | re.I) else None
    )
    return {
        "project_title": kv.first("Proj Title"),
        "current_plan": kv.first("Current Plan"),
        "units": units,
        "geometry_files": kv.values("Geom File"),
        "unsteady_files": kv.values("Unsteady File"),
        "plan_files": kv.values("Plan File"),
        "dss_file": kv.first("DSS File"),
    }


def parse_plan(text: str, plan_code: str | None = None) -> dict:
    kv = KVText(text)
    return {
        "plan_code": plan_code,
        "plan_title": kv.first("Plan Title"),
        "short_identifier": kv.first("Short Identifier"),
        "program_version": kv.first("Program Version"),
        "simulation_date": kv.first("Simulation Date"),
        "geometry_file": kv.first("Geom File"),
        "flow_file": kv.first("Flow File"),
        "computation_interval": kv.first("Computation Interval"),
        "output_interval": kv.first("Output Interval"),
        "instantaneous_interval": kv.first("Instantaneous Interval"),
        "mapping_interval": kv.first("Mapping Interval"),
        "two_d_names": kv.values("UNET D2 Name"),
    }


def important_plan_settings(text: str) -> dict[str, list[str] | str]:
    """Settings worth flagging if changed between review plans."""
    kv = KVText(text)
    keys = [
        "Simulation Date",
        "Computation Interval",
        "Output Interval",
        "Instantaneous Interval",
        "Mapping Interval",
        "UNET Theta",
        "UNET ZTol",
        "UNET ZSATol",
        "UNET MxIter",
        "UNET D2 Theta",
        "UNET D2 Z Tol",
        "UNET D2 Volume Tol",
        "UNET D2 Max Iterations",
        "UNET D2 Equation",
        "UNET D2 TimeSlices",
    ]
    result = {}
    for key in keys:
        vals = kv.values(key)
        if vals:
            result[key] = vals if len(vals) > 1 else vals[0]
    return result


def parse_unsteady_boundaries(text: str) -> list[dict]:
    """Parse Boundary Location blocks from a HEC-RAS unsteady-flow text file.

    This intentionally preserves raw key/value content; later versions can add
    type-specific normalization for every boundary-condition subtype.
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    current: dict | None = None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith("Boundary Location="):
            if current is not None:
                blocks.append(current)
            raw_loc = line.split("=", 1)[1]
            parts = [p.strip() for p in raw_loc.split(",")]
            while len(parts) < 8:
                parts.append("")
            current = {
                "location_raw": raw_loc,
                "river": parts[0],
                "reach": parts[1],
                "river_station": parts[2],
                "storage_area": parts[4],
                "two_d_area": parts[5],
                "connection": parts[6],
                "bc_line": parts[7],
                "parameters": {},
            }
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            current["parameters"][key.strip()] = value.strip()
    if current is not None:
        blocks.append(current)
    return blocks


def boundary_identity(b: dict) -> tuple:
    return (
        b.get("river", ""), b.get("reach", ""), b.get("river_station", ""),
        b.get("storage_area", ""), b.get("two_d_area", ""),
        b.get("connection", ""), b.get("bc_line", ""),
    )
