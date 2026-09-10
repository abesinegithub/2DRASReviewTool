from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def decode(value: Any) -> Any:
    """Decode HDF byte strings recursively into normal Python strings."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    if isinstance(value, np.ndarray):
        return [decode(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def records_to_dicts(array: np.ndarray) -> list[dict[str, Any]]:
    if array.dtype.names is None:
        raise TypeError("Expected a structured numpy array")
    rows: list[dict[str, Any]] = []
    for row in array:
        item: dict[str, Any] = {}
        for name in array.dtype.names:
            item[name] = decode(row[name])
        rows.append(item)
    return rows


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def coord_key(x: float, y: float, decimals: int | None = None) -> tuple[float, float]:
    """Stable spatial key for HEC-RAS coordinates. None preserves exact stored coordinates."""
    if decimals is None:
        return (float(x), float(y))
    return (round(float(x), decimals), round(float(y), decimals))
