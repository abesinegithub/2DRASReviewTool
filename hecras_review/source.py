from __future__ import annotations

import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


class ModelSource(Protocol):
    names: list[str]
    def close(self) -> None: ...
    def find_by_basename(self, basename: str) -> str: ...
    def find_sibling(self, source_name: str, basename: str) -> str: ...
    def find_project_files(self) -> list[str]: ...
    def read_bytes(self, source_name: str) -> bytes: ...
    def read_text(self, source_name: str) -> str: ...
    def materialize(self, source_name: str, directory: str | Path) -> Path: ...


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _norm(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).as_posix().lstrip("./")


def _project_candidates(names: list[str]) -> list[str]:
    """Return likely legacy HEC-RAS project files, preferring projects with sibling plan files.

    A workspace ZIP may contain projection .prj files and nested Model/Terrain/HMS folders.
    Plan files are required to be in the same directory as the candidate project; this avoids
    accidentally pairing a projection .prj with a similarly named plan elsewhere in the bundle.
    """
    normalized = [_norm(n) for n in names]
    name_set = {n.lower() for n in normalized}
    candidates = [n for n in normalized if n.lower().endswith(".prj")]
    scored: list[tuple[int, int, int, str]] = []
    for n in candidates:
        p = PurePosixPath(n)
        stem = p.stem
        parent = p.parent
        plan_count = 0
        for code in range(1, 1000):
            # HEC-RAS commonly uses p01...p99, but p001-style names are also tolerated.
            for suffix in (f"p{code:02d}", f"p{code:03d}"):
                candidate = (parent / f"{stem}.{suffix}").as_posix().lower()
                if candidate in name_set:
                    plan_count += 1
            if code > 120 and plan_count == 0:
                # Avoid unnecessary loops for ordinary projection files.
                break
        # Fallback lexical check for unusual plan numbering while still requiring same folder.
        if plan_count == 0:
            prefix = (parent / f"{stem}.p").as_posix().lower()
            plan_count = sum(1 for x in name_set if x.startswith(prefix) and re.search(r"\.p\d+$", x))
        # sort: projects with more sibling plans first, then shallower/shorter paths
        scored.append((-plan_count, n.count("/"), len(n), n))
    return [n for *_score, n in sorted(scored) if -_score[0] > 0]


@dataclass
class ZipModelSource:
    zip_path: Path

    def __post_init__(self) -> None:
        self.zip_path = Path(self.zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(self.zip_path)
        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"Not a ZIP file: {self.zip_path}")
        self._zip = zipfile.ZipFile(self.zip_path, "r")
        self.names = [_norm(n) for n in self._zip.namelist() if not n.endswith("/")]
        self._raw_name_map = {_norm(n): n for n in self._zip.namelist() if not n.endswith("/")}

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "ZipModelSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def find_by_basename(self, basename: str) -> str:
        matches = [n for n in self.names if os.path.basename(n).lower() == basename.lower()]
        if not matches:
            raise FileNotFoundError(f"{basename} not found in {self.zip_path.name}")
        matches.sort(key=lambda p: (p.count("/"), len(p)))
        return matches[0]

    def find_sibling(self, source_name: str, basename: str) -> str:
        source = PurePosixPath(_norm(source_name))
        candidate = (source.parent / basename).as_posix()
        for n in self.names:
            if n.lower() == candidate.lower():
                return n
        raise FileNotFoundError(f"Sibling {basename} not found next to {source_name}")

    def find_project_files(self) -> list[str]:
        return _project_candidates(self.names)

    def read_bytes(self, source_name: str) -> bytes:
        normalized = _norm(source_name)
        raw_name = self._raw_name_map.get(normalized, source_name)
        return self._zip.read(raw_name)

    def read_text(self, source_name: str) -> str:
        return _decode_text(self.read_bytes(source_name))

    def materialize(self, source_name: str, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / os.path.basename(source_name)
        normalized = _norm(source_name)
        raw_name = self._raw_name_map.get(normalized, source_name)
        with self._zip.open(raw_name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        return out


@dataclass
class DirectoryModelSource:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.names = [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def close(self) -> None:
        return None

    def __enter__(self) -> "DirectoryModelSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _path(self, source_name: str) -> Path:
        p = (self.root / Path(source_name)).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path escapes model root")
        return p

    def find_by_basename(self, basename: str) -> str:
        matches = [n for n in self.names if os.path.basename(n).lower() == basename.lower()]
        if not matches:
            raise FileNotFoundError(f"{basename} not found under {self.root}")
        matches.sort(key=lambda p: (p.count("/"), len(p)))
        return matches[0]

    def find_sibling(self, source_name: str, basename: str) -> str:
        source = PurePosixPath(_norm(source_name))
        candidate = (source.parent / basename).as_posix()
        for n in self.names:
            if n.lower() == candidate.lower():
                return n
        raise FileNotFoundError(f"Sibling {basename} not found next to {source_name}")

    def find_project_files(self) -> list[str]:
        return _project_candidates(self.names)

    def read_bytes(self, source_name: str) -> bytes:
        return self._path(source_name).read_bytes()

    def read_text(self, source_name: str) -> str:
        return _decode_text(self.read_bytes(source_name))

    def materialize(self, source_name: str, directory: str | Path) -> Path:
        # Directory sources are already materialized on disk. Returning the source path
        # avoids expensive copies of large HDF result files during repeated review calls.
        return self._path(source_name)


def open_model_source(path: str | Path) -> ZipModelSource | DirectoryModelSource:
    p = Path(path)
    if p.is_dir():
        return DirectoryModelSource(p)
    return ZipModelSource(p)
