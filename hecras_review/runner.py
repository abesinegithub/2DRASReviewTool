from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import zipfile
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any

from .source import DirectoryModelSource


@dataclass
class RasInstallation:
    version: str
    exe_path: str


@dataclass
class RasRunResult:
    plan_code: str
    command: list[str]
    command_line: str
    return_code: int | None
    status: str
    runtime_seconds: float
    result_hdf: str | None
    result_hdf_exists: bool
    result_hdf_updated: bool
    log_path: str
    log_tail: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_hecras_installations() -> list[RasInstallation]:
    """Discover installed HEC-RAS Ras.exe files on a local Windows workstation."""
    roots: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value) / "HEC" / "HEC-RAS")
    # Typical paths are useful even if the environment variables are absent.
    roots.extend([Path(r"C:\Program Files\HEC\HEC-RAS"), Path(r"C:\Program Files (x86)\HEC\HEC-RAS")])
    seen: set[str] = set()
    found: list[RasInstallation] = []
    for root in roots:
        if not root.exists():
            continue
        for exe in root.glob("*/Ras.exe"):
            key = str(exe.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(RasInstallation(version=exe.parent.name, exe_path=str(exe.resolve())))
    def version_key(item: RasInstallation):
        import re
        nums = [int(x) for x in re.findall(r"\d+", item.version)]
        return tuple(nums)
    return sorted(found, key=version_key, reverse=True)


def discover_7zip_executable(custom_path: str | Path | None = None) -> str | None:
    """Find a 7-Zip command-line executable for .7z extraction."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return str(p.resolve())
    for name in ("7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())
    candidates = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "7-Zip" / "7z.exe")
    candidates.extend([Path(r"C:\Program Files\7-Zip\7z.exe"), Path(r"C:\Program Files (x86)\7-Zip\7z.exe")])
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def _archive_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".zip" and zipfile.is_zipfile(path):
        return "zip"
    if suffix == ".7z":
        return "7z"
    raise ValueError(f"Unsupported archive type: {path}. Use .zip or .7z, or select an extracted local workspace folder.")


def _validate_7z_members(archive: Path, seven_zip_exe: str) -> None:
    """Reject obvious absolute/path-traversal entries before extraction."""
    proc = subprocess.run([seven_zip_exe, "l", "-slt", str(archive)], capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"7-Zip could not list archive {archive}: {proc.stderr or proc.stdout}")
    paths = []
    in_files = False
    for line in proc.stdout.splitlines():
        if line.startswith("----------"):
            in_files = True
            continue
        if in_files and line.startswith("Path = "):
            paths.append(line[7:].strip())
    for name in paths:
        normalized = name.replace("\\", "/")
        parts = [x for x in normalized.split("/") if x]
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) or ".." in parts:
            raise ValueError(f"Unsafe path in 7z archive: {name}")


def _safe_extract_7z(archive: Path, dest: Path, seven_zip_exe: str | None = None) -> None:
    exe = discover_7zip_executable(seven_zip_exe)
    if not exe:
        raise RuntimeError("A .7z archive was selected but 7-Zip was not found. Install 7-Zip or provide the full path to 7z.exe, or extract the archive manually and use Local extracted model + dependencies mode.")
    _validate_7z_members(archive, exe)
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([exe, "x", "-y", f"-o{dest}", str(archive)], capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"7-Zip extraction failed for {archive}: {proc.stderr or proc.stdout}")


def _safe_extract_archive(path: Path, dest: Path, seven_zip_exe: str | None = None) -> None:
    kind = _archive_kind(path)
    if kind == "zip":
        _safe_extract(path, dest)
    else:
        _safe_extract_7z(path, dest, seven_zip_exe)


def archive_uncompressed_size(path: str | Path, seven_zip_exe: str | None = None) -> int | None:
    """Best-effort uncompressed archive size, used only for disk-space warnings."""
    p = Path(path)
    kind = _archive_kind(p)
    if kind == "zip":
        with zipfile.ZipFile(p, "r") as zf:
            return sum(i.file_size for i in zf.infolist() if not i.is_dir())
    exe = discover_7zip_executable(seven_zip_exe)
    if not exe:
        return None
    proc = subprocess.run([exe, "l", "-slt", str(p)], capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        return None
    total = 0
    in_files = False
    for line in proc.stdout.splitlines():
        if line.startswith("----------"):
            in_files = True
            continue
        if in_files and line.startswith("Size = "):
            try:
                total += int(line[7:].strip())
            except Exception:
                pass
    return total or None


def discover_runnable_projects(root: str | Path, shallow_depth: int = 4) -> list[Path]:
    """Find runnable HEC-RAS project files without reading large dependency contents.

    Large FEMA/MAAPnext workspaces often place the RAS model near the workspace root while
    Terrain/HMS folders contain huge nested trees. We therefore do a bounded breadth-first
    directory-name scan first and only fall back to a full recursive walk when no runnable
    project is found in the shallow levels.
    """
    root = Path(root).resolve()
    if root.is_file() and root.suffix.lower() == ".prj":
        return [root] if any(root.parent.glob(f"{root.stem}.p[0-9][0-9]*")) else []
    if not root.is_dir():
        return []

    def runnable_in_dir(directory: Path) -> list[Path]:
        try:
            entries = list(directory.iterdir())
        except (PermissionError, OSError):
            return []
        files = [x for x in entries if x.is_file()]
        lower = {x.name.lower() for x in files}
        out = []
        for f in files:
            if not f.name.lower().endswith(".prj"):
                continue
            stem = f.stem.lower()
            if any(re.fullmatch(re.escape(stem) + r"\.p\d+", name) for name in lower):
                out.append(f.resolve())
        return out

    queue: list[tuple[Path, int]] = [(root, 0)]
    shallow_found: list[Path] = []
    visited: set[Path] = set()
    while queue:
        directory, depth = queue.pop(0)
        if directory in visited:
            continue
        visited.add(directory)
        shallow_found.extend(runnable_in_dir(directory))
        if depth >= shallow_depth:
            continue
        try:
            for child in directory.iterdir():
                if child.is_dir():
                    queue.append((child, depth + 1))
        except (PermissionError, OSError):
            continue
    if shallow_found:
        return sorted(set(shallow_found), key=lambda p: (len(p.parts), str(p).lower()))

    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        lower = {x.lower() for x in filenames}
        for name in filenames:
            if not name.lower().endswith(".prj"):
                continue
            stem = Path(name).stem.lower()
            if any(re.fullmatch(re.escape(stem) + r"\.p\d+", x) for x in lower):
                found.append((Path(dirpath) / name).resolve())
    return sorted(set(found), key=lambda p: (len(p.parts), str(p).lower()))


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise ValueError(f"Unsafe path in ZIP: {info.filename}")
        zf.extractall(dest)


def _single_top_folder(zip_path: Path) -> str | None:
    """Return the common top-level directory when every archive member lives under one."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        parts = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            n = info.filename.replace("\\", "/").strip("/")
            if not n:
                continue
            pieces = n.split("/")
            if len(pieces) < 2:
                return None
            parts.append(pieces[0])
        if parts and len(set(parts)) == 1:
            return parts[0]
    return None


def _extract_component_zip(zip_path: Path, workspace_root: Path) -> Path:
    """Extract a separate component ZIP as one sibling folder in a merged workspace.

    The component folder name is the ZIP stem. If the ZIP already contains one top-level
    folder with the same name (case-insensitive), that wrapper is stripped to avoid
    Model/Model, Terrain/Terrain, etc.
    """
    component_name = zip_path.stem
    component_dir = workspace_root / component_name
    component_dir.mkdir(parents=True, exist_ok=True)
    top = _single_top_folder(zip_path)
    temp = Path(tempfile.mkdtemp(prefix="hecras_component_"))
    try:
        _safe_extract(zip_path, temp)
        src_root = temp / top if top and top.lower() == component_name.lower() else temp
        for child in src_root.iterdir():
            target = component_dir / child.name
            if target.exists():
                raise FileExistsError(f"Workspace component collision: {target}")
            shutil.move(str(child), str(target))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return component_dir


def prepare_working_copy(
    archive_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    destination: str | Path | None = None,
    seven_zip_exe: str | Path | None = None,
) -> Path:
    """Create a safe computation workspace from .zip/.7z archive(s).

    For very large (multi-GB to 100+ GB) models, prefer an already extracted Local Workspace
    and do not call this function; extraction duplicates the archive's uncompressed footprint.
    """
    paths = [Path(x) for x in archive_path] if isinstance(archive_path, (list, tuple)) else [Path(archive_path)]
    if not paths:
        raise ValueError("At least one archive is required")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        _archive_kind(p)

    if destination is None:
        extract_root = Path(tempfile.mkdtemp(prefix="hecras_review_workspace_"))
    else:
        extract_root = Path(destination).expanduser().resolve()
        extract_root.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        _safe_extract_archive(paths[0], extract_root, str(seven_zip_exe) if seven_zip_exe else None)
        workspace_root = extract_root
    else:
        workspace_root = extract_root
        for p in paths:
            component_name = p.stem
            component_dir = workspace_root / component_name
            component_dir.mkdir(parents=True, exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix="hecras_component_"))
            try:
                _safe_extract_archive(p, temp, str(seven_zip_exe) if seven_zip_exe else None)
                children = list(temp.iterdir())
                src_root = temp
                if len(children) == 1 and children[0].is_dir() and children[0].name.lower() == component_name.lower():
                    src_root = children[0]
                for child in src_root.iterdir():
                    target = component_dir / child.name
                    if target.exists():
                        raise FileExistsError(f"Workspace component collision: {target}")
                    shutil.move(str(child), str(target))
            finally:
                shutil.rmtree(temp, ignore_errors=True)

    projects = discover_runnable_projects(workspace_root)
    if not projects:
        raise RuntimeError("No runnable HEC-RAS project was found after extraction. The tool searches recursively for a .prj file with sibling .p## plan files.")
    return workspace_root.resolve()

def find_project_file(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    projects = discover_runnable_projects(model_dir)
    if not projects:
        raise RuntimeError(f"No runnable HEC-RAS project .prj found under {model_dir}")
    return projects[0]

def find_plan_file(model_dir: str | Path, plan_code: str, project_file: str | Path | None = None) -> Path:
    project = Path(project_file).resolve() if project_file else find_project_file(model_dir)
    stem = project.stem
    code = str(plan_code).lower().lstrip(".")
    candidate = project.parent / f"{stem}.{code}"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate

def build_compute_command(ras_exe: str | Path, project_file: str | Path, plan_file: str | Path) -> list[str]:
    """Return the semantic HEC-RAS compute arguments.

    Do not pass this list directly to ``subprocess.Popen`` on Windows. HEC-RAS's
    command-line parser requires the project and plan *file parameters themselves*
    to be explicitly double-quoted, even when a path contains no spaces. Python's
    normal Windows list-to-command-line conversion omits unnecessary quotes, which
    HEC-RAS then rejects. Use :func:`build_compute_command_line` for execution.
    """
    return [str(Path(ras_exe)), "-c", str(Path(project_file)), str(Path(plan_file))]


def _hecras_quote(path: str | Path) -> str:
    """Quote a Windows HEC-RAS file parameter exactly once."""
    value = str(path).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    # A literal double quote is not valid in a normal Windows filename and would
    # make the raw command line ambiguous, so fail loudly rather than mangling it.
    if '"' in value:
        raise ValueError(f'Invalid double quote in HEC-RAS path: {value}')
    return f'"{value}"'


def build_compute_command_line(ras_exe: str | Path, project_file: str | Path, plan_file: str | Path) -> str:
    """Build the raw Windows command line required by HEC-RAS.

    HEC-RAS displays ``Optional project filename (must have quotes)`` and the same
    requirement for the plan filename. We therefore construct the raw command line
    ourselves instead of letting ``subprocess`` decide whether quoting is needed.
    """
    return f'{_hecras_quote(ras_exe)} -c {_hecras_quote(project_file)} {_hecras_quote(plan_file)}'


def _tail(path: Path, n: int = 30) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def run_hecras_plan(
    model_dir: str | Path,
    plan_code: str,
    ras_exe: str | Path,
    timeout_seconds: float = 4 * 3600,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    project_file: str | Path | None = None,
) -> RasRunResult:
    """Run one HEC-RAS plan in-place in an extracted working copy.

    Uses the supported Ras.exe command-line compute pattern: Ras.exe -c project.prj plan.p##.
    The original uploaded ZIP is never modified.
    """
    model_dir = Path(model_dir).resolve()
    ras_exe = Path(ras_exe)
    if os.name != "nt":
        raise RuntimeError("HEC-RAS computation is Windows-only. Run the web app on the Windows workstation where HEC-RAS is installed.")
    if not ras_exe.exists():
        raise FileNotFoundError(f"Ras.exe not found: {ras_exe}")

    project_file = Path(project_file).resolve() if project_file else find_project_file(model_dir)
    plan_file = find_plan_file(model_dir, plan_code, project_file=project_file)
    result_hdf = plan_file.with_suffix(plan_file.suffix + ".hdf")
    before_mtime = result_hdf.stat().st_mtime if result_hdf.exists() else None
    before_size = result_hdf.stat().st_size if result_hdf.exists() else None

    log_dir = model_dir / ".hecras_review_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"run_{plan_file.suffix.lstrip('.')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    command = build_compute_command(ras_exe, project_file, plan_file)
    command_line = build_compute_command_line(ras_exe, project_file, plan_file)
    start = time.monotonic()
    wall_start = time.time()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("Command: " + command_line + "\n")
        log.flush()
        try:
            # IMPORTANT: pass a raw command-line string on Windows. HEC-RAS requires
            # literal quotes around project/plan filenames. Passing a Python list lets
            # subprocess omit those quotes when the path contains no spaces, causing
            # HEC-RAS to report "unrecognized command parameter". shell=False keeps
            # this out of cmd.exe while preserving our explicit quotes for CreateProcess.
            proc = subprocess.Popen(
                command_line,
                cwd=str(project_file.parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                rc = proc.poll()
                elapsed = time.monotonic() - start
                hdf_exists = result_hdf.exists()
                hdf_size = result_hdf.stat().st_size if hdf_exists else 0
                hdf_mtime = result_hdf.stat().st_mtime if hdf_exists else None
                updated = bool(hdf_exists and (before_mtime is None or (hdf_mtime or 0) > before_mtime or hdf_size != before_size))
                if progress_callback:
                    progress_callback({
                        "plan_code": plan_code,
                        "state": "running" if rc is None else "finished",
                        "elapsed_seconds": elapsed,
                        "result_hdf_exists": hdf_exists,
                        "result_hdf_size_bytes": hdf_size,
                        "result_hdf_updated": updated,
                        "log_path": str(log_path),
                    })
                if rc is not None:
                    break
                if elapsed > timeout_seconds:
                    proc.kill()
                    proc.wait(timeout=30)
                    return RasRunResult(
                        plan_code=plan_code, command=command, command_line=command_line, return_code=proc.returncode,
                        status="timeout", runtime_seconds=elapsed,
                        result_hdf=str(result_hdf), result_hdf_exists=result_hdf.exists(),
                        result_hdf_updated=False, log_path=str(log_path), log_tail=_tail(log_path),
                        error=f"Run exceeded timeout of {timeout_seconds:.0f} seconds",
                    )
                time.sleep(1.0)
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RasRunResult(
                plan_code=plan_code, command=command, command_line=command_line, return_code=None, status="failed",
                runtime_seconds=elapsed, result_hdf=str(result_hdf), result_hdf_exists=result_hdf.exists(),
                result_hdf_updated=False, log_path=str(log_path), log_tail=_tail(log_path), error=str(exc),
            )

    elapsed = time.monotonic() - start
    exists = result_hdf.exists()
    mtime = result_hdf.stat().st_mtime if exists else None
    size = result_hdf.stat().st_size if exists else None
    updated = bool(exists and (before_mtime is None or (mtime or 0) > before_mtime or size != before_size or (mtime or 0) >= wall_start - 2))
    status = "completed" if proc.returncode == 0 and exists else ("completed_no_output" if proc.returncode == 0 else "failed")
    return RasRunResult(
        plan_code=plan_code, command=command, command_line=command_line, return_code=proc.returncode, status=status,
        runtime_seconds=elapsed, result_hdf=str(result_hdf) if exists else str(result_hdf),
        result_hdf_exists=exists, result_hdf_updated=updated,
        log_path=str(log_path), log_tail=_tail(log_path), error=None if status == "completed" else "HEC-RAS did not produce the expected plan HDF output.",
    )


def zip_working_model(model_dir: str | Path, output_zip: str | Path) -> Path:
    """Package the extracted working model (including newly generated outputs) as a new ZIP."""
    model_dir = Path(model_dir).resolve()
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    base = output_zip.with_suffix("")
    # Archive the workspace contents, not the randomly generated temporary directory name.
    # This preserves the submitted Model/Terrain/HMS tree at the ZIP root.
    made = shutil.make_archive(str(base), "zip", root_dir=str(model_dir), base_dir=".")
    made_path = Path(made)
    if made_path != output_zip:
        if output_zip.exists():
            output_zip.unlink()
        made_path.replace(output_zip)
    return output_zip


def _read_text_path(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _resolve_external_reference(base_dir: Path, reference: str) -> tuple[str, bool, bool]:
    """Resolve a HEC Windows-style path relative to the file that owns the reference."""
    import re
    ref = str(reference or "").strip().strip('"').strip()
    if not ref:
        return "", False, False
    absolute = bool(re.match(r"^[A-Za-z]:[\\/]", ref) or ref.startswith("\\\\"))
    if absolute:
        p = Path(ref)
    else:
        # HEC project files use Windows backslashes even when inspected on another OS.
        parts = [x for x in ref.replace("\\", "/").split("/") if x not in ("", ".")]
        p = base_dir
        for part in parts:
            p = p.parent if part == ".." else p / part
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    return str(resolved), resolved.exists(), absolute


def _workspace_file_candidates(workspace: Path, reference: str) -> list[Path]:
    """Find case-insensitive basename matches anywhere in the extracted workspace."""
    ref = str(reference or "").strip().strip('"').strip()
    if not ref:
        return []
    basename = ref.replace("\\", "/").rstrip("/").split("/")[-1]
    if not basename:
        return []
    matches = [p.resolve() for p in workspace.rglob("*") if p.is_file() and p.name.lower() == basename.lower()]
    # Prefer candidates whose trailing path also matches the non-.. part of the reference.
    tail = [x.lower() for x in ref.replace("\\", "/").split("/") if x not in ("", ".", "..")]
    def score(path: Path):
        parts = [x.lower() for x in path.parts]
        suffix_match = 0
        for n in range(1, min(len(tail), len(parts)) + 1):
            if parts[-n:] == tail[-n:]:
                suffix_match = n
        return (-suffix_match, len(path.parts), str(path).lower())
    return sorted(dict.fromkeys(matches), key=score)


def _link_or_copy(src: str, dst: str) -> str:
    """Cheaply clone dependency files inside the temporary workspace when possible."""
    try:
        os.link(src, dst)
        return dst
    except Exception:
        return shutil.copy2(src, dst)


def _stage_unique_dependency(workspace: Path, expected: Path, candidate: Path) -> dict[str, Any]:
    """Stage a uniquely located dependency into the relative location expected by HEC-RAS.

    If the candidate lives in a folder with the same name as the expected dependency folder,
    clone the whole folder tree so terrain sidecars / companion DSS files are preserved.
    Otherwise stage only the referenced file. The submitted ZIP is never changed; this only
    mutates the extracted temporary run workspace.
    """
    expected = expected.resolve()
    candidate = candidate.resolve()
    info: dict[str, Any] = {"staged": False, "staged_from": str(candidate), "staged_to": str(expected), "stage_mode": None}
    if expected.exists():
        info.update(staged=True, stage_mode="already_exists")
        return info
    expected.parent.mkdir(parents=True, exist_ok=True)

    # Find a candidate ancestor matching the expected parent folder name. This recovers
    # common layouts such as Wrapper/Test1D2D/terrain/... when the model expects ../terrain/....
    source_folder = None
    expected_folder_name = expected.parent.name.lower()
    cur = candidate.parent
    while workspace == cur or workspace in cur.parents:
        if cur.name.lower() == expected_folder_name:
            source_folder = cur
            break
        if cur == workspace:
            break
        cur = cur.parent

    if source_folder is not None and source_folder.is_dir():
        target_folder = expected.parent
        shutil.copytree(source_folder, target_folder, dirs_exist_ok=True, copy_function=_link_or_copy)
        info.update(staged=expected.exists(), stage_mode="folder_clone")
    else:
        if not expected.exists():
            _link_or_copy(str(candidate), str(expected))
        info.update(staged=expected.exists(), stage_mode="file_clone")
    return info


def _reference_row(workspace: Path, base_dir: Path, kind: str, plan: str, reference: str, auto_stage: bool) -> dict[str, Any]:
    resolved_str, exists, absolute = _resolve_external_reference(base_dir, reference)
    expected = Path(resolved_str) if resolved_str else None
    row: dict[str, Any] = {
        "kind": kind, "plan": plan, "reference": reference,
        "resolved_path": resolved_str, "exists": bool(exists), "absolute_reference": bool(absolute),
        "resolution_status": "exact" if exists else "missing",
        "found_elsewhere": "", "candidate_count": 0, "auto_staged": False,
    }
    if exists or absolute or expected is None:
        return row

    candidates = _workspace_file_candidates(workspace, reference)
    row["candidate_count"] = len(candidates)
    if len(candidates) == 1:
        row["found_elsewhere"] = str(candidates[0])
        row["resolution_status"] = "unique_relocated"
        if auto_stage:
            try:
                stage = _stage_unique_dependency(workspace, expected, candidates[0])
                row["auto_staged"] = bool(stage.get("staged"))
                row["stage_mode"] = stage.get("stage_mode")
                row["exists"] = expected.exists()
                row["resolution_status"] = "auto_staged" if row["exists"] else "stage_failed"
            except Exception as exc:
                row["stage_error"] = str(exc)
                row["resolution_status"] = "stage_failed"
    elif len(candidates) > 1:
        row["resolution_status"] = "ambiguous_relocated"
        row["found_elsewhere"] = " | ".join(str(x) for x in candidates[:5])
    return row


def workspace_reference_preflight(model_dir: str | Path, plan_codes: list[str] | tuple[str, ...] | None = None, auto_stage: bool = True, project_file: str | Path | None = None) -> dict[str, Any]:
    """Check common external run dependencies for an extracted HEC-RAS workspace.

    Currently checks:
    - DSS File= references in selected unsteady-flow files.
    - RAS Mapper TerrainLayer filenames.
    - RAS Mapper projection filename.

    This is deliberately a preflight, not a guarantee that every third-party/external
    dependency required by a particular HEC-RAS version has been captured.
    """
    import re
    from .text_parser import parse_project, parse_plan

    workspace = Path(model_dir).resolve()
    project_file = Path(project_file).resolve() if project_file else find_project_file(workspace)
    project_text = _read_text_path(project_file)
    project = parse_project(project_text)
    requested = [str(x).lower().lstrip(".") for x in (plan_codes or project.get("plan_files", []))]
    rows: list[dict[str, Any]] = []

    # Selected plan -> flow file -> external DSS dependencies.
    for code in requested:
        plan_file = find_plan_file(workspace, code, project_file=project_file)
        plan = parse_plan(_read_text_path(plan_file), code)
        flow_code = plan.get("flow_file")
        if not flow_code:
            continue
        flow_file = project_file.parent / f"{project_file.stem}.{flow_code}"
        if not flow_file.exists():
            rows.append({
                "kind": "Flow file", "plan": code, "reference": flow_file.name,
                "resolved_path": str(flow_file), "exists": False, "absolute_reference": False,
                "resolution_status": "missing", "found_elsewhere": "", "candidate_count": 0, "auto_staged": False,
            })
            continue
        flow_text = _read_text_path(flow_file)
        refs = []
        for m in re.finditer(r"(?im)^DSS File\s*=\s*(.+?)\s*$", flow_text):
            ref = m.group(1).strip()
            if ref and ref.lower() not in {"dss", "none"} and ref not in refs:
                refs.append(ref)
        for ref in refs:
            rows.append(_reference_row(workspace, flow_file.parent, "DSS", code, ref, auto_stage))

    # RAS Mapper terrain/projection references are project-wide rather than plan-specific.
    rasmap = project_file.with_suffix(".rasmap")
    if rasmap.exists():
        text = _read_text_path(rasmap)
        terrain_refs = re.findall(r'(?i)<Layer\b[^>]*Type="TerrainLayer"[^>]*Filename="([^"]+)"', text)
        for ref in dict.fromkeys(terrain_refs):
            rows.append(_reference_row(workspace, rasmap.parent, "Terrain", "All", ref, auto_stage))
        proj_match = re.search(r'(?i)<RASProjectionFilename\b[^>]*Filename="([^"]+)"', text)
        if proj_match:
            ref = proj_match.group(1)
            rows.append(_reference_row(workspace, rasmap.parent, "Projection", "All", ref, auto_stage))

    missing = [r for r in rows if not r["exists"]]
    absolute = [r for r in rows if r["absolute_reference"]]
    staged = [r for r in rows if r.get("auto_staged")]
    ambiguous = [r for r in rows if r.get("resolution_status") == "ambiguous_relocated"]
    return {
        "workspace_root": str(workspace),
        "project_file": str(project_file),
        "project_directory": str(project_file.parent),
        "checked_plan_codes": requested,
        "references": rows,
        "reference_count": len(rows),
        "missing_count": len(missing),
        "auto_staged_count": len(staged),
        "ambiguous_count": len(ambiguous),
        "absolute_reference_count": len(absolute),
        "ready": len(missing) == 0,
    }

# ---------------------------------------------------------------------------
# v1.0 local workspace / dependency manager
# ---------------------------------------------------------------------------

def _clean_folder_list(paths: list[str | Path] | tuple[str | Path, ...] | None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths or []:
        text = str(raw or "").strip().strip('"')
        if not text:
            continue
        p = Path(text).expanduser().resolve()
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _reference_parts(reference: str) -> tuple[int, list[str]]:
    """Return leading ``..`` count and the remaining Windows-style path parts."""
    parts = [x for x in str(reference or "").strip().strip('"').replace("\\", "/").split("/") if x not in ("", ".")]
    hops = 0
    while hops < len(parts) and parts[hops] == "..":
        hops += 1
    return hops, parts[hops:]


def _candidate_from_search_root(search_root: Path, reference: str) -> list[Path]:
    """Generate deterministic candidate paths without recursively walking huge folders.

    A reviewer can select either the actual dependency folder (e.g. ``D:\\...\\Terrain``)
    or a broader parent/search location. We try both layouts directly. This avoids an
    expensive ``rglob`` over 100+ GB MAAPnext repositories.
    """
    _hops, tail = _reference_parts(reference)
    if not tail:
        return []
    root = search_root.resolve()
    candidates: list[Path] = []

    # Search root is a common parent: D:\Project + Terrain\terrain.hdf
    candidates.append(root.joinpath(*tail))

    # Search root is the dependency folder itself: D:\Project\Terrain + terrain.hdf
    if root.name.lower() == tail[0].lower():
        candidates.append(root.joinpath(*tail[1:]))

    # Useful for a folder containing just the referenced file.
    candidates.append(root / tail[-1])

    # De-duplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def collect_external_references(
    project_file: str | Path,
    plan_codes: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Collect the external file references needed for local review/run validation.

    The current implementation intentionally focuses on the dependencies that have been
    observed in the development/FEMA-style workspaces: DSS files referenced by unsteady
    flow files plus RAS Mapper terrain and projection references. The returned owner file
    makes the resolution auditable and allows the same logic to be extended later.
    """
    from .text_parser import parse_project, parse_plan

    project_file = Path(project_file).expanduser().resolve()
    if not project_file.is_file():
        raise FileNotFoundError(project_file)
    project = parse_project(_read_text_path(project_file))
    requested = [str(x).lower().lstrip(".") for x in (plan_codes or project.get("plan_files", []))]
    rows: list[dict[str, Any]] = []

    for code in requested:
        plan_file = find_plan_file(project_file.parent, code, project_file=project_file)
        plan = parse_plan(_read_text_path(plan_file), code)
        flow_code = plan.get("flow_file")
        if not flow_code:
            continue
        flow_file = project_file.parent / f"{project_file.stem}.{flow_code}"
        if not flow_file.exists():
            rows.append({
                "kind": "Flow file", "plan": code, "owner_file": str(plan_file),
                "reference": flow_file.name, "absolute_reference": False,
                "expected_original": str(flow_file), "exists_original": False,
            })
            continue
        flow_text = _read_text_path(flow_file)
        refs: list[str] = []
        for m in re.finditer(r"(?im)^DSS File\s*=\s*(.+?)\s*$", flow_text):
            ref = m.group(1).strip()
            if ref and ref.lower() not in {"dss", "none"} and ref not in refs:
                refs.append(ref)
        for ref in refs:
            expected, exists, absolute = _resolve_external_reference(flow_file.parent, ref)
            rows.append({
                "kind": "DSS", "plan": code, "owner_file": str(flow_file),
                "reference": ref, "absolute_reference": absolute,
                "expected_original": expected, "exists_original": exists,
            })

    rasmap = project_file.with_suffix(".rasmap")
    if rasmap.exists():
        text = _read_text_path(rasmap)
        terrain_refs = re.findall(r'(?i)<Layer\b[^>]*Type="TerrainLayer"[^>]*Filename="([^"]+)"', text)
        for ref in dict.fromkeys(terrain_refs):
            expected, exists, absolute = _resolve_external_reference(rasmap.parent, ref)
            rows.append({
                "kind": "Terrain", "plan": "All", "owner_file": str(rasmap),
                "reference": ref, "absolute_reference": absolute,
                "expected_original": expected, "exists_original": exists,
            })
        proj_match = re.search(r'(?i)<RASProjectionFilename\b[^>]*Filename="([^"]+)"', text)
        if proj_match:
            ref = proj_match.group(1)
            expected, exists, absolute = _resolve_external_reference(rasmap.parent, ref)
            rows.append({
                "kind": "Projection", "plan": "All", "owner_file": str(rasmap),
                "reference": ref, "absolute_reference": absolute,
                "expected_original": expected, "exists_original": exists,
            })
    return rows


def validate_local_workspace(
    project_file: str | Path,
    dependency_folders: list[str | Path] | tuple[str | Path, ...] | None = None,
    plan_codes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate a local RAS model against explicitly selected dependency folders.

    Resolution order:
    1. the model's original relative/absolute path;
    2. deterministic paths under each selected dependency folder/search location.

    This function deliberately does not recursively search large repositories. If two
    selected folders satisfy the same missing reference, the row is marked ambiguous.
    """
    project_file = Path(project_file).expanduser().resolve()
    dependencies = _clean_folder_list(dependency_folders)
    refs = collect_external_references(project_file, plan_codes)
    resolved_rows: list[dict[str, Any]] = []

    for row in refs:
        item = dict(row)
        expected = Path(item["expected_original"]) if item.get("expected_original") else None
        if item.get("exists_original") and expected is not None:
            item.update({
                "status": "exact", "resolved_source": str(expected),
                "search_folder": "", "candidate_count": 1,
            })
            resolved_rows.append(item)
            continue

        if item.get("absolute_reference"):
            item.update({
                "status": "missing_absolute", "resolved_source": "",
                "search_folder": "", "candidate_count": 0,
            })
            resolved_rows.append(item)
            continue

        matches: list[tuple[Path, Path]] = []
        seen: set[str] = set()
        for search_root in dependencies:
            if not search_root.is_dir():
                continue
            for candidate in _candidate_from_search_root(search_root, item.get("reference", "")):
                if candidate.is_file():
                    key = str(candidate.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        matches.append((candidate.resolve(), search_root))
        if len(matches) == 1:
            candidate, root = matches[0]
            item.update({
                "status": "resolved_dependency", "resolved_source": str(candidate),
                "search_folder": str(root), "candidate_count": 1,
            })
        elif len(matches) > 1:
            item.update({
                "status": "ambiguous", "resolved_source": " | ".join(str(x[0]) for x in matches[:5]),
                "search_folder": " | ".join(str(x[1]) for x in matches[:5]), "candidate_count": len(matches),
            })
        else:
            item.update({
                "status": "missing", "resolved_source": "", "search_folder": "", "candidate_count": 0,
            })
        resolved_rows.append(item)

    unresolved = [r for r in resolved_rows if r["status"] not in {"exact", "resolved_dependency"}]
    ambiguous = [r for r in resolved_rows if r["status"] == "ambiguous"]
    missing = [r for r in resolved_rows if r["status"] in {"missing", "missing_absolute"}]
    return {
        "project_file": str(project_file),
        "project_directory": str(project_file.parent),
        "dependency_folders": [str(x) for x in dependencies],
        "references": resolved_rows,
        "reference_count": len(resolved_rows),
        "resolved_count": len(resolved_rows) - len(unresolved),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "ready": len(unresolved) == 0,
    }


def _create_directory_junction(link: Path, target: Path) -> str:
    """Create a directory junction/symlink without copying dependency data."""
    link = Path(link)
    target = Path(target).resolve()
    if not target.is_dir():
        raise NotADirectoryError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == target:
                return "existing"
        except Exception:
            pass
        raise FileExistsError(f"Cannot create dependency link because target path already exists: {link}")

    if os.name == "nt":
        # /J works without Developer Mode/admin rights for local NTFS directories.
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True, errors="replace",
        )
        if proc.returncode == 0 and link.exists():
            return "junction"
        # Network/UNC targets do not support junctions. Try a directory symlink as a
        # fallback; this may require Windows Developer Mode or elevated privileges.
        proc2 = subprocess.run(
            ["cmd", "/c", "mklink", "/D", str(link), str(target)],
            capture_output=True, text=True, errors="replace",
        )
        if proc2.returncode == 0 and link.exists():
            return "directory_symlink"
        raise RuntimeError((proc.stderr or proc.stdout or proc2.stderr or proc2.stdout or "mklink failed").strip())

    os.symlink(target, link, target_is_directory=True)
    return "symlink"


def _source_root_for_reference(source_file: Path, reference: str) -> tuple[Path | None, list[str]]:
    """Find the source directory corresponding to the first non-``..`` reference part."""
    _hops, tail = _reference_parts(reference)
    if not tail:
        return None, tail
    root_name = tail[0].lower()
    cur = source_file.parent
    while True:
        if cur.name.lower() == root_name:
            return cur, tail
        if cur.parent == cur:
            break
        cur = cur.parent
    # If the reference is simply ../file.ext there is no dependency-root directory.
    return None, tail


def _resolve_copy_reference(owner_copy_dir: Path, reference: str) -> Path:
    resolved, _exists, absolute = _resolve_external_reference(owner_copy_dir, reference)
    if absolute:
        return Path(resolved)
    return Path(resolved)


def create_protected_run_workspace(
    project_file: str | Path,
    dependency_folders: list[str | Path] | tuple[str | Path, ...] | None = None,
    plan_codes: list[str] | tuple[str, ...] | None = None,
    destination_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Create a lightweight protected HEC-RAS run workspace.

    Only the HEC-RAS model directory is copied. Large external dependency directories are
    exposed to the copied model through Windows junctions (or symlinks on non-Windows test
    hosts). The original model and dependency data remain untouched.
    """
    project_file = Path(project_file).expanduser().resolve()
    model_dir = project_file.parent
    validation = validate_local_workspace(project_file, dependency_folders, plan_codes)
    if not validation["ready"]:
        raise RuntimeError(
            f"Local workspace validation failed: {validation['missing_count']} missing and "
            f"{validation['ambiguous_count']} ambiguous dependency reference(s)."
        )

    # Add enough parent depth so one or more leading '..' references stay inside the sandbox.
    max_hops = 1
    for row in validation["references"]:
        if not row.get("absolute_reference"):
            hops, _tail = _reference_parts(row.get("reference", ""))
            max_hops = max(max_hops, hops)

    if destination_parent:
        base_parent = Path(destination_parent).expanduser().resolve()
        base_parent.mkdir(parents=True, exist_ok=True)
        run_root = Path(tempfile.mkdtemp(prefix=f"{project_file.stem}_reviewrun_", dir=str(base_parent)))
    else:
        run_root = Path(tempfile.mkdtemp(prefix=f"{project_file.stem}_reviewrun_"))

    container = run_root
    for i in range(max_hops):
        container = container / f"_relative_{i+1}"
        container.mkdir(exist_ok=True)
    run_model_dir = container / model_dir.name

    dependency_roots = _clean_folder_list(dependency_folders)
    # If a selected dependency directory lives inside the model folder, do not duplicate it.
    excluded_dirs: set[Path] = set()
    for dep in dependency_roots:
        try:
            dep.relative_to(model_dir)
            excluded_dirs.add(dep)
        except ValueError:
            pass

    def ignore_selected_dependencies(src: str, names: list[str]):
        srcp = Path(src).resolve()
        ignored: list[str] = []
        for name in names:
            child = (srcp / name).resolve()
            if child in excluded_dirs:
                ignored.append(name)
        return ignored

    shutil.copytree(model_dir, run_model_dir, ignore=ignore_selected_dependencies)
    run_project_file = run_model_dir / project_file.name

    link_records: list[dict[str, Any]] = []
    created_links: dict[str, str] = {}
    for row in validation["references"]:
        if row.get("absolute_reference"):
            # Absolute references continue to point at the same source path. They were
            # already validated above, so no staging is needed.
            link_records.append({**row, "run_status": "absolute_passthrough", "run_path": row.get("resolved_source", "")})
            continue
        source_file = Path(row["resolved_source"]).resolve()
        owner_original = Path(row["owner_file"]).resolve()
        try:
            owner_rel = owner_original.relative_to(model_dir)
        except ValueError:
            owner_rel = Path(owner_original.name)
        owner_copy = run_model_dir / owner_rel
        expected_copy = _resolve_copy_reference(owner_copy.parent, row["reference"])
        if expected_copy.exists():
            link_records.append({**row, "run_status": "copied_internal", "run_path": str(expected_copy)})
            continue

        hops, tail = _reference_parts(row["reference"])
        if not tail:
            link_records.append({**row, "run_status": "invalid_reference", "run_path": str(expected_copy)})
            continue

        # Determine the directory root represented by the first tail component. For a
        # normal ../Terrain/file.hdf reference, this maps the copied ../Terrain folder to
        # the original Terrain directory with a junction.
        source_root, _ = _source_root_for_reference(source_file, row["reference"])
        # If the reviewer explicitly selected the actual dependency directory but its
        # on-disk folder name differs from the HEC-RAS reference root, use that selected
        # folder when the referenced file is directly/nested beneath it in the expected
        # remainder layout. This keeps the local workflow universal without renaming data.
        if source_root is None and row.get("search_folder"):
            selected_root = Path(row["search_folder"]).resolve()
            try:
                rel = source_file.relative_to(selected_root)
                remainder = Path(*tail[1:]) if len(tail) > 1 else Path(tail[-1])
                if str(rel).lower() == str(remainder).lower() or (len(tail) == 2 and rel.name.lower() == tail[-1].lower()):
                    source_root = selected_root
            except ValueError:
                pass
        if source_root is not None:
            base = owner_copy.parent
            for _ in range(hops):
                base = base.parent
            target_root = base / tail[0]
            key = str(target_root).lower()
            existing_target = created_links.get(key)
            if existing_target and Path(existing_target).resolve() != source_root.resolve():
                raise RuntimeError(f"Conflicting dependency mappings for {target_root}: {existing_target} vs {source_root}")
            if not target_root.exists():
                mode = _create_directory_junction(target_root, source_root)
                created_links[key] = str(source_root)
            else:
                mode = "existing"
            if not expected_copy.exists():
                raise RuntimeError(f"Dependency link was created but the referenced file is still not visible: {expected_copy}")
            link_records.append({**row, "run_status": mode, "run_path": str(expected_copy), "linked_folder": str(target_root), "link_source": str(source_root)})
            continue

        # Rare ../file.ext layout: avoid copying huge dependency files when possible.
        expected_copy.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            try:
                os.symlink(source_file, expected_copy)
                mode = "file_symlink"
            except OSError:
                # A hard link is safe for input dependencies and requires no elevation,
                # but only works on the same volume. If that also fails we refuse to make
                # an expensive hidden copy of an arbitrarily large dependency.
                try:
                    os.link(source_file, expected_copy)
                    mode = "file_hardlink"
                except OSError as exc:
                    raise RuntimeError(
                        f"Cannot link dependency file {source_file} to {expected_copy}. "
                        "Select its containing dependency folder so a directory junction can be used."
                    ) from exc
        else:
            os.symlink(source_file, expected_copy)
            mode = "file_symlink"
        link_records.append({**row, "run_status": mode, "run_path": str(expected_copy), "link_source": str(source_file)})

    final_preflight = workspace_reference_preflight(
        run_root, plan_codes=plan_codes, auto_stage=False, project_file=run_project_file,
    )
    if not final_preflight["ready"]:
        raise RuntimeError(
            f"Protected run workspace was created, but HEC-RAS preflight still reports "
            f"{final_preflight['missing_count']} unresolved reference(s)."
        )

    return {
        "run_root": str(run_root),
        "model_directory": str(run_model_dir),
        "project_file": str(run_project_file),
        "source_project_file": str(project_file),
        "source_model_directory": str(model_dir),
        "dependency_folders": [str(x) for x in dependency_roots],
        "reference_validation": validation,
        "links": link_records,
        "preflight": final_preflight,
        "protected_original": True,
    }
