"""Per-run source-code snapshot + git state capture (exp-control / pyctrl side).

Direct analog of the SLM server's ``code_snapshot.py``
(``SLMnet/src/slmnet/experimental/tools/code_snapshot.py``), adapted for the
pyctrl backend.  Purpose: every scan records (in its ``data_<stamp>.json``
sidecar) which exact source files defined the sequence behavior that produced
it.  Git is useful but commits aren't frequent enough to correlate cleanly,
and the working tree is routinely dirty -- the content blobs capture the
ACTUAL bytes regardless of git.

Storage model (identical to the SLM server's):

  * **Content-addressed blobs.**  For each file we SHA256 the bytes and write
    them to ``<data_root>/_code_snapshots/<sha256><ext>`` exactly once;
    subsequent runs with identical content reuse the blob (free dedup).
  * **Per-run folder.**  ``<data_root>/_code_snapshots/_runs/<scan_id>/`` is a
    readable reconstruction of the snapshotted tree at its ORIGINAL relative
    paths (``YbSeqs/CoolingSeq.py`` ...), hardlinked to the dedup'd blobs, plus
    a ``manifest.json`` with full provenance (scan_id, seq, hashes, git state).
    Because the tree is reconstructed faithfully it is also directly
    importable -- which is what makes snapshot REPLAY (#3) possible without a
    temp copy: point ``sys.path`` at the per-run folder's experiment dirs.

Two roles per file (the boundary is :mod:`seq_reload`'s, and it is load-bearing):

  * **experiment** -- ``YbSteps`` / ``YbSeqs`` / ``YbScans`` / ``YbRearrangement``.
    Hot-reloadable per job, so they are also safe to REPLAY from a snapshot
    (``materialize_*`` + :func:`snapshot_syspath`).
  * **framework / config / runtime** -- ``lib`` / ``expConfig.py`` / ``config.yml``
    / ``YbExptCtrl``.  Captured for the RECORD only.  Reloading ``lib`` would
    mint new ``ExpSeq`` / ``ScanGroup`` classes and break ``isinstance`` against
    live instances (the exact reason ``seq_reload`` keeps it), so replay never
    swaps these; it only DETECTS a mismatch and warns.

Best-effort throughout: an unreadable file, a hashing error, a hardlink
failure -- all log at WARNING and go into ``errors`` but NEVER raise.  This
module must never break or delay the actual experiment flow.  Git inspection
is strictly READ-ONLY (never commits, tags, or touches the working tree).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys

logger = logging.getLogger("pyctrl.code_snapshot")

# Replayable experiment-definition dirs -- MUST match seq_reload._EXPERIMENT_DIRS
# (the per-job hot-reload boundary). Snapshotting + replaying these reproduces
# day-to-day seq/step edits safely.
_EXPERIMENT_DIRS = ("YbSteps", "YbSeqs", "YbScans", "YbRearrangement")
# Record-only: the framework (lib), the runtime harness (YbExptCtrl). Captured
# so "what code ran" is complete, but never swapped on replay.
_RECORD_DIRS = ("lib", "YbExptCtrl")
# Record-only single files at the pyctrl root.
_RECORD_FILES = ("expConfig.py", "config.yml")

_SNAPSHOT_DIRNAME = "_code_snapshots"
_RUNS_DIRNAME = "_runs"

# Environment override for the snapshot base dir (an explicit absolute path). Point it at
# ``<data_root>/_code_snapshots`` to restore the legacy on-OneDrive location.
_SNAPSHOT_DIR_ENV = "YB_CODE_SNAPSHOT_DIR"


def _local_default_base() -> str:
    """The DEFAULT (local) snapshot base: ``<superproject>/log/code_snapshots``.

    ``pyctrl_root()`` is ``…/pyctrl``; its parent is the experiment-control superproject, whose
    ``log/`` dir already holds the other runtime artifacts (``log/pyctrl_log`` …) and lives on a
    fast LOCAL disk -- unlike the OneDrive-synced data share that ``data_root`` points at."""
    return os.path.join(os.path.dirname(pyctrl_root()), "log", "code_snapshots")


def snapshot_base(data_root: str) -> str:
    """Resolve the ``_code_snapshots`` base dir (content blobs + per-run hardlink trees).

    DEFAULT is the LOCAL :func:`_local_default_base`, NOT ``<data_root>/_code_snapshots``. The
    ``data_root`` is the OneDrive-synced Data share, and building the ~130-file per-run hardlink
    tree through OneDrive's filter driver cost ~10 s per scan -- it blocked the FIRST SHOT of
    every scan. The snapshot is best-effort provenance (not shared data), so a local disk is its
    right home. Override with ``$YB_CODE_SNAPSHOT_DIR`` (absolute path); set it to
    ``<data_root>/_code_snapshots`` to restore the old behavior. Writer and replay readers BOTH
    resolve through here, so they always agree on where a snapshot lives."""
    override = os.environ.get(_SNAPSHOT_DIR_ENV)
    if override:
        return override
    try:
        return _local_default_base()
    except Exception:  # noqa: BLE001 - any path error -> legacy on-data_root location
        return os.path.join(data_root, _SNAPSHOT_DIRNAME)


def _rel_or_abs(path: str, start: str) -> str:
    """``relpath(path, start)`` as posix; the absolute posix path if they're on different drives.

    Windows ``os.path.relpath`` raises ``ValueError`` across drives -- which the local snapshot
    base now is, relative to the OneDrive ``data_root``. The recorded path is informational
    (replay recomputes via :func:`run_folder`), so an absolute fallback is fine."""
    try:
        return os.path.relpath(path, start).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


# Set to the source scan_id while a :func:`snapshot_syspath` replay is active (the runner
# runs one job at a time, so a module global is safe). Lets scan-prep record a replayed run
# HONESTLY -- as a pointer to the code that actually executed -- instead of re-snapshotting
# the live disk (which the replay did not run). Read via :func:`active_replay_source`.
_active_replay_source = None


def active_replay_source():
    """The source scan_id of the snapshot currently being replayed, or None."""
    return _active_replay_source


def pyctrl_root() -> str:
    """…/pyctrl/lib/code_snapshot.py -> …/pyctrl (the snapshot project root)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _role_of(rel_posix: str) -> str:
    head = rel_posix.split("/", 1)[0]
    if head in _EXPERIMENT_DIRS:
        return "experiment"
    if head in _RECORD_DIRS:
        return "framework" if head == "lib" else "runtime"
    return "config"


def code_files(root: str | None = None) -> list[str]:
    """Build the snapshot file list: every ``*.py`` under the experiment +
    record dirs, plus the record files, as posix paths relative to ``root``
    (default :func:`pyctrl_root`).  Only existing paths are returned, sorted.
    """
    root = root or pyctrl_root()
    out: list[str] = []
    for d in (*_EXPERIMENT_DIRS, *_RECORD_DIRS):
        abs_d = os.path.join(root, d)
        if not os.path.isdir(abs_d):
            continue
        for dirpath, _dirnames, filenames in os.walk(abs_d):
            # Skip caches / vendored env trees defensively.
            if "__pycache__" in dirpath or ".venv" in dirpath:
                continue
            for fn in filenames:
                if fn.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    out.append(rel.replace("\\", "/"))
    for f in _RECORD_FILES:
        if os.path.isfile(os.path.join(root, f)):
            out.append(f)
    return sorted(set(out))


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_run_id(run_id) -> str:
    s = str(run_id)
    if not s:
        return "_unset"
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _link_or_copy(src: str, dst: str) -> str:
    """Materialise ``dst`` from ``src``: hardlink (free on NTFS same-volume) ->
    symlink -> copy.  Idempotent.  Returns the mechanism used."""
    if os.path.exists(dst):
        try:
            if os.path.getsize(dst) == os.path.getsize(src):
                return "exists"
        except OSError:
            pass
        try:
            os.unlink(dst)
        except OSError:
            pass
    try:
        os.link(src, dst)
        return "hardlink"
    except (OSError, AttributeError, NotImplementedError):
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, AttributeError, NotImplementedError):
        pass
    shutil.copyfile(src, dst)
    return "copy"


def snapshot_code(project_root: str,
                  data_root: str,
                  files: list[str] | None = None,
                  *,
                  run_id=None,
                  run_started_iso: str | None = None,
                  seq_name: str | None = None,
                  git_state: dict | None = None,
                  build_run_folder: bool = True) -> dict:
    """Hash + dedup-store every file in ``files`` (default :func:`code_files`).

    Blobs land in ``<data_root>/_code_snapshots/<sha><ext>`` (write-once).
    When ``run_id`` is given AND ``build_run_folder``, also reconstruct a
    readable + importable per-run tree under
    ``<data_root>/_code_snapshots/_runs/<safe_run_id>/`` (files at their
    original relative paths, hardlinked to blobs) with a ``manifest.json``.

    Returns a compact dict suitable for embedding in the scan sidecar::

        {
          "scan_id": <run_id>,
          "snapshot_dir":  "_code_snapshots",
          "run_dir":       "_code_snapshots/_runs/<safe>" | None,
          "run_manifest":  ".../manifest.json" | None,
          "n_files": <int>, "n_experiment": <int>,
          "git": <git_state or None>,
          "hashes": {"<rel>": "<sha>"},   # role=='experiment' only (the rest are in the manifest)
          "missing": [...], "errors": [...],
        }

    Best-effort: never raises (a top-level failure returns an ``errors`` dict).
    """
    try:
        snap_dir = snapshot_base(data_root)
        os.makedirs(snap_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("code_snapshot: cannot create snapshot dir: %s", exc)
        return {"scan_id": run_id, "snapshot_dir": None, "run_dir": None,
                "run_manifest": None, "n_files": 0, "n_experiment": 0,
                "git": git_state, "hashes": {}, "missing": [],
                "errors": [{"path": data_root,
                            "error": "mkdir: %s: %s" % (type(exc).__name__, exc)}]}

    files = list(files) if files is not None else code_files(project_root)
    records: list[dict] = []      # one per stored file
    missing: list[str] = []
    errors: list[dict] = []
    exp_hashes: dict[str, str] = {}

    for rel in files:
        rel = rel.replace("\\", "/")
        src = os.path.join(project_root, rel)
        if not os.path.isfile(src):
            missing.append(rel)
            continue
        try:
            with open(src, "rb") as f:
                data = f.read()
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel, "error": "read: %s: %s" % (type(exc).__name__, exc)})
            continue
        digest = _hash_bytes(data)
        ext = os.path.splitext(src)[1] or ".dat"
        blob = os.path.join(snap_dir, "%s%s" % (digest, ext))
        if not os.path.exists(blob):
            try:
                tmp = blob + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, blob)
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": rel,
                               "error": "write blob: %s: %s" % (type(exc).__name__, exc)})
                continue
        role = _role_of(rel)
        records.append({"src_rel": rel, "sha256": digest, "ext": ext, "role": role})
        if role == "experiment":
            exp_hashes[rel] = digest

    run_dir_rel = None
    run_manifest_rel = None
    if run_id is not None and build_run_folder:
        run_dir_rel, run_manifest_rel = _build_run_folder(
            snap_dir, data_root, run_id, records, missing, errors,
            run_started_iso=run_started_iso, seq_name=seq_name, git_state=git_state)

    n_exp = sum(1 for r in records if r["role"] == "experiment")
    return {
        "scan_id": run_id,
        "snapshot_dir": _SNAPSHOT_DIRNAME,
        "run_dir": run_dir_rel,
        "run_manifest": run_manifest_rel,
        "n_files": len(records),
        "n_experiment": n_exp,
        "git": git_state,
        "hashes": exp_hashes,
        "missing": missing,
        "errors": errors,
    }


def _build_run_folder(snap_dir, data_root, run_id, records, missing, errors,
                      *, run_started_iso, seq_name, git_state):
    """Reconstruct the snapshotted tree at original rel paths (hardlinked to
    blobs) under ``_runs/<safe>/`` and write ``manifest.json``.  Returns
    ``(run_dir_rel, manifest_rel)`` relative to ``data_root`` (or None)."""
    safe = _safe_run_id(run_id)
    run_dir = os.path.join(snap_dir, _RUNS_DIRNAME, safe)
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        errors.append({"path": run_dir,
                       "error": "mkdir run_dir: %s: %s" % (type(exc).__name__, exc)})
        return None, None
    for rec in records:
        rel = rec["src_rel"]
        blob = os.path.join(snap_dir, "%s%s" % (rec["sha256"], rec["ext"]))
        dst = os.path.join(run_dir, rel.replace("/", os.sep))
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            rec["materialise"] = _link_or_copy(blob, dst)
        except Exception as exc:  # noqa: BLE001
            rec["materialise"] = "failed"
            errors.append({"path": rel, "error": "materialise: %s" % exc})
    manifest = {
        "scan_id": run_id,
        "safe_run_id": safe,
        "run_started_iso": run_started_iso,
        "seq_name": seq_name,
        "experiment_dirs": list(_EXPERIMENT_DIRS),
        "files": sorted(records, key=lambda r: r["src_rel"]),
        "missing": missing,
        "errors": errors,
        "git": git_state,
        "snapshot_dir_rel": _SNAPSHOT_DIRNAME,
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    try:
        tmp = manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, manifest_path)
    except Exception as exc:  # noqa: BLE001
        errors.append({"path": manifest_path,
                       "error": "write manifest: %s: %s" % (type(exc).__name__, exc)})
    rel_run = _rel_or_abs(run_dir, data_root)
    rel_man = _rel_or_abs(manifest_path, data_root)
    return rel_run, rel_man


# =========================================================================== #
# git state (read-only; never modifies the tree). Verbatim from the SLM server
# plus a best-effort submodule/superproject pair (pyctrl is a submodule here).
# =========================================================================== #
def read_git_state(project_root: str | None = None) -> dict | None:
    """Read-only git inspection of ``project_root`` (default :func:`pyctrl_root`).

    Returns ``{commit, branch, dirty, status, remote, superproject_commit}`` or
    ``None`` if not a git repo / git unavailable.  Never blocks the experiment
    (3 s per call) and never mutates git state."""
    root = project_root or pyctrl_root()
    if not os.path.exists(os.path.join(root, ".git")):
        return None

    def _git(*args, cwd=root, timeout=3.0):
        try:
            r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                               text=True, timeout=timeout, check=False)
            return r.stdout.rstrip("\n") if r.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("code_snapshot: git %s failed: %s", " ".join(args), exc)
            return None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain") or ""
    super_commit = _git("rev-parse", "HEAD", cwd=os.path.dirname(root))
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "(detached)",
        "dirty": bool(status.strip()),
        "status": status,
        "remote": _git("remote", "get-url", "origin"),
        "superproject_commit": super_commit,
    }


# =========================================================================== #
# Snapshot REPLAY (#3) -- import experiment code from a per-run snapshot folder.
# =========================================================================== #
def run_folder(data_root: str, run_id) -> str:
    """Absolute path of a run's snapshot folder under the CURRENT base (no existence check).
    For READING a run that may predate the local-dir switch, use :func:`_existing_run_folder`."""
    return os.path.join(snapshot_base(data_root), _RUNS_DIRNAME, _safe_run_id(run_id))


def existing_run_folder(data_root: str, run_id) -> str:
    """The run folder that actually has a ``manifest.json`` -- the current (local) base first,
    then the legacy ``<data_root>/_code_snapshots`` location, so runs snapshotted BEFORE the
    local-dir switch still replay / reconstruct. Falls back to the current-base path (for the
    not-found message). Public: used by replay (here) and ``tools/reconstruct_scan.py``."""
    primary = run_folder(data_root, run_id)
    if os.path.isfile(os.path.join(primary, "manifest.json")):
        return primary
    legacy = os.path.join(data_root, _SNAPSHOT_DIRNAME, _RUNS_DIRNAME, _safe_run_id(run_id))
    if os.path.isfile(os.path.join(legacy, "manifest.json")):
        return legacy
    return primary


def snapshot_experiment_dirs(data_root: str, run_id) -> list[str] | None:
    """Return the absolute experiment-dir paths inside a run's snapshot folder
    that exist (``…/_runs/<id>/YbSeqs`` etc.), or ``None`` if the folder/manifest
    is absent.  These are what :func:`snapshot_syspath` prepends to ``sys.path``."""
    folder = existing_run_folder(data_root, run_id)
    if not os.path.isfile(os.path.join(folder, "manifest.json")):
        return None
    dirs = [os.path.join(folder, d) for d in _EXPERIMENT_DIRS]
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs or None


def lib_mismatch(data_root: str, run_id, project_root: str | None = None) -> list[str]:
    """Best-effort: list framework/config files whose CURRENT bytes differ from
    the snapshot (``lib`` / ``expConfig`` / ``config`` -- the parts replay does
    NOT swap).  A non-empty result means "this run's framework differs from what
    runs today; full reproduction needs a restart on the snapshot's lib."  Never
    raises; returns ``[]`` when it can't tell."""
    root = project_root or pyctrl_root()
    man = os.path.join(existing_run_folder(data_root, run_id), "manifest.json")
    try:
        with open(man, encoding="utf-8") as f:
            files = json.load(f).get("files", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for rec in files:
        if rec.get("role") == "experiment":
            continue
        rel = rec.get("src_rel", "")
        cur = os.path.join(root, rel.replace("/", os.sep))
        try:
            with open(cur, "rb") as f:
                if _hash_bytes(f.read()) != rec.get("sha256"):
                    out.append(rel)
        except Exception:  # noqa: BLE001
            out.append(rel)
    return out


@contextlib.contextmanager
def snapshot_syspath(data_root: str, run_id, *, reload_modules=None, log=None):
    """Context manager that makes a run's snapshotted EXPERIMENT code importable.

    Prepends the snapshot's ``YbSeqs`` / ``YbSteps`` / ``YbScans`` /
    ``YbRearrangement`` dirs to ``sys.path``, evicts the cached experiment
    modules (via ``reload_modules`` -- normally ``seq_reload.reload_experiment_modules``)
    so the next import resolves from the snapshot, yields, then restores
    ``sys.path`` and evicts again so the NEXT job re-imports the LIVE tree.

    ``lib`` / ``expConfig`` are NOT swapped (the framework-stability boundary);
    a mismatch there is logged as a warning but replay proceeds with live lib.

    Best-effort: if the snapshot folder is missing or anything fails, it logs
    and yields WITHOUT changing ``sys.path`` -- the caller transparently falls
    back to the live tree (so a bad/absent snapshot never breaks a re-queue).
    """
    log = log or (lambda _m: None)
    dirs = None
    try:
        dirs = snapshot_experiment_dirs(data_root, run_id)
    except Exception as exc:  # noqa: BLE001
        log("code_snapshot replay: lookup failed (%s); using live code" % exc)
    if not dirs:
        log("code_snapshot replay: no snapshot for run %s; using live code" % run_id)
        yield False
        return

    try:
        mism = lib_mismatch(data_root, run_id)
        if mism:
            log("code_snapshot replay: framework/config differs from snapshot "
                "(%d file(s), e.g. %s) -- lib is NOT swapped; restart on that lib "
                "for full reproduction" % (len(mism), mism[0]))
    except Exception:  # noqa: BLE001
        pass

    global _active_replay_source
    added = list(dirs)
    saved_path = list(sys.path)
    saved_src = _active_replay_source
    try:
        sys.path[:0] = added                 # snapshot dirs win over live ones
        _active_replay_source = run_id       # scan-prep records this run as a replay
        if reload_modules is not None:
            try:
                reload_modules()             # evict live experiment modules
            except Exception:  # noqa: BLE001
                pass
        log("code_snapshot replay: importing experiment code from snapshot %s "
            "(%d dir(s))" % (run_id, len(added)))
        yield True
    finally:
        sys.path[:] = saved_path             # restore exactly
        _active_replay_source = saved_src
        if reload_modules is not None:
            try:
                reload_modules()             # next job re-imports the LIVE tree
            except Exception:  # noqa: BLE001
                pass
