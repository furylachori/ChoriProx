"""Safe I/O helpers for skill prompt files and JSON definitions.

This module owns two public functions and one exception:

* :class:`SkillPathError` — raised on any filesystem failure (subclass of
  ``OSError`` for catchability, distinct from skill-registration errors).
* :func:`get_skills_dir` — resolves the configured skills directory from the
  ``PROXY_SKILLS_DIR`` env var.
* :func:`read_contained_regular_file` — opens a regular file inside the
  skills directory via ``O_RDONLY | O_CLOEXEC | O_NOFOLLOW``, ``fstat``s
  the descriptor (regular file + size check), reads at most ``max_bytes +
  1``, and returns strict UTF-8 decoded text. Rejects ``..`` components,
  absolute paths, empty paths, symlinks, directories, oversize content,
  invalid UTF-8, and UTF-8 BOM.

The safe-read helper replaces a prior ``resolve_skill_path`` + reopen pattern
that resolved the path with ``os.path.realpath`` and re-opened by name,
leaving a TOCTOU window between validation and read. The new helper opens
the file first and validates only via the open descriptor.

Platform notes (macOS/Linux):

* ``O_NOFOLLOW`` is required and is available on both. On macOS
  ``O_NOFOLLOW`` is implemented at the kernel level and supported by the
  BSD-derived syscall.
* ``O_DIRECTORY`` is Linux-only; on macOS we omit it. Path.parts provides
  textual containment; intermediate directory entries are checked with
  ``os.scandir(..., follow_symlinks=False)`` which is portable and has the
  same effect: a symlinked intermediate component is reported (and we
  reject it) rather than traversed.
* If a future Windows port is needed, fail closed by raising
  :class:`SkillPathError` from :func:`read_contained_regular_file` when
  ``O_NOFOLLOW`` is unavailable. ``NotImplementedError`` is reserved for
  platform cases that simply are not implemented.
"""
import os
import stat
from pathlib import Path
from typing import Optional


class SkillPathError(OSError):
    """Raised on any filesystem failure inside :mod:`proxy.skills.path_utils`.

    Subclasses ``OSError`` so generic ``except OSError`` blocks continue to
    work. Distinct from
    :class:`proxy.skills.skill_definition.SkillRegistrationError`
    (a ``PermissionError`` subclass) so callers can catch them separately.
    """


_REPO_ROOT = Path(__file__).resolve().parents[3]


_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def get_skills_dir() -> Optional[Path]:
    """Return the configured skills directory, or ``None`` if explicitly disabled.

    Order of precedence, based on ``os.environ`` membership (NOT truthiness):

    | Environment state | Return value |
    |---|---|
    | ``PROXY_SKILLS_DIR`` absent | ``<repo-root>/skills`` (the canonical default) |
    | ``PROXY_SKILLS_DIR=""`` (empty string) | ``None`` (custom loading explicitly disabled) |
    | ``PROXY_SKILLS_DIR="/abs/path"`` | ``Path("/abs/path")`` verbatim |
    | ``PROXY_SKILLS_DIR="rel/path"`` | ``Path("<repo-root>/rel/path")`` |

    Downstream callers treat a missing directory on disk as a no-op. Only
    the explicit-empty case signals "do not attempt to load."
    """
    env_val = os.environ.get("PROXY_SKILLS_DIR")
    if env_val is not None:
        if env_val == "":
            return None
        p = Path(env_val)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        return p
    return _REPO_ROOT / "skills"


def read_contained_regular_file(
    root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> str:
    """Safely read a regular file beneath ``root``.

    The path is opened only by descriptor; no validation-then-reopen window
    exists. Intermediate directory entries that are symlinks are rejected
    rather than traversed; the leaf component is opened with ``O_NOFOLLOW``
    so a leaf symlink raises ``OSError`` (translated to
    :class:`SkillPathError`).

    Args:
        root: Directory that contains the file. May be a symlink — the
            resolved target is the authoritative containment root.
            Relative ``root`` is resolved against the repo root.
        relative_path: Path string relative to ``root``. MUST be a
            non-empty string with no absolute or ``..`` components.
        max_bytes: Hard size limit in bytes. The leaf's ``st_size`` is
            checked against this limit BEFORE any read, and ``os.read``
            is bounded by ``max_bytes + 1`` so a file that grew between
            ``fstat`` and ``read`` is rejected with the same error.

    Returns:
        Decoded UTF-8 text. Empty string is valid output (a 0-byte file).

    Raises:
        SkillPathError: For absolute paths, empty paths, ``..`` components,
            symlinks (intermediate or leaf), missing parents, non-regular
            files (directories, devices, sockets, etc.), oversize files,
            files that exceed ``max_bytes`` during read, UTF-8 BOM, and
            non-UTF-8 byte sequences.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise SkillPathError(
            f"relative_path must be a non-empty string, got {relative_path!r}"
        )
    if os.path.isabs(relative_path):
        raise SkillPathError(
            f"relative_path must be relative to the skills directory, "
            f"got absolute path '{relative_path}'"
        )
    parts = Path(relative_path).parts
    if any(p == ".." for p in parts):
        raise SkillPathError(
            f"relative_path must not contain '..' components, "
            f"got '{relative_path}'"
        )
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        raise SkillPathError(
            f"max_bytes must be an int, got {type(max_bytes).__name__}"
        )
    if max_bytes < 0:
        raise SkillPathError(f"max_bytes must be non-negative, got {max_bytes}")

    root_path = Path(root) if not isinstance(root, Path) else root
    if not root_path.is_absolute():
        root_path = _REPO_ROOT / root_path
    root_real = os.path.realpath(str(root_path))

    if not os.path.isdir(root_real):
        raise SkillPathError(
            f"root '{root}' resolves to '{root_real}' which is not a directory"
        )

    current_dir = Path(root_real)

    for part in parts[:-1]:
        try:
            entries = list(os.scandir(current_dir))
        except OSError as e:
            raise SkillPathError(
                f"cannot list directory '{current_dir}' while resolving "
                f"'{relative_path}': {e}"
            )
        found = next((entry for entry in entries if entry.name == part), None)
        if found is None:
            raise SkillPathError(
                f"path component '{part}' not found in '{current_dir}' "
                f"(relative_path='{relative_path}')"
            )
        if found.is_symlink():
            raise SkillPathError(
                f"path component '{part}' is a symlink in '{current_dir}'; "
                f"refusing to traverse (relative_path='{relative_path}')"
            )
        if not found.is_dir(follow_symlinks=False):
            raise SkillPathError(
                f"path component '{part}' in '{current_dir}' is not a "
                f"directory (relative_path='{relative_path}')"
            )
        current_dir = Path(found.path)

    leaf_path = str(current_dir / parts[-1])
    flags = os.O_RDONLY | _CLOEXEC | _NO_FOLLOW
    try:
        leaf_fd = os.open(leaf_path, flags)
    except OSError as e:
        raise SkillPathError(
            f"cannot open '{relative_path}' (resolved to '{leaf_path}'): "
            f"{e.strerror if e.strerror else e}"
        )

    try:
        st = os.fstat(leaf_fd)
        if not stat.S_ISREG(st.st_mode):
            raise SkillPathError(
                f"path '{relative_path}' (resolved to '{leaf_path}') is "
                f"not a regular file (st_mode=0o{st.st_mode & 0o170000:o})"
            )

        if st.st_size > max_bytes:
            raise SkillPathError(
                f"file '{relative_path}' is {st.st_size} bytes, exceeds "
                f"limit of {max_bytes} bytes"
            )

        data = os.read(leaf_fd, max_bytes + 1)
        if len(data) > max_bytes:
            raise SkillPathError(
                f"file '{relative_path}' read returned {len(data)} bytes, "
                f"exceeds limit of {max_bytes} bytes (file may have been "
                f"modified during read)"
            )

        if data.startswith(b"\xef\xbb\xbf"):
            raise SkillPathError(
                f"file '{relative_path}' starts with UTF-8 BOM (EF BB BF); "
                f"BOM is not allowed"
            )

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SkillPathError(
                f"file '{relative_path}' is not valid strict UTF-8: {e}"
            )
    finally:
        os.close(leaf_fd)
