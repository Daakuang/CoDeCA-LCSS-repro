"""Create reproducibility metadata for the LCSS V2 revision."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence


PUBLICATION_SEED = 23
UNAVAILABLE = "unavailable"
PACKAGE_DISTRIBUTIONS = ("numpy", "scipy", "gurobipy", "pytest", "matplotlib")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = UNAVAILABLE
    return versions


def _gurobi_version() -> str:
    try:
        import gurobipy as gp
    except ImportError:
        return UNAVAILABLE

    try:
        return ".".join(str(part) for part in gp.gurobi.version())
    except AttributeError:
        return UNAVAILABLE


def _git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return UNAVAILABLE
    if completed.returncode != 0:
        return UNAVAILABLE
    return completed.stdout.strip() or UNAVAILABLE


def build_manifest(repository_root: Path) -> dict[str, Any]:
    """Return deterministic provenance fields for the current repository."""
    mother_relative = Path("LcssV1.1") / "CoDeCA_LCSS.tex"
    return {
        "schema_version": 1,
        "mother_tex": {
            "path": mother_relative.as_posix(),
            "sha256": sha256_file(repository_root / mother_relative),
        },
        "python": {
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "gurobi": {"version": _gurobi_version()},
        "publication_seed": PUBLICATION_SEED,
        "git_commit": _git_commit(repository_root),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the LCSS V2 reproducibility manifest to an explicit path."""
    arguments = _parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    output_path: Path = arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_manifest(repository_root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
