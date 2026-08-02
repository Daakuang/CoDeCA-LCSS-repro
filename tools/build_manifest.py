"""Build the deterministic SHA-256 manifest for the public release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final


RELEASE_VERSION: Final[str] = "1.0.0"
MANIFEST_NAME: Final[str] = "PUBLIC_RELEASE_MANIFEST.json"
EXCLUDED_PARTS: Final[frozenset[str]] = frozenset(
    {".git", ".pytest_cache", "__pycache__"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != MANIFEST_NAME
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        and path.suffix != ".pyc"
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": 1,
        "release_version": RELEASE_VERSION,
        "release_date": "2026-08-02",
        "file_count": len(entries),
        "files": entries,
    }
    target = root / MANIFEST_NAME
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
