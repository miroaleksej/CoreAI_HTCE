#!/usr/bin/env python3
"""Regenerate RELEASE_MANIFEST.json and HASHES.txt from current repository files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.kernel.serialization import canonical_json_bytes, release_manifest_self_hash

MANIFEST_PATH = ROOT / "RELEASE_MANIFEST.json"
HASHES_PATH = ROOT / "HASHES.txt"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    file_hashes = manifest.setdefault("file_hashes_sha256", {})

    for relpath in sorted(list(file_hashes.keys())):
        if relpath in {"RELEASE_MANIFEST.json", "HASHES.txt"}:
            continue

        path = ROOT / relpath
        if not path.exists():
            raise SystemExit(f"manifest file missing: {relpath}")

        file_hashes[relpath] = sha256_file(path)

    file_hashes.pop("RELEASE_MANIFEST.json", None)
    file_hashes["RELEASE_MANIFEST.json"] = release_manifest_self_hash(manifest)

    MANIFEST_PATH.write_bytes(canonical_json_bytes(manifest))

    HASHES_PATH.write_text(
        "".join(f"{digest}  {relpath}\n" for relpath, digest in sorted(file_hashes.items())),
        encoding="utf-8",
    )

    print("regenerate_manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
