#!/usr/bin/env python3
"""Verify the release-manifest self hash and per-file hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.kernel.serialization import verify_release_manifest_self_hash
manifest_path = ROOT / "RELEASE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not verify_release_manifest_self_hash(manifest):
    raise SystemExit("release manifest self-hash verification failed")
for relpath, expected in sorted(manifest.get("file_hashes_sha256", {}).items()):
    if relpath in {"RELEASE_MANIFEST.json", "HASHES.txt"}:
        continue
    path = ROOT / relpath
    if not path.exists():
        raise SystemExit(f"manifest file missing: {relpath}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"manifest hash mismatch: {relpath}")
print("release_manifest: PASS")