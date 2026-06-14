#!/usr/bin/env python3
"""P23 release-integrity gate: synchronize __version__, pyproject, capabilities and manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import htce_origin


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    py_version = str(pyproject["project"]["version"])
    init_version = str(htce_origin.__version__)
    caps = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
    caps_version = str(caps.get("version", ""))
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    manifest_version = py_version
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_version = str(manifest.get("version", ""))
    versions = {
        "pyproject.toml": py_version,
        "htce_origin.__version__": init_version,
        "capabilities.json": caps_version,
        "RELEASE_MANIFEST.json": manifest_version,
    }
    mismatches = {name: value for name, value in versions.items() if value != py_version}
    if mismatches:
        print("version_sync: FAIL")
        for name, value in versions.items():
            print(f"{name}: {value}")
        raise SystemExit(1)
    print(f"version_sync: PASS {py_version}")


if __name__ == "__main__":
    main()
