#!/usr/bin/env python3
"""Prepare and verify user-supplied HTCE training/benchmark data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.evaluation.training_data import build_training_data_report
from htce_origin.kernel.serialization import canonical_json_bytes


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HTCE training/benchmark data readiness artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    report, assets = build_training_data_report(args.data_root)
    artifacts = Path(args.artifacts_dir)
    report_payload = report.as_payload()
    _write(artifacts / "training_data_readiness_report.json", report_payload)
    _write(artifacts / "training_data_manifest.json", {
        "schema_version": "htce-training-data-manifest-v1",
        "asset_count": len(assets),
        "assets": [asset.as_payload() for asset in assets],
    })
    _write(artifacts / "training_curriculum.json", {
        "schema_version": "htce-training-curriculum-v1",
        "curriculum": [stage.as_payload() for stage in report.curriculum],
    })

    print("HTCE training/data readiness")
    print(f"data_root: {args.data_root}")
    print(f"asset_count: {report.asset_count}")
    print(f"manifest_entries_checked: {report.manifest_entries_checked}")
    print(f"manifest_hash_mismatches: {report.manifest_hash_mismatches}")
    print(f"ready_for_htce_training_contour: {bool(report.ready_for_htce_training_contour)}")
    print(f"artifacts_dir: {artifacts}")
    return 0 if report.ready_for_htce_training_contour and report.manifest_hash_mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
