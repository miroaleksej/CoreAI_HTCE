#!/usr/bin/env python3
"""Run P19 Q256 hardware-width arithmetic verification and export artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

from htce_origin.kernel.serialization import canonical_json_bytes
from htce_origin.kernel.uint256 import hardware_claim_payload, verify_uint256_arithmetic_model


def _write(name: str, payload: dict[str, object]) -> None:
    (ARTIFACTS / name).write_bytes(canonical_json_bytes(payload))


def main() -> None:
    report = verify_uint256_arithmetic_model(sample_count=32)
    payload = report.as_payload()
    _write("hardware_width_verification_report.json", payload)
    _write("hardware_operation_manifest.json", payload["manifest"])
    _write("hardware_claim_boundary.json", dict(hardware_claim_payload()))
    if not report.passed:
        raise SystemExit("P19 hardware-width verification failed")
    print("hardware_width_verification: PASS")
    print(f"sample_count: {report.sample_count}")
    print(f"hardware_claim_status: {report.hardware_claim_status}")
    print(f"board_measurement_status: {report.board_measurement_status}")


if __name__ == "__main__":
    main()
