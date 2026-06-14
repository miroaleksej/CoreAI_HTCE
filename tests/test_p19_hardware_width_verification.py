import json
import subprocess
import sys
from pathlib import Path

import pytest

from htce_origin.kernel.q16 import DEFAULT_MODULUS, Q256_MODULUS, q_add, q_mod, q_sub
from htce_origin.kernel.uint256 import (
    BOARD_MEASUREMENT_STATUS,
    HARDWARE_CLAIM_STATUS,
    MAX_SMALL_MULTIPLIER,
    UINT256_MASK,
    UINT256_MODULUS,
    UInt256ModelError,
    generate_uint256_hardware_manifest,
    uint256_add,
    uint256_mod,
    uint256_mul_small,
    uint256_sin_lut_index,
    uint256_sub,
    verify_uint256_arithmetic_model,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p19_uint256_constants_match_runtime_q256():
    assert DEFAULT_MODULUS == Q256_MODULUS == UINT256_MODULUS == 1 << 256
    assert UINT256_MASK == UINT256_MODULUS - 1


def test_p19_wraparound_edges_match_q256_runtime():
    assert uint256_add(UINT256_MASK, 1) == 0
    assert uint256_sub(0, 1) == UINT256_MASK
    assert uint256_add(UINT256_MASK, UINT256_MASK) == q_add(UINT256_MASK, UINT256_MASK, UINT256_MODULUS)
    assert uint256_sub(7, 19) == q_sub(7, 19, UINT256_MODULUS)
    assert uint256_mod(UINT256_MODULUS * 3 + 123) == q_mod(UINT256_MODULUS * 3 + 123, UINT256_MODULUS)


def test_p19_mul_by_small_is_bounded_and_wraps():
    assert uint256_mul_small(UINT256_MASK, 2) == UINT256_MASK - 1
    assert uint256_mul_small(123, MAX_SMALL_MULTIPLIER) == q_mod(123 * MAX_SMALL_MULTIPLIER, UINT256_MODULUS)
    with pytest.raises(UInt256ModelError):
        uint256_mul_small(1, MAX_SMALL_MULTIPLIER + 1)


def test_p19_q_sin_lut_address_is_shift_only_range():
    assert 0 <= uint256_sin_lut_index(0) < 1024
    assert 0 <= uint256_sin_lut_index(UINT256_MASK) < 1024
    assert 0 <= uint256_sin_lut_index(UINT256_MODULUS // 4) < 1024


def test_p19_generated_manifest_claim_boundary():
    manifest = generate_uint256_hardware_manifest().as_payload()
    assert manifest["hardware_claim_status"] == HARDWARE_CLAIM_STATUS
    assert manifest["board_measurement_status"] == BOARD_MEASUREMENT_STATUS
    assert "board_measured_claim" in manifest["disallowed_operations"]
    assert any(op["name"] == "q_sin_lut_address" for op in manifest["operations"])
    assert all(op["bit_width"] == 256 for op in manifest["operations"])


def test_p19_verification_report_passes_without_board_claim():
    report = verify_uint256_arithmetic_model(sample_count=32)
    assert report.passed is True
    payload = report.as_payload()
    assert payload["hardware_claim_status"] == "arithmetic_model_verified"
    assert payload["board_measurement_status"] == "not_board_measured"
    assert payload["failed_count"] == 0


def test_p19_script_writes_hardware_artifacts():
    completed = subprocess.run(
        [sys.executable, "scripts/03_topology_and_hardware/run_hardware_width.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report_path = ROOT / "artifacts" / "hardware_width_verification_report.json"
    manifest_path = ROOT / "artifacts" / "hardware_operation_manifest.json"
    claim_path = ROOT / "artifacts" / "hardware_claim_boundary.json"
    assert report_path.exists()
    assert manifest_path.exists()
    assert claim_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] == 1
    assert report["hardware_claim_status"] == "arithmetic_model_verified"
    assert report["board_measurement_status"] == "not_board_measured"
