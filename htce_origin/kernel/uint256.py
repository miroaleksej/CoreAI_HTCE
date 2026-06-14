"""Explicit 256-bit unsigned arithmetic model for HTCE-Origin P19.

Scope boundary:
- Hardware-oriented arithmetic model only.
- Models fixed-width ``logic [255:0]`` wrap semantics for the Q256 runtime.
- Does not claim FPGA/ASIC synthesis, timing, power, or board measurements.
- Uses integer-only operations and power-of-two masking; no floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from htce_origin.kernel.q16 import COS_LUT_SIZE, DEFAULT_MODULUS, q_add, q_mod, q_sin_lut, q_sub
from htce_origin.kernel.serialization import sha256_hex

UINT256_BITS = 256
UINT256_MODULUS = 1 << UINT256_BITS
UINT256_MASK = UINT256_MODULUS - 1
SIN_LUT_ADDR_BITS = 10
SIN_LUT_SIZE = 1 << SIN_LUT_ADDR_BITS
Q256_SIN_INDEX_SHIFT = UINT256_BITS - SIN_LUT_ADDR_BITS
Q256_QUARTER_TURN = UINT256_MODULUS >> 2
MAX_SMALL_MULTIPLIER_BITS = 16
MAX_SMALL_MULTIPLIER = (1 << MAX_SMALL_MULTIPLIER_BITS) - 1
HARDWARE_CLAIM_STATUS = "arithmetic_model_verified"
BOARD_MEASUREMENT_STATUS = "not_board_measured"


class UInt256ModelError(ValueError):
    """Raised when the fixed-width hardware arithmetic contract is violated."""


def _require_power_of_two_q256() -> None:
    if DEFAULT_MODULUS != UINT256_MODULUS:
        raise UInt256ModelError("DEFAULT_MODULUS must be exactly 2^256 for P19 hardware-width verification")
    if COS_LUT_SIZE != SIN_LUT_SIZE:
        raise UInt256ModelError("COS_LUT_SIZE must remain 1024 so q_sin/q_cos address extraction is shift-only")


def uint256(value: int) -> int:
    """Reduce a Python integer to an explicit unsigned 256-bit lane.

    Hardware equivalent: ``logic [255:0] out = value[255:0]``.
    """
    return int(value) & UINT256_MASK


def uint256_add(a: int, b: int) -> int:
    """Fixed-width unsigned addition with wrap modulo 2^256."""
    return (int(a) + int(b)) & UINT256_MASK


def uint256_sub(a: int, b: int) -> int:
    """Fixed-width unsigned subtraction with wrap modulo 2^256."""
    return (int(a) - int(b)) & UINT256_MASK


def uint256_mod(value: int) -> int:
    """Modulo 2^256 reduction implemented as a bit mask, not a divider."""
    return int(value) & UINT256_MASK


def uint256_mul_small(a: int, small: int) -> int:
    """Fixed-width multiply by an explicitly bounded small integer.

    P19 only claims add/sub/mask and multiply-by-small datapaths.  General
    256x256 multiplication is outside the P19 hardware claim boundary.
    """
    k = int(small)
    if not 0 <= k <= MAX_SMALL_MULTIPLIER:
        raise UInt256ModelError(f"small multiplier must be in [0, {MAX_SMALL_MULTIPLIER}]")
    return (int(a) * k) & UINT256_MASK


def uint256_cos_lut_index(phase: int) -> int:
    """Shift-only LUT address extraction for Q256 phase -> 1024-entry LUT."""
    return (uint256(phase) >> Q256_SIN_INDEX_SHIFT) & (SIN_LUT_SIZE - 1)


def uint256_sin_lut_index(phase: int) -> int:
    """Sine address extraction through exact toroidal quarter-turn identity."""
    return uint256_cos_lut_index(uint256_sub(phase, Q256_QUARTER_TURN))


def uint256_vector(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(uint256(value) for value in values)


@dataclass(frozen=True)
class UInt256OperationConstraint:
    """Machine-readable claim for one hardware-oriented primitive."""

    name: str
    bit_width: int
    semantics: str
    verilog_like: str
    allowed: bool
    forbidden_claims: tuple[str, ...] = ()
    notes: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bit_width": self.bit_width,
            "semantics": self.semantics,
            "verilog_like": self.verilog_like,
            "allowed": int(self.allowed),
            "forbidden_claims": tuple(self.forbidden_claims),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class UInt256VectorCase:
    """One deterministic comparison between Python Q256 and explicit uint256 lanes."""

    case_id: str
    a: int
    b: int
    small: int
    add_match: bool
    sub_match: bool
    mod_match: bool
    mul_small_wrap_match: bool
    sin_index_match: bool
    edge_wrap_verified: bool

    @property
    def passed(self) -> bool:
        return (
            self.add_match
            and self.sub_match
            and self.mod_match
            and self.mul_small_wrap_match
            and self.sin_index_match
            and self.edge_wrap_verified
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "a_hex": hex(uint256(self.a)),
            "b_hex": hex(uint256(self.b)),
            "small": self.small,
            "add_match": int(self.add_match),
            "sub_match": int(self.sub_match),
            "mod_match": int(self.mod_match),
            "mul_small_wrap_match": int(self.mul_small_wrap_match),
            "sin_index_match": int(self.sin_index_match),
            "edge_wrap_verified": int(self.edge_wrap_verified),
            "passed": int(self.passed),
        }


@dataclass(frozen=True)
class UInt256HardwareManifest:
    """Generated Verilog-like operation manifest for P19."""

    schema_version: str
    modulus: str
    hardware_claim_status: str
    board_measurement_status: str
    operations: tuple[UInt256OperationConstraint, ...]
    disallowed_operations: tuple[str, ...]
    notes: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "modulus": self.modulus,
            "bit_width": UINT256_BITS,
            "hardware_claim_status": self.hardware_claim_status,
            "board_measurement_status": self.board_measurement_status,
            "operations": tuple(op.as_payload() for op in self.operations),
            "disallowed_operations": self.disallowed_operations,
            "notes": self.notes,
        }
        payload["manifest_sha256"] = sha256_hex(repr(payload).encode("utf-8"))
        return payload


@dataclass(frozen=True)
class UInt256VerificationReport:
    """P19 arithmetic-model verification result."""

    sample_count: int
    passed_count: int
    failed_count: int
    cases: tuple[UInt256VectorCase, ...]
    manifest: UInt256HardwareManifest
    hardware_claim_status: str = HARDWARE_CLAIM_STATUS
    board_measurement_status: str = BOARD_MEASUREMENT_STATUS
    proof_scope: str = "fixed_width_arithmetic_model_only"

    @property
    def passed(self) -> bool:
        return self.failed_count == 0 and self.hardware_claim_status == HARDWARE_CLAIM_STATUS and self.board_measurement_status == BOARD_MEASUREMENT_STATUS

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "htce-p19-hardware-width-verification-v1",
            "sample_count": self.sample_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "passed": int(self.passed),
            "hardware_claim_status": self.hardware_claim_status,
            "board_measurement_status": self.board_measurement_status,
            "proof_scope": self.proof_scope,
            "cases": tuple(case.as_payload() for case in self.cases),
            "manifest": self.manifest.as_payload(),
        }
        payload["report_sha256"] = sha256_hex(repr(payload).encode("utf-8"))
        return payload


def generate_uint256_hardware_manifest() -> UInt256HardwareManifest:
    """Return machine-readable hardware-width operation constraints."""
    _require_power_of_two_q256()
    operations = (
        UInt256OperationConstraint(
            name="uint256_add",
            bit_width=UINT256_BITS,
            semantics="out = (a + b) mod 2^256",
            verilog_like="assign out[255:0] = a[255:0] + b[255:0];",
            allowed=True,
            notes="Carry-out beyond bit 255 is intentionally discarded as toroidal wrap.",
        ),
        UInt256OperationConstraint(
            name="uint256_sub",
            bit_width=UINT256_BITS,
            semantics="out = (a - b) mod 2^256",
            verilog_like="assign out[255:0] = a[255:0] - b[255:0];",
            allowed=True,
            notes="Borrow beyond bit 255 wraps in the unsigned residue class.",
        ),
        UInt256OperationConstraint(
            name="uint256_mod",
            bit_width=UINT256_BITS,
            semantics="out = value[255:0] for modulus 2^256",
            verilog_like="assign out[255:0] = value[255:0];",
            allowed=True,
            notes="Power-of-two modulus is implemented by truncation/masking; no divider claim.",
        ),
        UInt256OperationConstraint(
            name="uint256_mul_small",
            bit_width=UINT256_BITS,
            semantics="out = (a * k) mod 2^256, 0 <= k < 2^16",
            verilog_like="assign out[255:0] = (a[255:0] * k[15:0])[255:0];",
            allowed=True,
            notes="Only multiply-by-small is in scope. General 256x256 multiplication is disallowed.",
        ),
        UInt256OperationConstraint(
            name="q_sin_lut_address",
            bit_width=UINT256_BITS,
            semantics="idx = ((phase - 2^254) mod 2^256) >> 246 for 1024-entry sine-through-cosine LUT",
            verilog_like="assign sin_idx[9:0] = (phase[255:0] - 256'h4000000000000000000000000000000000000000000000000000000000000000) >> 246;",
            allowed=True,
            notes="Sine uses exact toroidal quarter-turn identity and shift-only LUT addressing.",
        ),
    )
    return UInt256HardwareManifest(
        schema_version="htce-p19-uint256-hardware-manifest-v1",
        modulus="2^256",
        hardware_claim_status=HARDWARE_CLAIM_STATUS,
        board_measurement_status=BOARD_MEASUREMENT_STATUS,
        operations=operations,
        disallowed_operations=(
            "general_256x256_multiplier_claim",
            "floating_point_datapath_claim",
            "divider_based_mod_2_256_claim",
            "measured_fpga_timing_claim",
            "measured_fpga_power_claim",
            "board_measured_claim",
        ),
        notes=(
            "This manifest verifies arithmetic transfer constraints only.",
            "It does not claim synthesis, place-and-route, board timing, power, or energy measurements.",
            "Python arbitrary-precision integers are used only as an oracle compared against explicit 256-bit masking semantics.",
        ),
    )


def _lcg_next(state: int) -> int:
    return uint256((state * 6364136223846793005) + 1442695040888963407)


def _case(case_id: str, a: int, b: int, small: int) -> UInt256VectorCase:
    add_expected = q_add(a, b, UINT256_MODULUS)
    sub_expected = q_sub(a, b, UINT256_MODULUS)
    mod_expected = q_mod(a + (UINT256_MODULUS << 1), UINT256_MODULUS)
    mul_expected = q_mod(a * small, UINT256_MODULUS)
    sin_idx_expected = ((q_sub(a, UINT256_MODULUS // 4, UINT256_MODULUS) * COS_LUT_SIZE) // UINT256_MODULUS) % COS_LUT_SIZE
    edge_ok = (
        uint256_add(UINT256_MASK, 1) == 0
        and uint256_sub(0, 1) == UINT256_MASK
        and uint256_mul_small(UINT256_MASK, 2) == UINT256_MASK - 1
    )
    return UInt256VectorCase(
        case_id=case_id,
        a=a,
        b=b,
        small=small,
        add_match=uint256_add(a, b) == add_expected,
        sub_match=uint256_sub(a, b) == sub_expected,
        mod_match=uint256_mod(a + (UINT256_MODULUS << 1)) == mod_expected,
        mul_small_wrap_match=uint256_mul_small(a, small) == mul_expected,
        sin_index_match=uint256_sin_lut_index(a) == sin_idx_expected and isinstance(q_sin_lut(a, UINT256_MODULUS), int),
        edge_wrap_verified=edge_ok,
    )


def verify_uint256_arithmetic_model(sample_count: int = 256, seed: int = 19) -> UInt256VerificationReport:
    """Verify Q256 runtime arithmetic against explicit fixed-width lanes."""
    _require_power_of_two_q256()
    if sample_count <= 0:
        raise UInt256ModelError("sample_count must be positive")
    state = uint256(seed)
    cases: list[UInt256VectorCase] = []
    edge_inputs = (
        (0, 0, 0),
        (0, 1, 1),
        (UINT256_MASK, 1, 2),
        (UINT256_MASK, UINT256_MASK, MAX_SMALL_MULTIPLIER),
        (Q256_QUARTER_TURN, UINT256_MODULUS >> 1, 3),
    )
    for idx, (a, b, small) in enumerate(edge_inputs):
        cases.append(_case(f"edge_{idx}", a, b, small))
    for idx in range(int(sample_count)):
        state = _lcg_next(state + idx + 1)
        a = state
        state = _lcg_next(state ^ (idx + 17))
        b = state
        small = (state >> (UINT256_BITS - MAX_SMALL_MULTIPLIER_BITS)) & MAX_SMALL_MULTIPLIER
        cases.append(_case(f"sample_{idx}", a, b, small))
    passed_count = sum(1 for case in cases if case.passed)
    failed_count = len(cases) - passed_count
    return UInt256VerificationReport(
        sample_count=len(cases),
        passed_count=passed_count,
        failed_count=failed_count,
        cases=tuple(cases),
        manifest=generate_uint256_hardware_manifest(),
    )


def hardware_claim_payload() -> Mapping[str, object]:
    """Buyer-safe machine-readable P19 hardware claim boundary."""
    return {
        "hardware_claim_status": HARDWARE_CLAIM_STATUS,
        "board_measurement_status": BOARD_MEASUREMENT_STATUS,
        "allowed_claim": "hardware-oriented integer arithmetic model",
        "prohibited_claim": "measured FPGA/ASIC implementation",
        "modulus": "2^256",
        "bit_width": UINT256_BITS,
    }
