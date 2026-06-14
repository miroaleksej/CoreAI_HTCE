"""Integer-only torus primitives for HTCE-Origin.

Default protected runtime scope:
- N = 2**256 modulus for the large-range Q256 torus profile.
- Scalar and vector phase arithmetic over ``(Z / N Z)^d``.
- Minimal LUT-backed periodic loss with no runtime floating-point path.
- Deterministic toroidal distance primitives used by L1/L2/L3 logic.

The module name is kept for source compatibility with earlier Q16 releases, but
``DEFAULT_MODULUS`` now points to the Q256 profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

Q16_MODULUS = 2**16
Q256_MODULUS = 2**256
N = Q256_MODULUS
DEFAULT_MODULUS = N
COS_LUT_SIZE = 1024
COS_SCALE = 32767
LOSS_SCALE = 65535


class Q16Error(ValueError):
    """Raised when a Q16 operation receives invalid dimensions or modulus."""


@dataclass(frozen=True)
class Q16PropertyStressReport:
    """Deterministic property-stress report for Q16 torus arithmetic.

    The checker is intentionally dependency-free: it exercises a reproducible
    pseudo-random stream and records only integer counters. It is not a formal
    proof, but it is a release-gate stress contract for the algebraic invariants
    that HTCE relies on.
    """

    sample_count: int
    distance_symmetry_failures: int
    add_associativity_failures: int
    distance_identity_failures: int
    wrap_neighbor_failures: int
    modulus: int = DEFAULT_MODULUS

    @property
    def total_failures(self) -> int:
        return (
            self.distance_symmetry_failures
            + self.add_associativity_failures
            + self.distance_identity_failures
            + self.wrap_neighbor_failures
        )

    @property
    def passed(self) -> bool:
        return self.total_failures == 0


def q_mod(value: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Return value reduced into the toroidal phase interval [0, modulus)."""
    if modulus <= 0:
        raise Q16Error("modulus must be positive")
    return int(value) % int(modulus)


def q_add(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Toroidal addition."""
    return q_mod(int(a) + int(b), modulus)


def q_sub(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Toroidal subtraction."""
    return q_mod(int(a) - int(b), modulus)


def q_mul(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Toroidal integer multiplication."""
    return q_mod(int(a) * int(b), modulus)


def q_delta(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Shortest circular phase distance between two scalar phases."""
    left = q_sub(a, b, modulus)
    right = q_sub(b, a, modulus)
    return left if left <= right else right


def q_vector(values: Iterable[int], modulus: int = DEFAULT_MODULUS) -> tuple[int, ...]:
    """Normalize an iterable of integer phases into an immutable torus vector."""
    return tuple(q_mod(v, modulus) for v in values)


def _same_dim(a: tuple[int, ...], b: tuple[int, ...]) -> None:
    if len(a) != len(b):
        raise Q16Error("vectors must have the same dimension")


def q_vector_add(a: Iterable[int], b: Iterable[int], modulus: int = DEFAULT_MODULUS) -> tuple[int, ...]:
    av = q_vector(a, modulus)
    bv = q_vector(b, modulus)
    _same_dim(av, bv)
    return tuple(q_add(x, y, modulus) for x, y in zip(av, bv))


def q_vector_sub(a: Iterable[int], b: Iterable[int], modulus: int = DEFAULT_MODULUS) -> tuple[int, ...]:
    av = q_vector(a, modulus)
    bv = q_vector(b, modulus)
    _same_dim(av, bv)
    return tuple(q_sub(x, y, modulus) for x, y in zip(av, bv))


def q_distance(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Alias for shortest scalar toroidal distance."""
    return q_delta(a, b, modulus)


def q_distance_vector(a: Iterable[int], b: Iterable[int], modulus: int = DEFAULT_MODULUS) -> int:
    """Squared integer toroidal distance across a vector."""
    av = q_vector(a, modulus)
    bv = q_vector(b, modulus)
    _same_dim(av, bv)
    return sum(q_delta(x, y, modulus) ** 2 for x, y in zip(av, bv))


def _quad_cos(pos: int, quarter: int) -> int:
    # Parabolic integer approximation of cosine on one quadrant: 1 -> 0.
    pos = max(0, min(int(pos), quarter))
    numerator = COS_SCALE * (quarter * quarter - pos * pos)
    return numerator // (quarter * quarter)


def _quad_sin(pos: int, quarter: int) -> int:
    # Parabolic integer approximation of sine on one quadrant: 0 -> 1.
    pos = max(0, min(int(pos), quarter))
    numerator = COS_SCALE * (2 * quarter * pos - pos * pos)
    return numerator // (quarter * quarter)


def _build_cos_lut() -> tuple[int, ...]:
    quarter = COS_LUT_SIZE // 4
    values: list[int] = []
    for idx in range(COS_LUT_SIZE):
        quadrant = idx // quarter
        pos = idx % quarter
        if quadrant == 0:
            values.append(_quad_cos(pos, quarter))
        elif quadrant == 1:
            values.append(-_quad_sin(pos, quarter))
        elif quadrant == 2:
            values.append(-_quad_cos(pos, quarter))
        else:
            values.append(_quad_sin(pos, quarter))
    return tuple(values)


COS_LUT = _build_cos_lut()


def q_cos_lut(phase_delta: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Return fixed-point cosine approximation from the integer LUT.

    Output range is approximately [-COS_SCALE, COS_SCALE].
    """
    phase = q_mod(phase_delta, modulus)
    if modulus > 0 and (modulus & (modulus - 1)) == 0 and (COS_LUT_SIZE & (COS_LUT_SIZE - 1)) == 0:
        # Exact hardware-width power-of-two path.  For Q256 and a 1024-entry LUT:
        # floor(phase * 2^10 / 2^256) == phase >> 246.
        shift = (int(modulus).bit_length() - 1) - (COS_LUT_SIZE.bit_length() - 1)
        idx = phase >> max(0, shift)
    else:
        idx = (phase * COS_LUT_SIZE) // modulus
    return COS_LUT[idx % COS_LUT_SIZE]


def q_sin_lut(phase_delta: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Return fixed-point sine approximation from the integer LUT.

    The sine table is derived from the existing cosine table through the exact
    toroidal quarter-turn identity ``sin(x) = cos(x - N/4)``.  This keeps the
    runtime path integer-only and works for both the legacy Q16 profile and the
    default Q256 profile.
    """
    return q_cos_lut(q_sub(phase_delta, modulus // 4, modulus), modulus)


def q_toroidal_loss_lut(phase_delta: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Return integer-scaled periodic loss for a scalar phase delta."""
    cos_value = q_cos_lut(phase_delta, modulus)
    return ((COS_SCALE - cos_value) * LOSS_SCALE) // (2 * COS_SCALE)


def q_toroidal_loss_vector(a: Iterable[int], b: Iterable[int], modulus: int = DEFAULT_MODULUS) -> int:
    """Sum LUT-backed periodic loss across a vector."""
    av = q_vector(a, modulus)
    bv = q_vector(b, modulus)
    _same_dim(av, bv)
    return sum(q_toroidal_loss_lut(q_sub(x, y, modulus), modulus) for x, y in zip(av, bv))


@dataclass(frozen=True)
class Q16Value:
    """Single Q16 phase value."""

    value: int
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", q_mod(self.value, self.modulus))

    def add(self, other: int) -> "Q16Value":
        return Q16Value(q_add(self.value, other, self.modulus), self.modulus)

    def sub(self, other: int) -> "Q16Value":
        return Q16Value(q_sub(self.value, other, self.modulus), self.modulus)

    def distance(self, other: int) -> int:
        return q_distance(self.value, other, self.modulus)


@dataclass(frozen=True)
class Q16Vector:
    """Immutable vector in (Z/NZ)^d."""

    values: tuple[int, ...]
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", q_vector(self.values, self.modulus))

    @classmethod
    def zero(cls, dimension: int, modulus: int = DEFAULT_MODULUS) -> "Q16Vector":
        if dimension <= 0:
            raise Q16Error("dimension must be positive")
        return cls(tuple(0 for _ in range(dimension)), modulus)

    @property
    def dimension(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    def add(self, other: Iterable[int]) -> "Q16Vector":
        return Q16Vector(q_vector_add(self.values, other, self.modulus), self.modulus)

    def sub(self, other: Iterable[int]) -> "Q16Vector":
        return Q16Vector(q_vector_sub(self.values, other, self.modulus), self.modulus)

    def distance_sq(self, other: Iterable[int]) -> int:
        return q_distance_vector(self.values, other, self.modulus)

    def loss(self, other: Iterable[int]) -> int:
        return q_toroidal_loss_vector(self.values, other, self.modulus)


def _lcg_next(state: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Full-range deterministic integer generator for release-gate stress tests.

    For power-of-two moduli, the multiplier is congruent to one modulo four and
    the increment is odd, preserving the standard full-period LCG condition over
    ``Z / 2**k Z``.  Unlike the earlier Q16 stress generator, it does not clamp
    the stream to a 31-bit auxiliary modulus.
    """
    return q_mod(1103515245 * int(state) + 12345, modulus)


def q16_property_stress(
    *,
    sample_count: int = 100000,
    seed: int = 1,
    modulus: int = DEFAULT_MODULUS,
) -> Q16PropertyStressReport:
    """Check core torus-algebra invariants over deterministic samples.

    Checked properties:
    - d_N(a,b) = d_N(b,a)
    - (a+b)+c = a+(b+c) over Z/NZ
    - d_N(a,a) = 0
    - d_N(a,a+N-1) = 1

    The function uses no floats and no external randomness.
    """
    if sample_count <= 0:
        raise Q16Error("sample_count must be positive")
    if modulus <= 1:
        raise Q16Error("modulus must be greater than one")

    state = q_mod(seed, modulus)
    distance_symmetry_failures = 0
    add_associativity_failures = 0
    distance_identity_failures = 0
    wrap_neighbor_failures = 0

    for _ in range(int(sample_count)):
        state = _lcg_next(state, modulus)
        a = state
        state = _lcg_next(state + 17, modulus)
        b = state
        state = _lcg_next(state + 31, modulus)
        c = state

        if q_distance(a, b, modulus) != q_distance(b, a, modulus):
            distance_symmetry_failures += 1
        if q_add(q_add(a, b, modulus), c, modulus) != q_add(a, q_add(b, c, modulus), modulus):
            add_associativity_failures += 1
        if q_distance(a, a, modulus) != 0:
            distance_identity_failures += 1
        if q_distance(a, q_add(a, modulus - 1, modulus), modulus) != 1:
            wrap_neighbor_failures += 1

    return Q16PropertyStressReport(
        sample_count=int(sample_count),
        distance_symmetry_failures=distance_symmetry_failures,
        add_associativity_failures=add_associativity_failures,
        distance_identity_failures=distance_identity_failures,
        wrap_neighbor_failures=wrap_neighbor_failures,
        modulus=modulus,
    )
