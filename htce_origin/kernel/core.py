"""Core typed frames and deterministic phase projections for HTCE-Origin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from htce_origin.kernel.q16 import DEFAULT_MODULUS, Q16Error, Q16Vector, q_add, q_mod, q_vector_add

DEFAULT_TORUS_DIMENSION = 64
PERSON_SIZE = 8


class CoreError(ValueError):
    """Raised when core frame or projection inputs are invalid."""


class FrameKind(str, Enum):
    FACT = "fact"
    QUERY = "query"
    ACTION_SIM = "action_sim"
    CLAIM = "claim"
    EVIDENCE = "evidence"


def _canonical_text(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        raise CoreError("identifier must be non-empty")
    return text


@dataclass(frozen=True)
class EntityId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _canonical_text(self.value))


@dataclass(frozen=True)
class RelationId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _canonical_text(self.value))


@dataclass(frozen=True)
class EvidenceId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _canonical_text(self.value))


@dataclass(frozen=True)
class TorusVector:
    """Core alias for active toroidal state vectors."""

    phases: tuple[int, ...]
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(q_mod(v, self.modulus) for v in self.phases))

    @classmethod
    def zero(cls, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS) -> "TorusVector":
        return cls(tuple(0 for _ in range(dimension)), modulus)

    @property
    def dimension(self) -> int:
        return len(self.phases)

    def add(self, delta: Iterable[int]) -> "TorusVector":
        return TorusVector(q_vector_add(self.phases, tuple(delta), self.modulus), self.modulus)

    def to_q16(self) -> Q16Vector:
        return Q16Vector(self.phases, self.modulus)


@dataclass(frozen=True)
class FactFrame:
    subject: EntityId
    relation: RelationId
    object: EntityId
    evidence: EvidenceId
    confidence_bp: int = 10000
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= int(self.confidence_bp) <= 10000:
            raise CoreError("confidence_bp must be in [0, 10000]")


@dataclass(frozen=True)
class QueryFrame:
    subject: EntityId
    relation: RelationId
    evidence_required: bool = True


@dataclass(frozen=True)
class ActionFrame:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    simulated: bool = True


@dataclass(frozen=True)
class ClaimFrame:
    text: str
    evidence: EvidenceId | None = None
    claim_type: str = "bounded"


@dataclass(frozen=True)
class FactDelta:
    fact: FactFrame
    delta: tuple[int, ...]
    weight: int = 1
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", tuple(q_mod(v, self.modulus) for v in self.delta))
        if self.weight <= 0:
            raise CoreError("fact delta weight must be positive")


@dataclass(frozen=True)
class CollisionCalibrationReport:
    """Deterministic collision/capacity report for phase projections.

    Collisions are reported as diagnostics only. A collision never authorizes a
    fact, proof, evidence relation, or policy decision.
    """

    sample_kind: str
    sample_count: int
    dimension: int
    exact_collision_count: int
    exact_collision_rate_bp: int
    coordinate_collision_count: int
    coordinate_total: int
    coordinate_collision_rate_bp: int
    same_input_stable: bool
    collision_is_proof: bool = False

    @property
    def passed_boundary(self) -> bool:
        return self.same_input_stable and not self.collision_is_proof


def _digest_bytes(label: str, value: str, person: str) -> bytes:
    person_bytes = person.encode("utf-8")[:PERSON_SIZE].ljust(PERSON_SIZE, b"_")
    h = hashlib.blake2b(digest_size=32, person=person_bytes)
    h.update(label.encode("utf-8"))
    h.update(b"\0")
    h.update(value.encode("utf-8"))
    return h.digest()


def _phase_from_digest(data: bytes, offset: int, modulus: int) -> int:
    chunk = data[offset : offset + 4]
    if len(chunk) < 4:
        chunk = (chunk + data)[:4]
    return int.from_bytes(chunk, "big") % modulus


def hash_to_phase(
    identifier: str,
    *,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
    namespace: str = "entity",
) -> tuple[int, ...]:
    """Deterministically project an identifier into a torus vector.

    For small moduli this keeps the previous chunked projection path.  For the
    Q256 profile each coordinate consumes a full 256-bit digest-derived phase so
    the active torus is not accidentally restricted to a legacy 32-bit range.
    """
    if dimension <= 0:
        raise CoreError("dimension must be positive")
    if modulus <= 0:
        raise CoreError("modulus must be positive")
    text = _canonical_text(identifier)
    phases: list[int] = []
    if int(modulus).bit_length() <= 32:
        block = 0
        while len(phases) < dimension:
            digest = _digest_bytes(namespace, f"{text}:{block}", "HTCEv01")
            for offset in range(0, len(digest), 4):
                phases.append(_phase_from_digest(digest, offset, modulus))
                if len(phases) == dimension:
                    break
            block += 1
        return tuple(phases)

    shake = hashlib.shake_256()
    shake.update(b"HTCEv01\0")
    shake.update(str(namespace).encode("utf-8"))
    shake.update(b"\0")
    shake.update(text.encode("utf-8"))
    raw = shake.digest(32 * int(dimension))
    for coordinate in range(int(dimension)):
        start = coordinate * 32
        phases.append(int.from_bytes(raw[start : start + 32], "big") % int(modulus))
    return tuple(phases)


def integer_matrix_projection(
    bits: Iterable[int],
    matrix: Iterable[Iterable[int]],
    *,
    bias: Iterable[int] | None = None,
    modulus: int = DEFAULT_MODULUS,
) -> tuple[int, ...]:
    """Integer-safe projection with entries expected from {-1, 0, +1}."""
    bit_values = tuple(1 if int(v) else 0 for v in bits)
    bias_values = tuple(bias) if bias is not None else None
    projected: list[int] = []
    for row_index, row in enumerate(matrix):
        row_values = tuple(int(v) for v in row)
        if len(row_values) != len(bit_values):
            raise CoreError("projection row and bit vector dimensions differ")
        acc = 0 if bias_values is None else int(bias_values[row_index])
        for coeff, bit in zip(row_values, bit_values):
            acc += coeff * bit
        projected.append(q_mod(acc, modulus))
    return tuple(projected)


def _combine_vectors(vectors: Iterable[tuple[int, ...]], modulus: int) -> tuple[int, ...]:
    items = tuple(vectors)
    if not items:
        raise CoreError("at least one vector is required")
    dimension = len(items[0])
    if any(len(item) != dimension for item in items):
        raise CoreError("projection dimensions differ")
    result = tuple(0 for _ in range(dimension))
    for item in items:
        result = tuple(q_add(a, b, modulus) for a, b in zip(result, item))
    return result


def compute_fact_delta(
    fact: FactFrame,
    *,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
) -> FactDelta:
    """Build deterministic fact-as-delta vector from subject/relation/object/evidence."""
    subject_phase = hash_to_phase(fact.subject.value, dimension=dimension, modulus=modulus, namespace="subject")
    relation_phase = hash_to_phase(fact.relation.value, dimension=dimension, modulus=modulus, namespace="relation")
    object_phase = hash_to_phase(fact.object.value, dimension=dimension, modulus=modulus, namespace="object")
    evidence_phase = hash_to_phase(fact.evidence.value, dimension=dimension, modulus=modulus, namespace="evidence")
    delta = _combine_vectors((subject_phase, relation_phase, object_phase, evidence_phase), modulus)
    weight = max(1, int(fact.confidence_bp))
    return FactDelta(fact=fact, delta=delta, weight=weight, modulus=modulus)


fact_delta = compute_fact_delta


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _json_default(getattr(value, key))
            for key in sorted(value.__dataclass_fields__.keys())
        }
    if isinstance(value, Mapping):
        return {str(k): _json_default(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    if isinstance(value, list):
        return [_json_default(v) for v in value]
    return value


def active_state_digest(payload: object) -> str:
    """Canonical SHA-256 digest for runtime state material."""
    canonical = json.dumps(_json_default(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _collision_report_from_vectors(
    vectors: Iterable[tuple[int, ...]],
    *,
    sample_kind: str,
    sample_count: int,
    dimension: int,
    same_input_stable: bool,
) -> CollisionCalibrationReport:
    seen: set[tuple[int, ...]] = set()
    exact_collision_count = 0
    coordinate_seen: list[set[int]] = [set() for _ in range(dimension)]
    coordinate_collision_count = 0

    for vector in vectors:
        if vector in seen:
            exact_collision_count += 1
        else:
            seen.add(vector)
        if len(vector) != dimension:
            raise CoreError("collision calibration vector dimension mismatch")
        for idx, value in enumerate(vector):
            phase = q_mod(value)
            if phase in coordinate_seen[idx]:
                coordinate_collision_count += 1
            else:
                coordinate_seen[idx].add(phase)

    coordinate_total = int(sample_count) * int(dimension)
    exact_collision_rate_bp = (exact_collision_count * 10000) // max(1, int(sample_count))
    coordinate_collision_rate_bp = (coordinate_collision_count * 10000) // max(1, coordinate_total)
    return CollisionCalibrationReport(
        sample_kind=sample_kind,
        sample_count=int(sample_count),
        dimension=int(dimension),
        exact_collision_count=exact_collision_count,
        exact_collision_rate_bp=exact_collision_rate_bp,
        coordinate_collision_count=coordinate_collision_count,
        coordinate_total=coordinate_total,
        coordinate_collision_rate_bp=coordinate_collision_rate_bp,
        same_input_stable=same_input_stable,
        collision_is_proof=False,
    )


def hash_to_phase_collision_rate_bp(
    entity_count: int,
    *,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
    namespace: str = "entity",
    prefix: str = "entity",
) -> CollisionCalibrationReport:
    """Report deterministic hash-to-phase collision rates for capacity auditing.

    This reports exact vector collisions and coordinate-level collisions. It does
    not interpret collisions as proof or evidence; downstream gates must still
    require evidence/proof/policy authorization.
    """
    if entity_count <= 0:
        raise CoreError("entity_count must be positive")
    if dimension <= 0:
        raise CoreError("dimension must be positive")
    stable = hash_to_phase(f"{prefix}_0", dimension=dimension, modulus=modulus, namespace=namespace) == hash_to_phase(
        f" {prefix}_0 ", dimension=dimension, modulus=modulus, namespace=namespace
    )
    vectors = (
        hash_to_phase(f"{prefix}_{idx}", dimension=dimension, modulus=modulus, namespace=namespace)
        for idx in range(int(entity_count))
    )
    return _collision_report_from_vectors(
        vectors,
        sample_kind="hash_to_phase",
        sample_count=int(entity_count),
        dimension=int(dimension),
        same_input_stable=stable,
    )


def fact_delta_collision_rate_bp(
    fact_count: int,
    *,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
    prefix: str = "fact",
) -> CollisionCalibrationReport:
    """Report deterministic fact-delta collision rates for capacity auditing."""
    if fact_count <= 0:
        raise CoreError("fact_count must be positive")
    if dimension <= 0:
        raise CoreError("dimension must be positive")

    stable_fact = FactFrame(
        subject=EntityId(f"{prefix}_subject_0"),
        relation=RelationId("located_in"),
        object=EntityId(f"{prefix}_object_0"),
        evidence=EvidenceId(f"{prefix}_evidence_0"),
    )
    stable_same = FactFrame(
        subject=EntityId(f" {prefix}_subject_0 "),
        relation=RelationId(" located_in "),
        object=EntityId(f" {prefix}_object_0 "),
        evidence=EvidenceId(f" {prefix}_evidence_0 "),
    )
    stable = compute_fact_delta(stable_fact, dimension=dimension, modulus=modulus).delta == compute_fact_delta(
        stable_same, dimension=dimension, modulus=modulus
    ).delta

    def _vectors() -> Iterable[tuple[int, ...]]:
        for idx in range(int(fact_count)):
            fact = FactFrame(
                subject=EntityId(f"{prefix}_subject_{idx}"),
                relation=RelationId("located_in"),
                object=EntityId(f"{prefix}_object_{idx % 9973}"),
                evidence=EvidenceId(f"{prefix}_evidence_{idx}"),
            )
            yield compute_fact_delta(fact, dimension=dimension, modulus=modulus).delta

    return _collision_report_from_vectors(
        _vectors(),
        sample_kind="fact_delta",
        sample_count=int(fact_count),
        dimension=int(dimension),
        same_input_stable=stable,
    )
