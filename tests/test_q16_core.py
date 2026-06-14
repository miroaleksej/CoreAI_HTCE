import ast
import itertools
from pathlib import Path

import pytest

from htce_origin.kernel.core import (
    DEFAULT_TORUS_DIMENSION,
    EntityId,
    EvidenceId,
    FactFrame,
    RelationId,
    TorusVector,
    active_state_digest,
    fact_delta,
    hash_to_phase,
    hash_to_phase_collision_rate_bp,
    fact_delta_collision_rate_bp,
)
from htce_origin.kernel.q16 import (
    COS_SCALE,
    DEFAULT_MODULUS,
    N,
    Q16_MODULUS,
    Q256_MODULUS,
    Q16Vector,
    q_add,
    q_cos_lut,
    q_delta,
    q_distance,
    q_distance_vector,
    q_mod,
    q_mul,
    q_sub,
    q_toroidal_loss_lut,
    q_vector_add,
    q16_property_stress,
)

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            count += 1
    return count


def test_scalar_wraparound_and_distance_acceptance():
    assert N == 2**256
    assert DEFAULT_MODULUS == Q256_MODULUS
    assert Q16_MODULUS == 65536
    assert q_add(Q256_MODULUS - 1, 1) == 0
    assert q_sub(0, 1) == Q256_MODULUS - 1
    assert q_mul(Q256_MODULUS - 1, 2) == Q256_MODULUS - 2
    assert q_mod(-1) == Q256_MODULUS - 1
    assert q_distance(0, Q256_MODULUS - 1) == 1
    assert q_delta(0, Q256_MODULUS - 1) == 1
    assert q_delta(0, Q256_MODULUS // 2) == Q256_MODULUS // 2


def test_integer_torus_vector_operations():
    vector = Q16Vector((0, 65535, 65536, -1), Q16_MODULUS)
    assert vector.values == (0, 65535, 0, 65535)
    assert vector.add((1, 1, 1, 2)).values == (1, 0, 1, 1)
    assert vector.sub((1, 65535, 1, 65535)).values == (65535, 0, 65535, 0)
    assert vector.distance_sq((65535, 0, 0, 0)) == 1 + 1 + 0 + 1
    assert q_vector_add((65535, 1), (1, 65535), Q16_MODULUS) == (0, 0)


def test_vector_dimension_mismatch_is_blocked():
    with pytest.raises(Exception):
        q_distance_vector((1, 2), (1, 2, 3))


def test_cosine_lut_and_loss_are_integer_only_and_periodic():
    assert isinstance(q_cos_lut(0), int)
    assert q_cos_lut(0) == COS_SCALE
    assert q_cos_lut(Q256_MODULUS // 2) == -COS_SCALE
    assert q_cos_lut(0) == q_cos_lut(Q256_MODULUS)
    assert q_toroidal_loss_lut(0) == 0
    assert q_toroidal_loss_lut(Q256_MODULUS // 2) == 65535
    assert q_toroidal_loss_lut(1) >= 0


def test_runtime_float_count_is_zero_in_q16_and_core():
    assert _float_constant_count(ROOT / "htce_origin" / "kernel" / "q16.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "kernel" / "core.py") == 0


def test_hash_to_phase_is_deterministic_and_stable_for_same_entity():
    mary_1 = hash_to_phase("Mary", dimension=DEFAULT_TORUS_DIMENSION)
    mary_2 = hash_to_phase(" mary ", dimension=DEFAULT_TORUS_DIMENSION)
    assert mary_1 == mary_2
    assert len(mary_1) == 64
    assert all(0 <= value < Q256_MODULUS for value in mary_1)
    assert any(value >= 2**128 for value in mary_1)


def test_different_entities_low_collision_profile():
    names = [f"entity_{idx}" for idx in range(64)]
    vectors = [hash_to_phase(name, dimension=32) for name in names]
    exact_duplicates = len(vectors) - len(set(vectors))
    assert exact_duplicates == 0

    # Coordinate-level collision rate should stay very low for deterministic hash projection.
    total_coordinates = 0
    coordinate_collisions = 0
    for left, right in itertools.combinations(vectors, 2):
        for a, b in zip(left, right):
            total_coordinates += 1
            coordinate_collisions += int(a == b)
    collision_bp = (coordinate_collisions * 10000) // total_coordinates
    assert collision_bp < 10


def test_fact_delta_is_deterministic_and_evidence_sensitive():
    fact = FactFrame(
        subject=EntityId("Mary"),
        relation=RelationId("located_in"),
        object=EntityId("Office"),
        evidence=EvidenceId("event:42"),
    )
    same = FactFrame(
        subject=EntityId("mary"),
        relation=RelationId("located_in"),
        object=EntityId("office"),
        evidence=EvidenceId("event:42"),
    )
    changed_evidence = FactFrame(
        subject=EntityId("Mary"),
        relation=RelationId("located_in"),
        object=EntityId("Office"),
        evidence=EvidenceId("event:43"),
    )
    delta_1 = fact_delta(fact)
    delta_2 = fact_delta(same)
    delta_3 = fact_delta(changed_evidence)
    assert delta_1.delta == delta_2.delta
    assert delta_1.delta != delta_3.delta
    assert len(delta_1.delta) == 64
    assert delta_1.weight == 10000


def test_torus_vector_and_state_digest_are_deterministic():
    state = TorusVector.zero().add((1 for _ in range(64)))
    assert state.dimension == 64
    assert state.phases[:3] == (1, 1, 1)
    assert active_state_digest(state) == active_state_digest(state)
    assert len(active_state_digest(state)) == 64


def test_q16_property_based_release_invariants():
    samples = (0, 1, 2, 3, 7, 31, 255, 1024, 32767, 32768, 65534, 65535)
    for a in samples:
        assert q_distance(a, a) == 0
        assert q_distance(a, q_add(a, N - 1)) == 1
        for b in samples:
            assert q_distance(a, b) == q_distance(b, a)
            for c in (0, 1, 32768, 65535):
                assert q_add(q_add(a, b), c) == q_add(a, q_add(b, c))


def test_q16_deterministic_100k_property_stress():
    report = q16_property_stress(sample_count=100000, seed=42)
    assert report.sample_count == 100000
    assert report.passed
    assert report.total_failures == 0


def test_kernel_collision_capacity_reports_do_not_become_proof():
    entity_report = hash_to_phase_collision_rate_bp(1000, dimension=16, prefix="capacity_entity")
    fact_report = fact_delta_collision_rate_bp(1000, dimension=16, prefix="capacity_fact")

    assert entity_report.same_input_stable
    assert fact_report.same_input_stable
    assert entity_report.sample_count == 1000
    assert fact_report.sample_count == 1000
    assert entity_report.exact_collision_rate_bp >= 0
    assert fact_report.exact_collision_rate_bp >= 0
    assert entity_report.coordinate_collision_rate_bp >= 0
    assert fact_report.coordinate_collision_rate_bp >= 0
    assert entity_report.collision_is_proof is False
    assert fact_report.collision_is_proof is False
    assert entity_report.passed_boundary
    assert fact_report.passed_boundary


def test_kernel_collision_capacity_supports_1k_10k_100k_profiles():
    for entity_count in (1000, 10000, 100000):
        entity_report = hash_to_phase_collision_rate_bp(entity_count, dimension=8, prefix=f"entity_profile_{entity_count}")
        fact_report = fact_delta_collision_rate_bp(entity_count, dimension=8, prefix=f"fact_profile_{entity_count}")
        assert entity_report.sample_count == entity_count
        assert fact_report.sample_count == entity_count
        assert entity_report.exact_collision_rate_bp >= 0
        assert fact_report.exact_collision_rate_bp >= 0
        assert entity_report.coordinate_collision_rate_bp >= 0
        assert fact_report.coordinate_collision_rate_bp >= 0
        assert entity_report.collision_is_proof is False
        assert fact_report.collision_is_proof is False
        assert entity_report.passed_boundary
        assert fact_report.passed_boundary
