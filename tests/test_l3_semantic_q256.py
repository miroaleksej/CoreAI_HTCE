import ast
from pathlib import Path

from htce_origin.body.layers import L123Body
from htce_origin.cognition.learning import (
    L3CouplingMatrix,
    L3DeductiveEngine,
    L3SemanticState,
    ToroidalSleepConsolidator,
)
from htce_origin.kernel.q16 import COS_SCALE, DEFAULT_MODULUS, Q256_MODULUS, q_add, q_distance, q_sin_lut
from htce_origin.topology.guard import validate_l3_semantic_window

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_l3_runtime_sources_remain_float_free():
    assert _float_constant_count(ROOT / "htce_origin" / "cognition" / "learning.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "topology" / "guard.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "kernel" / "q16.py") == 0


def test_q256_sine_lut_uses_full_modulus_quarter_turns():
    assert abs(q_sin_lut(0)) <= 1
    assert q_sin_lut(DEFAULT_MODULUS // 4) == COS_SCALE
    assert abs(q_sin_lut(DEFAULT_MODULUS // 2)) <= 1
    assert q_sin_lut((DEFAULT_MODULUS * 3) // 4) == -COS_SCALE
    assert DEFAULT_MODULUS == 2**256


def test_l3_phase_relaxation_reduces_two_phase_distance_q256():
    quarter = Q256_MODULUS // 4
    engine = L3DeductiveEngine(dimension=2, relaxation_steps=4, eta_bp=1500)
    engine.set_state((0, quarter))
    matrix = L3CouplingMatrix(((0, COS_SCALE), (COS_SCALE, 0)), 2)
    engine.load_coupling_matrix(matrix)

    before = q_distance(engine.state.phases[0], engine.state.phases[1])
    report = engine.relax_with_report()
    after = q_distance(engine.state.phases[0], engine.state.phases[1])

    assert report.answer_authorized is False
    assert report.max_phase_motion > 0
    assert after < before
    assert all(0 <= value < Q256_MODULUS for value in engine.state.phases)


def test_sleep_consolidator_exports_l3_engine_and_q15_coupling_matrix():
    residual = (Q256_MODULUS // 1024, Q256_MODULUS // 2048, Q256_MODULUS // 4096, Q256_MODULUS // 8192)
    starts = ((1, 2, 3, 4), (100, 200, 300, 400))
    ends = tuple(tuple(q_add(value, delta) for value, delta in zip(start, residual)) for start in starts)

    consolidator = ToroidalSleepConsolidator(dim_l2=4, dim_l3=4, learning_rate_bp=10000, sparsity_lambda_bp=0)
    report = consolidator.consolidate_offline(starts, ends, epochs=1)
    state = consolidator.export_l3_semantic_state()
    matrix = consolidator.derive_l3_coupling_matrix(starts, ends)
    engine = consolidator.build_l3_deductive_engine(starts, ends, relaxation_steps=2)

    assert report.loss_improved is True
    assert isinstance(state, L3SemanticState)
    assert state.phases == residual
    assert isinstance(matrix, L3CouplingMatrix)
    assert matrix.dimension == 4
    assert all(-COS_SCALE <= value <= COS_SCALE for row in matrix.weights for value in row)
    assert isinstance(engine, L3DeductiveEngine)
    assert engine.state.phases == residual


def test_l3_semantic_state_can_commit_to_body_when_dimension_matches():
    body = L123Body(dimension=4)
    state = L3SemanticState((10, 20, 30, 40))
    transition = body.commit_l3_semantic_state(state, evidence_id="sleep_l3")

    assert transition.target_layer.value == "L3"
    assert body.l3.vector == state.phases
    assert body.l2.clock == 0


def test_l3_integer_topology_guard_validates_coherent_window():
    base = (1000, 2000, 3000, 4000)
    states = tuple(tuple(q_add(value, offset) for value in base) for offset in (0, 1, 2, 3))
    report = validate_l3_semantic_window(states, radius=16, expected_beta1=None)

    assert report.passed is True
    assert report.beta0 == 1
    assert report.point_count == 4
