from pathlib import Path
import pytest
import ast

from htce_origin.kernel.core import EvidenceId, FactFrame, EntityId, RelationId, fact_delta
from htce_origin.body.layers import InterLevelProjection, L123Body, LayerName
from htce_origin.kernel.q16 import DEFAULT_MODULUS
from htce_origin.body.runtime import HTCERuntime, RuntimeRequest
from htce_origin.governance.policy import DecisionKind

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_body_layers_memory_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "body" / "layers.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "body" / "memory.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "body" / "runtime.py") == 0


def test_l123_body_state_is_q16_torus_and_transitions_digest():
    body = L123Body(dimension=8)
    before = body.digest()
    event = body.observe_l1("Mary located_in office", evidence_id="event42")
    after = body.digest()

    assert before != after
    assert event.source_layer == LayerName.L1
    assert event.target_layer == LayerName.L1
    assert all(0 <= value < DEFAULT_MODULUS for value in body.l1.vector)
    assert body.l1.clock == 1
    assert body.l2.clock == 0


def test_inter_level_projection_is_deterministic_integer_map():
    projection = InterLevelProjection(dimension=4)
    vector = (1, 2, 3, 4)
    first = projection.project(vector, source=LayerName.L1, target=LayerName.L2)
    second = projection.project(vector, source=LayerName.L1, target=LayerName.L2)

    assert first == second
    assert len(first) == 4
    assert all(isinstance(value, int) for value in first)
    assert all(0 <= value < DEFAULT_MODULUS for value in first)


def test_l2_fact_commit_changes_l2_not_l3():
    body = L123Body(dimension=8)
    fact = FactFrame(EntityId("Mary"), RelationId("located_in"), EntityId("office"), EvidenceId("event42"))
    delta = fact_delta(fact, dimension=8)
    before_l2 = body.l2.digest
    before_l3 = body.l3.digest
    transition = body.commit_l2_fact(delta)

    assert body.l2.digest != before_l2
    assert body.l3.digest == before_l3
    assert transition.target_layer == LayerName.L2
    assert body.l2.clock == 1


def test_runtime_fact_commit_updates_body_and_memory():
    runtime = HTCERuntime()
    before = runtime.health()
    response = runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    after = runtime.health()

    assert response.decision.kind == DecisionKind.ANSWER
    assert response.decision.trace_id
    assert after["l2_clock"] == before["l2_clock"] + 1
    assert after["latest_fact_count"] == 1
    assert after["body_digest"] != before["body_digest"]
    assert runtime.trace.verify()


def test_l1_simulated_sensory_observation_updates_only_l1():
    from htce_origin.control.homeostasis import SensoryObservation
    from htce_origin.body.layers import simulated_observation_delta

    body = L123Body(dimension=4)
    observation = SensoryObservation(
        modality="vision_sim",
        value="edge_cluster",
        intensity_bp=8000,
        reliability_bp=9000,
        phase=(10, 20, 30, 40),
        evidence_id="sim_obs_l1",
    )
    delta = simulated_observation_delta(observation, dimension=4)
    before = body.digest()
    transition = body.observe_simulated(observation)

    assert len(delta) == 4
    assert transition.target_layer == LayerName.L1
    assert transition.source_layer == LayerName.L1
    assert transition.evidence_id == "sim_obs_l1"
    assert body.digest() != before
    assert body.l1.clock == 1
    assert body.l2.clock == 0
    assert body.l3.clock == 0
    assert all(0 <= value < DEFAULT_MODULUS for value in body.l1.vector)


def test_l1_simulated_observation_rejects_real_sensor_commit_flag():
    from htce_origin.body.layers import LayerError, simulated_observation_delta

    bad_observation = {
        "modality": "camera",
        "value": "real_frame",
        "intensity_bp": 5000,
        "reliability_bp": 5000,
        "phase": (1, 2, 3, 4),
        "real_sensor_commit_allowed": True,
    }

    with pytest.raises(LayerError):
        simulated_observation_delta(bad_observation, dimension=4)
