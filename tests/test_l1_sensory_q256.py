import ast
from pathlib import Path

from htce_origin import HTCERuntime
from htce_origin.body.layers import L123Body
from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.sensory.l1_encoder import L1SensoryEncoder, RawSensorPacket

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_l1_sensory_encoder_has_no_float_literals_and_uses_q256_default():
    assert _float_constant_count(ROOT / "htce_origin" / "sensory" / "l1_encoder.py") == 0
    encoder = L1SensoryEncoder(torus_dimension=4, input_dim=8)
    assert encoder.modulus == Q256_MODULUS
    assert encoder.projection.dimension == 4
    assert encoder.projection.input_dim == 8
    assert all(weight in (-1, 0, 1) for row in encoder.projection.weights for weight in row)


def test_l1_quantization_maps_integer_sensor_range_into_full_q256_torus():
    encoder = L1SensoryEncoder(torus_dimension=4, input_dim=4)
    packet = RawSensorPacket("vision", (0, 128, 255), sample_min=0, sample_max=255, evidence_id="l1_ev")

    q_state = encoder.quantize_packet(packet)

    assert q_state.q_values[0] == 0
    assert q_state.q_values[1] == (128 * (Q256_MODULUS - 1)) // 255
    assert q_state.q_values[2] == Q256_MODULUS - 1
    assert q_state.q_values[3] == 0
    assert any(value >= 2**128 for value in q_state.q_values)


def test_l1_projection_and_delta_are_deterministic_and_set_body_l1_exactly():
    encoder = L1SensoryEncoder(torus_dimension=4, input_dim=8)
    packet = RawSensorPacket("proprio", (1, 2, 3, 4), sample_min=0, sample_max=10, evidence_id="l1_prop")
    body = L123Body(dimension=4, modulus=Q256_MODULUS)

    encoded_a = encoder.encode(packet, current_l1_phase=body.l1.vector)
    encoded_b = encoder.encode(packet, current_l1_phase=body.l1.vector)
    transition = body.observe_l1_encoded(encoded_a)

    assert encoded_a.observed_phase == encoded_b.observed_phase
    assert body.l1.vector == encoded_a.observed_phase
    assert transition.target_layer.value == "L1"
    assert body.l2.clock == 0
    assert body.l3.clock == 0


def test_runtime_l1_packet_commit_does_not_create_l2_fact():
    rt = HTCERuntime()
    rt.wake()
    packet = RawSensorPacket("audio", (0, 10, 20, 30), sample_min=0, sample_max=30, evidence_id="l1_audio_ev")

    response = rt.observe_l1_packet(packet)

    assert response.decision.kind.value == "hypothesis"
    assert response.decision.trace_id
    assert rt.body.l1.clock == 1
    assert rt.body.l2.clock == 0
    assert rt.health()["latest_fact_count"] == 0
    assert response.diagnostics["encoder"]["prediction_error_bp"] >= 0
