import ast
from pathlib import Path

from htce_origin import HTCERuntime
from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy
from htce_origin.cognition.world import Q256WorldAction, Q256WorldModel
from htce_origin.kernel.core import TorusVector
from htce_origin.kernel.q16 import Q256_MODULUS

ROOT = Path(__file__).resolve().parents[1]


def test_world_selection_uses_expected_free_energy_raw():
    model = Q256WorldModel(dimension=1, modulus=Q256_MODULUS)
    state = TorusVector((0,), Q256_MODULUS)
    target = TorusVector((10,), Q256_MODULUS)
    near = Q256WorldAction("near", (10,), modulus=Q256_MODULUS)
    far = Q256WorldAction("far", (Q256_MODULUS // 2,), modulus=Q256_MODULUS)

    near_eval = model.evaluate_action_expected_free_energy(state, near, target_state=target)
    far_eval = model.evaluate_action_expected_free_energy(state, far, target_state=target)
    selected = model.select_min_expected_free_energy_action(state, (far, near), target_state=target)

    assert selected.action_name == "near"
    assert near_eval.expected_free_energy_raw < far_eval.expected_free_energy_raw
    assert near_eval.risk_raw == 0
    assert far_eval.risk_raw > 0


def test_closed_loop_trace_contains_raw_decision_scores():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_closed_loop_simulation(steps=3)

    assert report.trace_verified is True
    assert all(isinstance(step.efe_raw, int) for step in report.steps)
    assert all(isinstance(step.surprise_raw, int) for step in report.steps)
    assert report.total_efe_raw == sum(step.efe_raw for step in report.steps)
    assert report.total_surprise_raw == sum(step.surprise_raw for step in report.steps)


def test_habitat_gate_can_block_on_raw_model_error_even_when_bp_is_low():
    policy = SimulationHabitatPolicy()
    gate = HabitatGateInput(
        model_error_bp=0,
        model_error_raw=101,
        max_model_error_raw=100,
        action_class="simulated",
    )
    decision = policy.evaluate(gate)

    assert decision.allowed is False
    assert "raw model error" in decision.reason


def test_world_selector_does_not_use_expected_free_energy_bp_key():
    source = (ROOT / "htce_origin" / "cognition" / "world.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    suspect = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            text = ast.get_source_segment(source, node) or ""
            if "expected_free_energy_bp" in text:
                suspect.append(text)
    assert suspect == []
