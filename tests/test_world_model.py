import ast
from pathlib import Path

import pytest

from htce_origin.kernel.core import TorusVector
from htce_origin.kernel.q16 import Q16_MODULUS
from htce_origin.cognition.world import (
    AdaptiveQ256Dynamics,
    Q256WorldAction,
    Q256WorldModel,
    WorldModelError,
    q256_saturating_signed,
    signed_delta_n,
)

ROOT = Path(__file__).resolve().parents[1]


def q16_action(*args, **kwargs):
    kwargs.setdefault("modulus", Q16_MODULUS)
    return Q256WorldAction(*args, **kwargs)


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            count += 1
    return count


def test_world_model_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "cognition" / "world.py") == 0


def test_predicts_deterministic_transition_with_wraparound():
    model = Q256WorldModel(dimension=3, modulus=Q16_MODULUS)
    state = TorusVector((65535, 10, 0), Q16_MODULUS)
    action = q16_action(name="move", delta=(1, 5, 65535))

    first = model.predict_next_state(state, action)
    second = model.predict_next_state(state, action)

    assert first.predicted_state.phases == (0, 15, 65535)
    assert first.predicted_digest == second.predicted_digest
    assert first.confidence_bp == 10000
    assert first.uncertainty_bp == 0


def test_prediction_error_measured_for_observed_next_state():
    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS)
    prediction = model.predict_next_state(TorusVector((10, 20), Q16_MODULUS), q16_action("step", (5, 5)))

    matched = model.measure_prediction_error(prediction, TorusVector((15, 25), Q16_MODULUS))
    mismatched = model.measure_prediction_error(prediction, TorusVector((32768, 32768), Q16_MODULUS))

    assert matched.loss == 0
    assert matched.error_bp == 0
    assert matched.matched is True
    assert mismatched.loss > 0
    assert mismatched.error_bp > 0
    assert mismatched.matched is False


def test_high_error_lowers_confidence_and_raises_uncertainty():
    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS, high_error_threshold_bp=100)
    prediction = model.predict_next_state(TorusVector((0, 0), Q16_MODULUS), q16_action("bad_step", (0, 0)))

    observed = model.observe(prediction, TorusVector((32768, 32768), Q16_MODULUS))

    assert observed.error is not None
    assert observed.error.error_bp >= 100
    assert model.self_model.uncertainty_bp > 0
    assert observed.confidence_bp < 10000
    assert model.can_confidently_answer(max_uncertainty_bp=0) is False


def test_imagined_rollout_composes_explicit_q16_actions():
    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS)
    actions = (
        q16_action("east", (1, 0)),
        q16_action("north", (0, 2)),
        q16_action("wrap", (65535, 65535)),
    )

    rollout = model.imagined_rollout(TorusVector((0, 0), Q16_MODULUS), actions)

    assert rollout.final_state.phases == (0, 1)
    assert len(rollout.steps) == 3
    assert [step.action_name for step in rollout.steps] == ["east", "north", "wrap"]
    assert rollout.confidence_bp == 10000
    assert rollout.uncertainty_bp == 0


def test_world_model_cannot_fabricate_facts():
    action = q16_action("explicit_delta_only", (1, 2, 3))
    model = Q256WorldModel(dimension=3, modulus=Q16_MODULUS)
    prediction = model.predict_next_state(TorusVector((0, 0, 0), Q16_MODULUS), action)
    rollout = model.imagined_rollout(TorusVector((0, 0, 0), Q16_MODULUS), (action,))

    forbidden_fact_fields = {"subject", "relation", "object", "fact", "claim"}
    assert forbidden_fact_fields.isdisjoint(action.__dataclass_fields__)
    assert forbidden_fact_fields.isdisjoint(prediction.__dataclass_fields__)
    assert forbidden_fact_fields.isdisjoint(rollout.__dataclass_fields__)
    assert prediction.predicted_state.phases == (1, 2, 3)


def test_world_model_rejects_dimension_mismatch():
    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS)
    with pytest.raises(WorldModelError):
        model.predict_next_state(TorusVector((0, 0), Q16_MODULUS), q16_action("bad", (1, 2, 3)))


def test_q256_saturating_signed_and_signed_delta_are_integer_bounded():
    assert q256_saturating_signed(0) == 0
    assert q256_saturating_signed(10**9, limit=Q16_MODULUS // 4) <= 16384
    assert q256_saturating_signed(-(10**9), limit=Q16_MODULUS // 4) >= -16384
    assert signed_delta_n(65535, 0, Q16_MODULUS) == 1
    assert signed_delta_n(0, 65535, Q16_MODULUS) == -1


def test_adaptive_world_model_is_opt_in_and_produces_audited_details():
    deterministic = Q256WorldModel(dimension=2, modulus=Q16_MODULUS)
    adaptive = Q256WorldModel(dimension=2, modulus=Q16_MODULUS, adaptive_enabled=True)
    state = TorusVector((100, 200), Q16_MODULUS)
    action = q16_action("adaptive_step", (3, 4), metadata={"domain": "grid"})

    deterministic_result = deterministic.predict_next_state(state, action)
    adaptive_result = adaptive.predict_next_state(state, action)

    assert deterministic_result.adaptive_details is None
    assert adaptive_result.adaptive_details is not None
    assert adaptive_result.predicted_state.dimension == 2
    assert adaptive_result.action_name == "adaptive_step"


def test_adaptive_update_uses_circular_error_without_creating_facts():
    dynamics = AdaptiveQ256Dynamics(dimension=2, modulus=Q16_MODULUS, learning_rate_phase=1)
    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS, adaptive_enabled=True, adaptive_dynamics=dynamics)
    state = TorusVector((0, 0), Q16_MODULUS)
    action = q16_action("learn_shift", (0, 0), metadata={"context": "same"})

    first = model.predict_adaptive(state, action)
    observed = first.predicted_state.add((1, 0))
    before = model.measure_prediction_error(first, observed)
    updated = model.update_from_observation(first, observed)
    second = model.predict_adaptive(state, action)
    after = model.measure_prediction_error(second, observed)

    assert updated.error is not None
    assert before.loss > 0
    assert after.loss < before.loss
    assert first.adaptive_details is not None
    forbidden_fact_fields = {"subject", "relation", "object", "fact", "claim"}
    assert forbidden_fact_fields.isdisjoint(first.adaptive_details.__dataclass_fields__)


def test_adaptive_dynamics_rejects_dimension_mismatch():
    dynamics = AdaptiveQ256Dynamics(dimension=2)
    with pytest.raises(WorldModelError):
        dynamics.predict(TorusVector((0, 0, 0), Q16_MODULUS), q16_action("bad", (1, 2, 3)))


def test_adaptive_world_model_ablation_reduces_circular_loss_over_sequence():
    from htce_origin.cognition.world import WorldModelAblationRow

    model = Q256WorldModel(dimension=2, modulus=Q16_MODULUS, adaptive_enabled=True)
    state = TorusVector((0, 0), Q16_MODULUS)
    action = q16_action("ablate_shift", (0, 0), metadata={"context": "p2"})
    initial_prediction = model.predict_adaptive(state, action)
    observed = initial_prediction.predicted_state.add((8, 0))
    row = WorldModelAblationRow(state, action, observed)

    report = model.adaptive_ablation((row,), update_passes=12)

    assert report.baseline_loss > 0
    assert report.adaptive_loss_after_k < report.baseline_loss
    assert report.improvement_bp > 0
    assert report.update_count == 12
    assert report.transition_count == 1
    assert report.facts_created is False
    assert report.integer_only is True


def test_q256_world_model_default_profile_predicts_large_range_wraparound():
    from htce_origin.cognition.world import Q256WorldModel
    from htce_origin.kernel.q16 import Q256_MODULUS, DEFAULT_MODULUS

    assert DEFAULT_MODULUS == Q256_MODULUS
    model = Q256WorldModel(dimension=2)
    state = TorusVector((Q256_MODULUS - 1, Q256_MODULUS // 2), Q256_MODULUS)
    action = Q256WorldAction("q256_step", (1, Q256_MODULUS // 2), modulus=Q256_MODULUS)

    prediction = model.predict_next_state(state, action)

    assert prediction.predicted_state.phases == (0, 0)
    assert prediction.predicted_state.modulus == Q256_MODULUS


def test_world_model_expected_free_energy_selects_closer_q256_action():
    from htce_origin.kernel.q16 import Q256_MODULUS

    model = Q256WorldModel(dimension=1, modulus=Q256_MODULUS)
    state = TorusVector((0,), Q256_MODULUS)
    target = TorusVector((10,), Q256_MODULUS)
    near = Q256WorldAction("near", (10,), modulus=Q256_MODULUS)
    far = Q256WorldAction("far", (Q256_MODULUS // 2,), modulus=Q256_MODULUS)

    selected = model.select_min_expected_free_energy_action(state, (far, near), target_state=target)
    near_eval = model.evaluate_action_expected_free_energy(state, near, target_state=target)
    far_eval = model.evaluate_action_expected_free_energy(state, far, target_state=target)

    assert selected.action_name == "near"
    assert near_eval.expected_free_energy_bp < far_eval.expected_free_energy_bp
    assert near_eval.risk_bp == 0
    assert far_eval.risk_bp > 0


def test_grounded_perception_action_step_updates_self_model_without_fact_fields_q256():
    from htce_origin.kernel.q16 import Q256_MODULUS

    model = Q256WorldModel(dimension=2, modulus=Q256_MODULUS, adaptive_enabled=True)
    state = TorusVector((0, 0), Q256_MODULUS)
    action = Q256WorldAction("sense_move", (0, 0), metadata={"sensor": "l1"}, modulus=Q256_MODULUS)
    observed = TorusVector((1, 0), Q256_MODULUS)

    result = model.grounded_perception_action_step(
        state,
        (action,),
        observed,
        context={"sensor": "l1"},
        complexity_bp=100,
        novelty_gain_bp=0,
        goal_progress_bp=0,
    )

    assert result.selected_action_name == "sense_move"
    assert result.surprise_bp >= 0
    assert model.self_model.observations == 1
    assert result.adaptive_updated is True
    assert model.adaptive_dynamics is not None
    assert len(model.adaptive_dynamics.correction_memory) == 1
    forbidden_fact_fields = {"subject", "relation", "object", "fact", "claim"}
    assert forbidden_fact_fields.isdisjoint(result.__dataclass_fields__)
    assert forbidden_fact_fields.isdisjoint(result.selected_evaluation.__dataclass_fields__)
