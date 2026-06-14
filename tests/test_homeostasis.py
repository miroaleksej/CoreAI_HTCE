import ast
from pathlib import Path

import pytest

from htce_origin.control.homeostasis import (
    ActiveInferenceSurrogate,
    ControlSignal,
    DomainNoise,
    HomeostasisError,
    HomeostaticActionEffect,
    HomeostaticSetpoint,
    HomeostaticState,
    HomeostaticWeights,
)
from htce_origin.control.planner import ExpectedFreeEnergyScorer, PlanStep, ProofGuidedPlanner, RiskAwareUtility
from htce_origin.governance.policy import DecisionKind

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_homeostasis_and_planner_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "control" / "homeostasis.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "control" / "planner.py") == 0


def test_viability_score_is_bounded_and_pressure_sensitive():
    healthy = HomeostaticState()
    stressed = HomeostaticState(risk_bp=9000, uncertainty_bp=8000)

    assert healthy.viability_bp() == 10000
    assert 0 <= stressed.viability_bp() < healthy.viability_bp()


def test_high_uncertainty_triggers_ask_clarification():
    surrogate = ActiveInferenceSurrogate()
    evaluation = surrogate.evaluate(HomeostaticState(uncertainty_bp=9000))
    planner = ProofGuidedPlanner(surrogate)
    result = planner.plan_with_homeostasis("clarify_goal", HomeostaticState(uncertainty_bp=9000))

    assert evaluation.signal == ControlSignal.ASK_CLARIFICATION
    assert evaluation.allow_simulated_action is False
    assert result.decision == DecisionKind.ASK_CLARIFICATION
    assert not result.allow_simulated_action
    assert "uncertainty" in result.reason


def test_high_sleep_pressure_triggers_sleep_required():
    state = HomeostaticState(sleep_pressure_bp=9000)
    evaluation = ActiveInferenceSurrogate().evaluate(state)
    result = ProofGuidedPlanner().plan_with_homeostasis("continue", state)

    assert evaluation.signal == ControlSignal.SLEEP_REQUIRED
    assert result.decision == DecisionKind.SLEEP_REQUIRED
    assert not result.allow_simulated_action
    assert "sleep pressure" in result.reason


def test_high_risk_blocks_simulated_action():
    state = HomeostaticState(risk_bp=9500, novelty_bp=9000)
    evaluation = ActiveInferenceSurrogate().evaluate(state)
    result = ProofGuidedPlanner().plan_with_homeostasis(
        "explore",
        state,
        candidate_steps=(PlanStep("simulated_probe"),),
    )

    assert evaluation.signal == ControlSignal.BLOCK_SIMULATED_ACTION
    assert not evaluation.allow_simulated_action
    assert result.decision == DecisionKind.REFUSE
    assert result.steps == ()
    assert not result.allow_simulated_action
    assert "risk" in result.reason


def test_novelty_can_trigger_exploration_in_simulation():
    state = HomeostaticState(novelty_bp=8000, risk_bp=1000, uncertainty_bp=1000)
    result = ProofGuidedPlanner().plan_with_homeostasis(
        "inspect_unknown_pattern",
        state,
        candidate_steps=(PlanStep("simulated_inspection"),),
        proof_verified=True,
    )

    assert result.decision == DecisionKind.ACT_SIMULATED
    assert result.allow_simulated_action is True
    assert result.explore_simulation is True
    assert result.steps[0].action_name == "simulated_inspection"
    assert "novelty" in result.reason


def test_expected_free_energy_surrogate_is_integer_and_gap_sensitive():
    scorer = ExpectedFreeEnergyScorer()
    base = scorer.score(HomeostaticState())
    uncertain = scorer.score(HomeostaticState(uncertainty_bp=8000), evidence_gap_bp=5000)

    assert isinstance(base, int)
    assert isinstance(uncertain, int)
    assert 0 <= base <= 10000
    assert 0 <= uncertain <= 10000
    assert uncertain > base


def test_risk_aware_utility_penalizes_risk_uncertainty_deviation_and_evidence_gap():
    utility = RiskAwareUtility()
    safe = utility.utility_bp(goal_progress_bp=9000, risk_bp=0, uncertainty_bp=0, homeostatic_deviation_bp=0, evidence_gap_bp=0)
    unsafe = utility.utility_bp(goal_progress_bp=9000, risk_bp=8000, uncertainty_bp=8000, homeostatic_deviation_bp=8000, evidence_gap_bp=8000)

    assert safe == 9000
    assert unsafe < safe


def test_simulated_plan_requires_proof_before_planning():
    planner = ProofGuidedPlanner()
    refused = planner.plan("move_demo", proof_verified=False)
    accepted = planner.plan("move_demo", proof_verified=True)

    assert refused.decision == DecisionKind.REFUSE
    assert refused.steps == ()
    assert accepted.decision == DecisionKind.ACT_SIMULATED
    assert accepted.steps[0].simulated is True


def test_homeostasis_rejects_out_of_range_basis_points():
    with pytest.raises(HomeostasisError):
        HomeostaticState(energy_bp=10001)


def test_setpoint_weighted_deviation_is_integer_and_q16_bounded():
    surrogate = ActiveInferenceSurrogate(
        setpoint=HomeostaticSetpoint(energy_bp=10000, sleep_pressure_bp=0, risk_bp=0, integrity_bp=10000, novelty_bp=0, uncertainty_bp=0),
        weights=HomeostaticWeights(energy_bp=10, sleep_pressure_bp=10, risk_bp=30, integrity_bp=10, novelty_bp=10, uncertainty_bp=30),
    )
    healthy = HomeostaticState()
    stressed = HomeostaticState(energy_bp=6000, sleep_pressure_bp=2000, risk_bp=8000, integrity_bp=7000, uncertainty_bp=5000)

    assert surrogate.homeostatic_deviation_bp(healthy) == 0
    assert 0 < surrogate.homeostatic_deviation_bp(stressed) <= 10000
    assert surrogate.viability_bp(stressed) < surrogate.viability_bp(healthy)


def test_action_effects_change_body_state_deterministically():
    state = HomeostaticState(energy_bp=8000, sleep_pressure_bp=2000, risk_bp=1000, integrity_bp=9000, novelty_bp=1000, uncertainty_bp=2000)
    effect = HomeostaticActionEffect(
        energy_delta_bp=-500,
        risk_delta_bp=1200,
        novelty_delta_bp=700,
        uncertainty_delta_bp=-300,
        goal_progress_bp=2000,
    )
    surrogate = ActiveInferenceSurrogate()

    first = surrogate.apply_action_effect(state, effect)
    second = surrogate.apply_action_effect(state, effect)

    assert first == second
    assert first.energy_bp == 7500
    assert first.risk_bp == 2200
    assert first.novelty_bp == 1700
    assert first.uncertainty_bp == 1700


def test_sleep_action_effect_reduces_sleep_pressure_without_erasing_uncertainty():
    state = HomeostaticState(energy_bp=5000, sleep_pressure_bp=9000, uncertainty_bp=6000)
    slept = ActiveInferenceSurrogate().apply_action_effect(state, HomeostaticActionEffect.sleep_cycle(recovery_bp=4000))

    assert slept.sleep_pressure_bp == 5000
    assert slept.energy_bp > state.energy_bp
    assert 0 < slept.uncertainty_bp < state.uncertainty_bp


def test_high_risk_dominates_setpoint_viability():
    surrogate = ActiveInferenceSurrogate()
    low_risk = surrogate.viability_bp(HomeostaticState(risk_bp=1000, uncertainty_bp=1000))
    high_risk = surrogate.viability_bp(HomeostaticState(risk_bp=9000, uncertainty_bp=1000))

    assert high_risk < low_risk
    assert high_risk <= 500


def test_uncertainty_never_becomes_artificial_zero():
    state = HomeostaticState(uncertainty_bp=4500)
    effect = HomeostaticActionEffect(uncertainty_delta_bp=-1000, goal_progress_bp=1000)
    updated = ActiveInferenceSurrogate().apply_action_effect(state, effect)

    assert updated.uncertainty_bp == 3500
    assert updated.uncertainty_bp != 0


def test_domain_noise_is_seeded_and_deterministic():
    first = DomainNoise.from_seed("terrain-A", magnitude_bp=250)
    second = DomainNoise.from_seed("terrain-A", magnitude_bp=250)
    third = DomainNoise.from_seed("terrain-B", magnitude_bp=250)

    assert first == second
    assert first != third


def test_expected_free_energy_uses_model_error_complexity_progress_and_novelty_gain():
    surrogate = ActiveInferenceSurrogate()
    state = HomeostaticState(risk_bp=1000, uncertainty_bp=2000, novelty_bp=1000)
    plain = surrogate.expected_free_energy_bp(state, evidence_gap_bp=1000)
    hard = surrogate.expected_free_energy_bp(state, evidence_gap_bp=1000, model_error_bp=5000, complexity_bp=5000)
    useful = surrogate.expected_free_energy_bp(
        state,
        evidence_gap_bp=1000,
        model_error_bp=5000,
        complexity_bp=5000,
        novelty_gain_bp=5000,
        goal_progress_bp=5000,
    )

    assert hard > plain
    assert useful < hard


def test_curiosity_signal_uses_prediction_error_novelty_reliability_and_risk():
    from htce_origin.control.homeostasis import CuriosityDrive, SensoryObservation
    from htce_origin.kernel.q16 import Q16_MODULUS

    drive = CuriosityDrive(modulus=Q16_MODULUS)
    observation = SensoryObservation(
        modality="vision_sim",
        value="unexpected_edge",
        intensity_bp=7000,
        reliability_bp=8000,
        phase=(20000, 30000, 40000, 50000),
        evidence_id="sim_obs_1",
        modulus=Q16_MODULUS,
    )
    low_risk = drive.evaluate(
        predicted_phase=(0, 0, 0, 0),
        observation=observation,
        risk_bp=500,
    )
    high_risk = drive.evaluate(
        predicted_phase=(0, 0, 0, 0),
        observation=observation,
        risk_bp=9000,
    )

    assert low_risk.prediction_error_bp > 0
    assert low_risk.curiosity_bp > high_risk.curiosity_bp
    assert low_risk.explore_simulation is True
    assert high_risk.explore_simulation is False
    assert low_risk.real_sensor_commit_allowed is False


def test_hypothesis_testing_loop_never_authorizes_commit_or_real_sensor():
    from htce_origin.control.homeostasis import HypothesisTestingLoop, SensoryObservation

    state = HomeostaticState(risk_bp=1000, uncertainty_bp=1000)
    observation = SensoryObservation(
        modality="text_sim",
        value="new_symbol",
        intensity_bp=9000,
        reliability_bp=9000,
        phase=(12000, 22000, 32000, 42000),
        evidence_id="sim_obs_2",
    )
    result = HypothesisTestingLoop().evaluate(
        state=state,
        predicted_phase=(0, 0, 0, 0),
        observation=observation,
    )

    assert result.curiosity.curiosity_bp > 0
    assert result.next_state.novelty_bp >= state.novelty_bp
    assert result.next_state.uncertainty_bp >= state.uncertainty_bp
    assert result.commit_allowed is False
    assert result.real_sensor_commit_allowed is False
    assert result.suggested_action in {"simulate_hypothesis_test", "hold_observation_for_evidence"}


def test_sensory_observation_rejects_real_sensor_commit_authority():
    from htce_origin.control.homeostasis import SensoryObservation

    with pytest.raises(HomeostasisError):
        SensoryObservation(
            modality="camera",
            value="real_frame",
            intensity_bp=5000,
            reliability_bp=5000,
            phase=(1, 2, 3, 4),
            real_sensor_commit_allowed=True,
        )


def test_homeostasis_calibration_separates_safe_unsafe_and_uncertain_replay():
    from htce_origin.control.homeostasis import calibrate_homeostasis_weights

    safe_replay = (
        HomeostaticState(energy_bp=9500, integrity_bp=9600, risk_bp=300, uncertainty_bp=400, sleep_pressure_bp=200),
        HomeostaticState(energy_bp=9300, integrity_bp=9400, risk_bp=500, uncertainty_bp=600, sleep_pressure_bp=300),
    )
    unsafe_replay = (
        HomeostaticState(energy_bp=3500, integrity_bp=3000, risk_bp=9500, uncertainty_bp=5000, sleep_pressure_bp=3000),
        HomeostaticState(energy_bp=4000, integrity_bp=3500, risk_bp=9000, uncertainty_bp=5500, sleep_pressure_bp=3500),
    )
    uncertainty_replay = (
        HomeostaticState(energy_bp=8000, integrity_bp=8200, risk_bp=1000, uncertainty_bp=9200, sleep_pressure_bp=1000),
        HomeostaticState(energy_bp=7800, integrity_bp=8000, risk_bp=1200, uncertainty_bp=8800, sleep_pressure_bp=1200),
    )

    report = calibrate_homeostasis_weights(safe_replay, unsafe_replay, uncertainty_replay)
    surrogate = report.surrogate()
    safe_min = min(surrogate.viability_bp(state) for state in safe_replay)
    unsafe_max = max(surrogate.viability_bp(state) for state in unsafe_replay)
    uncertain_evaluations = tuple(surrogate.evaluate(state) for state in uncertainty_replay)

    assert report.safe_count == 2
    assert report.unsafe_count == 2
    assert report.uncertainty_count == 2
    assert safe_min >= 7000
    assert unsafe_max < safe_min
    assert all(row.signal == ControlSignal.ASK_CLARIFICATION for row in uncertain_evaluations)
    assert all(row.allow_simulated_action is False for row in uncertain_evaluations)
    assert report.uncertain_signal == ControlSignal.ASK_CLARIFICATION
