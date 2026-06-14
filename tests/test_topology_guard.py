import ast
from pathlib import Path

import pytest

from htce_origin.kernel.core import TorusVector
from htce_origin.kernel.q16 import Q16_MODULUS
from htce_origin.topology.guard import (
    CalibrationProfile,
    TopologyAction,
    TopologyError,
    TopologyGuard,
    TrajectoryWindow,
    profile_clean_torus,
    profile_noisy_torus,
    profile_partial_subtorus,
    profile_phase_shock,
    profile_short_path,
)

ROOT = Path(__file__).resolve().parents[1]


def tv(values):
    return TorusVector(values, Q16_MODULUS)


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_topology_guard_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "topology" / "guard.py") == 0


def test_normal_trajectory_passes():
    guard = TopologyGuard(CalibrationProfile(dimension=2, modulus=Q16_MODULUS))
    trajectory = TrajectoryWindow((
        tv((0, 0)),
        tv((128, 0)),
        tv((256, 0)),
        tv((384, 0)),
    ))

    decision = guard.evaluate_trajectory(trajectory)

    assert decision.passed is True
    assert decision.action == TopologyAction.PASS
    assert decision.anomaly_score_bp < 100
    assert decision.details["state_count"] == 4


def test_phase_shock_warns_for_large_delta_change_without_blocking():
    guard = TopologyGuard(CalibrationProfile(
        dimension=1,
        modulus=Q16_MODULUS,
        jump_warn_bp=9000,
        jump_block_bp=10000,
        shock_warn_bp=1000,
        shock_block_bp=9000,
        persistent_warn_bp=9000,
        persistent_block_bp=10000,
        dirty_warning_count=3,
    ))
    first = guard.evaluate_transition(tv((0,)), tv((100,)))
    second = guard.evaluate_transition(tv((100,)), tv((8000,)))

    assert first.passed is True
    assert second.passed is True
    assert second.action == TopologyAction.WARN
    assert "phase_shock_warn" in second.warnings
    assert second.details["shock_bp"] >= 1000


def test_phase_shock_blocks_for_large_local_jump():
    guard = TopologyGuard(CalibrationProfile(dimension=1, modulus=Q16_MODULUS, jump_block_bp=8000))

    decision = guard.evaluate_transition(tv((0,)), tv((32768,)))

    assert decision.passed is False
    assert decision.action == TopologyAction.BLOCK
    assert "phase_jump_block" in decision.warnings
    assert decision.details["jump_bp"] == 10000


def test_short_path_not_punished_for_beta1_not_equal_dimension():
    profile = CalibrationProfile(dimension=2, modulus=Q16_MODULUS, expected_beta1=64, min_betti_points=100)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0, 0)),
        tv((10, 0)),
        tv((20, 0)),
    ))

    decision = guard.evaluate_trajectory(trajectory, observed_beta1=0)

    assert decision.passed is True
    assert decision.details["beta_checked"] is False
    assert decision.details["beta_mismatch"] is False
    assert decision.action == TopologyAction.PASS


def test_calibrated_long_window_can_block_beta1_mismatch():
    profile = CalibrationProfile(dimension=1, modulus=Q16_MODULUS, expected_beta1=1, min_betti_points=3)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((tv((0,)), tv((10,)), tv((20,))), max_points=10)

    decision = guard.evaluate_trajectory(trajectory, observed_beta1=0)

    assert decision.passed is False
    assert decision.action == TopologyAction.BLOCK
    assert decision.details["beta_checked"] is True
    assert decision.details["beta_mismatch"] is True
    assert "calibrated_beta1_mismatch" in decision.warnings


def test_dirty_trajectory_marked_anomalous_by_repeated_warnings():
    profile = CalibrationProfile(
        dimension=1,
        modulus=Q16_MODULUS,
        jump_warn_bp=9000,
        jump_block_bp=10000,
        shock_warn_bp=1000,
        shock_block_bp=9000,
        persistent_warn_bp=10000,
        persistent_block_bp=10000,
        dirty_warning_count=2,
    )
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0,)),
        tv((100,)),
        tv((8000,)),
        tv((8100,)),
        tv((16000,)),
    ))

    decision = guard.evaluate_trajectory(trajectory)

    assert decision.passed is False
    assert decision.action == TopologyAction.BLOCK
    assert "dirty_trajectory" in decision.warnings
    assert decision.details["warning_count"] >= 2


def test_persistent_anomaly_score_smooths_history():
    guard = TopologyGuard(CalibrationProfile(
        dimension=1,
        modulus=Q16_MODULUS,
        jump_warn_bp=10000,
        jump_block_bp=10000,
        persistent_warn_bp=1,
        persistent_block_bp=10000,
    ))

    first = guard.evaluate_transition(tv((0,)), tv((32768,)))
    second = guard.evaluate_transition(tv((32768,)), tv((32768,)))

    assert first.details["persistent_score_bp"] > 0
    assert second.details["persistent_score_bp"] > 0
    assert guard.persistent_anomaly_score_bp == second.details["persistent_score_bp"]


def test_topology_profile_rejects_invalid_thresholds():
    with pytest.raises(TopologyError):
        CalibrationProfile(dimension=0)
    with pytest.raises(TopologyError):
        CalibrationProfile(jump_warn_bp=9000, jump_block_bp=100)



def test_named_short_path_profile_does_not_punish_beta_mismatch():
    profile = profile_short_path(dimension=2, modulus=Q16_MODULUS)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0, 0)),
        tv((32, 0)),
        tv((64, 0)),
    ))

    decision = guard.evaluate_trajectory(trajectory, observed_beta1=0)

    assert decision.passed is True
    assert decision.details["beta_checked"] is False
    assert decision.details["profile_name"] == "short_path"


def test_named_clean_torus_profile_accepts_clean_loop():
    profile = profile_clean_torus(dimension=2, modulus=Q16_MODULUS)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0, 0)),
        tv((128, 0)),
        tv((128, 128)),
        tv((0, 128)),
        tv((0, 0)),
    ), max_points=10)

    decision = guard.evaluate_trajectory(trajectory, observed_beta1=2)

    assert decision.passed is True
    assert decision.action == TopologyAction.PASS
    assert decision.details["beta_checked"] is True
    assert decision.details["beta_mismatch"] is False
    assert decision.details["profile_name"] == "clean_torus"


def test_named_noisy_torus_profile_accepts_noisy_continuous_path():
    profile = profile_noisy_torus(dimension=2, modulus=Q16_MODULUS)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0, 0)),
        tv((512, 48)),
        tv((1024, 96)),
        tv((1536, 64)),
        tv((2048, 128)),
    ), max_points=10)

    decision = guard.evaluate_trajectory(trajectory)

    assert decision.passed is True
    assert decision.details["profile_name"] == "noisy_torus"
    assert decision.anomaly_score_bp < profile.jump_warn_bp


def test_named_partial_subtorus_profile_accepts_local_projection():
    profile = profile_partial_subtorus(dimension=3, modulus=Q16_MODULUS)
    guard = TopologyGuard(profile)
    trajectory = TrajectoryWindow((
        tv((0, 0, 0)),
        tv((64, 0, 0)),
        tv((128, 0, 0)),
        tv((192, 0, 0)),
    ), max_points=10)

    decision = guard.evaluate_trajectory(trajectory, observed_beta1=0)

    assert decision.passed is True
    assert decision.details["beta_checked"] is False
    assert decision.details["profile_name"] == "partial_subtorus"


def test_named_phase_shock_profile_blocks_large_jump():
    profile = profile_phase_shock(dimension=1, modulus=Q16_MODULUS)
    guard = TopologyGuard(profile)

    decision = guard.evaluate_transition(tv((0,)), tv((20000,)))

    assert decision.passed is False
    assert decision.action == TopologyAction.BLOCK
    assert decision.details["profile_name"] == "phase_shock"
