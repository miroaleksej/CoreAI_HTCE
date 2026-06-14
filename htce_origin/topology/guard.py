"""Q16 runtime topology guard for HTCE-Origin.

Scope boundary:
- Fast integer-only runtime guard for local trajectory health.
- Persistent anomaly score, local jump detector, phase-shock detector,
  trajectory continuity, and calibrated profile families.
- Does not run full persistent homology / Betti computation. Betti calibration
  remains offline/replay only.
- Does not require beta_1 == dimension for short live windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, TorusVector
from htce_origin.kernel.q16 import DEFAULT_MODULUS, Q16Error, q_distance, q_toroidal_loss_vector, q_vector_sub, q_distance_vector

MAX_BP = 10000


class TopologyError(ValueError):
    """Raised when topology guard inputs are outside the runtime contract."""


class TopologyProfileKind(str, Enum):
    """Named runtime calibration profile regimes."""

    CLEAN_TORUS = "clean_torus"
    NOISY_TORUS = "noisy_torus"
    PARTIAL_SUBTORUS = "partial_subtorus"
    SHORT_PATH = "short_path"
    PHASE_SHOCK = "phase_shock"


class TopologyAction(str, Enum):
    """Bounded topology decision classes."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


def _require_bp(value: int, name: str) -> int:
    bp = int(value)
    if not 0 <= bp <= MAX_BP:
        raise TopologyError(f"{name} must be in [0, 10000]")
    return bp


@dataclass(frozen=True)
class CalibrationProfile:
    """Integer calibration thresholds for the runtime topology guard.

    expected_beta1 is metadata for offline/replay calibration. The runtime guard
    never punishes a short live trajectory for beta1 != expected_beta1 unless the
    window has at least min_betti_points samples and an explicit observed_beta1
    is supplied by a calibration backend.
    """

    dimension: int = DEFAULT_TORUS_DIMENSION
    modulus: int = DEFAULT_MODULUS
    expected_beta1: int | None = None
    min_betti_points: int = 128
    jump_warn_bp: int = 3500
    jump_block_bp: int = 8000
    shock_warn_bp: int = 3500
    shock_block_bp: int = 8000
    persistent_warn_bp: int = 3000
    persistent_block_bp: int = 7000
    dirty_warning_count: int = 2
    smoothing_previous_weight: int = 3
    smoothing_denominator: int = 4
    profile_name: str = "default"

    @classmethod
    def profile_clean_torus(cls, *, dimension: int = 2, modulus: int = DEFAULT_MODULUS) -> "CalibrationProfile":
        """Profile for a clean calibrated torus replay cloud."""

        return cls(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=dimension,
            min_betti_points=4,
            jump_warn_bp=3500,
            jump_block_bp=8000,
            shock_warn_bp=3500,
            shock_block_bp=8000,
            persistent_warn_bp=3000,
            persistent_block_bp=7000,
            dirty_warning_count=2,
            profile_name=TopologyProfileKind.CLEAN_TORUS.value,
        )

    @classmethod
    def profile_noisy_torus(cls, *, dimension: int = 2, modulus: int = DEFAULT_MODULUS) -> "CalibrationProfile":
        """Profile for noisy but locally continuous torus replay clouds."""

        return cls(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=dimension,
            min_betti_points=8,
            jump_warn_bp=6500,
            jump_block_bp=9500,
            shock_warn_bp=6500,
            shock_block_bp=9500,
            persistent_warn_bp=6500,
            persistent_block_bp=9500,
            dirty_warning_count=4,
            profile_name=TopologyProfileKind.NOISY_TORUS.value,
        )

    @classmethod
    def profile_partial_subtorus(
        cls, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
    ) -> "CalibrationProfile":
        """Local subtorus profile that does not require full torus Betti."""

        return cls(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=None,
            min_betti_points=128,
            jump_warn_bp=4500,
            jump_block_bp=8500,
            shock_warn_bp=4500,
            shock_block_bp=8500,
            persistent_warn_bp=4500,
            persistent_block_bp=8500,
            dirty_warning_count=3,
            profile_name=TopologyProfileKind.PARTIAL_SUBTORUS.value,
        )

    @classmethod
    def profile_short_path(
        cls, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
    ) -> "CalibrationProfile":
        """Short live-window profile: continuity only, no Betti mismatch check."""

        return cls(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=None,
            min_betti_points=256,
            jump_warn_bp=3500,
            jump_block_bp=8000,
            shock_warn_bp=3500,
            shock_block_bp=8000,
            persistent_warn_bp=3000,
            persistent_block_bp=7000,
            dirty_warning_count=2,
            profile_name=TopologyProfileKind.SHORT_PATH.value,
        )

    @classmethod
    def profile_phase_shock(
        cls, *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
    ) -> "CalibrationProfile":
        """Strict profile for phase-shock detection calibration."""

        return cls(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=None,
            min_betti_points=128,
            jump_warn_bp=1000,
            jump_block_bp=2500,
            shock_warn_bp=1000,
            shock_block_bp=2500,
            persistent_warn_bp=1000,
            persistent_block_bp=3000,
            dirty_warning_count=1,
            profile_name=TopologyProfileKind.PHASE_SHOCK.value,
        )

    def __post_init__(self) -> None:
        if not self.profile_name:
            raise TopologyError("profile_name is required")
        if self.dimension <= 0:
            raise TopologyError("dimension must be positive")
        if self.modulus <= 0:
            raise Q16Error("modulus must be positive")
        if self.min_betti_points <= 0:
            raise TopologyError("min_betti_points must be positive")
        if self.expected_beta1 is not None and self.expected_beta1 < 0:
            raise TopologyError("expected_beta1 must be non-negative")
        for name in (
            "jump_warn_bp",
            "jump_block_bp",
            "shock_warn_bp",
            "shock_block_bp",
            "persistent_warn_bp",
            "persistent_block_bp",
        ):
            _require_bp(getattr(self, name), name)
        if self.jump_warn_bp > self.jump_block_bp:
            raise TopologyError("jump_warn_bp must be <= jump_block_bp")
        if self.shock_warn_bp > self.shock_block_bp:
            raise TopologyError("shock_warn_bp must be <= shock_block_bp")
        if self.persistent_warn_bp > self.persistent_block_bp:
            raise TopologyError("persistent_warn_bp must be <= persistent_block_bp")
        if self.dirty_warning_count <= 0:
            raise TopologyError("dirty_warning_count must be positive")
        if not 0 <= self.smoothing_previous_weight <= self.smoothing_denominator:
            raise TopologyError("invalid smoothing weights")
        if self.smoothing_denominator <= 0:
            raise TopologyError("smoothing_denominator must be positive")


@dataclass(frozen=True)
class TopologyDecision:
    """Topology guard decision for a transition or trajectory window."""

    passed: bool
    anomaly_score_bp: int
    reason: str
    action: TopologyAction = TopologyAction.PASS
    warnings: tuple[str, ...] = ()
    details: Mapping[str, int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_bp(self.anomaly_score_bp, "anomaly_score_bp")


@dataclass(frozen=True)
class TrajectoryWindow:
    """Immutable short runtime trajectory window."""

    states: tuple[TorusVector, ...] = ()
    max_points: int = 64

    def __post_init__(self) -> None:
        if self.max_points <= 0:
            raise TopologyError("max_points must be positive")
        if len(self.states) > self.max_points:
            object.__setattr__(self, "states", self.states[-self.max_points :])
        if self.states:
            dimension = self.states[0].dimension
            modulus = self.states[0].modulus
            for state in self.states:
                if state.dimension != dimension:
                    raise TopologyError("all trajectory states must have the same dimension")
                if state.modulus != modulus:
                    raise TopologyError("all trajectory states must have the same modulus")

    def append(self, state: TorusVector | Iterable[int], *, modulus: int | None = None) -> "TrajectoryWindow":
        if isinstance(state, TorusVector):
            item = state
        else:
            if self.states:
                item = TorusVector(tuple(state), self.states[0].modulus)
            else:
                item = TorusVector(tuple(state), modulus or DEFAULT_MODULUS)
        return TrajectoryWindow(self.states + (item,), self.max_points)

    @property
    def count(self) -> int:
        return len(self.states)


@dataclass(frozen=True)
class TransitionMetrics:
    """Integer metrics for one local transition."""

    jump_bp: int
    shock_bp: int
    persistent_score_bp: int
    jump_distance_sq: int
    shock_loss: int


def _vr_1_skeleton_betti(
    states: tuple[TorusVector, ...],
    *,
    epsilon: int,
) -> tuple[int, int, int]:
    """Return integer (beta0, beta1, edge_count) for a VR 1-skeleton.

    This is a bounded runtime diagnostic: beta1 is the graph cyclomatic
    number E - V + C of the Vietoris-Rips 1-skeleton at integer radius
    ``epsilon``.  It is not a full persistent-homology computation and does not
    create beta2.
    """
    if not states:
        return 0, 0, 0

    parent = list(range(len(states)))
    rank = [0 for _ in states]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    edge_count = 0
    dimension_factor = max(1, states[0].dimension)
    threshold_sq = int(epsilon) * int(epsilon) * dimension_factor
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            if q_distance_vector(states[left].phases, states[right].phases, states[left].modulus) <= threshold_sq:
                edge_count += 1
                union(left, right)

    beta0 = len({find(index) for index in range(len(states))})
    beta1 = max(0, edge_count - len(states) + beta0)
    return int(beta0), int(beta1), int(edge_count)


def profile_clean_torus(*, dimension: int = 2, modulus: int = DEFAULT_MODULUS) -> CalibrationProfile:
    """Return the named clean-torus runtime calibration profile."""

    return CalibrationProfile.profile_clean_torus(dimension=dimension, modulus=modulus)


def profile_noisy_torus(*, dimension: int = 2, modulus: int = DEFAULT_MODULUS) -> CalibrationProfile:
    """Return the named noisy-torus runtime calibration profile."""

    return CalibrationProfile.profile_noisy_torus(dimension=dimension, modulus=modulus)


def profile_partial_subtorus(
    *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
) -> CalibrationProfile:
    """Return the named partial-subtorus local calibration profile."""

    return CalibrationProfile.profile_partial_subtorus(dimension=dimension, modulus=modulus)


def profile_short_path(
    *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
) -> CalibrationProfile:
    """Return the named short-path calibration profile."""

    return CalibrationProfile.profile_short_path(dimension=dimension, modulus=modulus)


def profile_phase_shock(
    *, dimension: int = DEFAULT_TORUS_DIMENSION, modulus: int = DEFAULT_MODULUS
) -> CalibrationProfile:
    """Return the named phase-shock calibration profile."""

    return CalibrationProfile.profile_phase_shock(dimension=dimension, modulus=modulus)


class TopologyGuard:
    """Fast integer-only runtime topology guard."""

    def __init__(self, profile: CalibrationProfile | None = None) -> None:
        self.profile = profile or CalibrationProfile()
        self.persistent_anomaly_score_bp = 0
        self._previous_delta: tuple[int, ...] | None = None

    def reset(self) -> None:
        self.persistent_anomaly_score_bp = 0
        self._previous_delta = None

    def _state(self, state: TorusVector | Iterable[int]) -> TorusVector:
        if isinstance(state, TorusVector):
            result = state
        else:
            result = TorusVector(tuple(state), self.profile.modulus)
        if result.dimension != self.profile.dimension:
            raise TopologyError("state dimension does not match calibration profile")
        if result.modulus != self.profile.modulus:
            raise TopologyError("state modulus does not match calibration profile")
        return result

    def _jump_bp(self, previous: TorusVector, current: TorusVector) -> tuple[int, int]:
        distance_sq = q_distance_vector(previous.phases, current.phases, self.profile.modulus)
        half = self.profile.modulus // 2
        max_distance_sq = self.profile.dimension * half * half
        jump_bp = (distance_sq * MAX_BP) // max_distance_sq if max_distance_sq else MAX_BP
        return min(MAX_BP, jump_bp), distance_sq

    def _shock_bp(self, delta: tuple[int, ...]) -> tuple[int, int]:
        if self._previous_delta is None:
            return 0, 0
        loss = q_toroidal_loss_vector(delta, self._previous_delta, self.profile.modulus)
        max_loss = self.profile.dimension * 65535
        shock_bp = (loss * MAX_BP) // max_loss if max_loss else MAX_BP
        return min(MAX_BP, shock_bp), loss

    def _update_persistent_score(self, raw_score_bp: int) -> int:
        raw = _require_bp(raw_score_bp, "raw_score_bp")
        profile = self.profile
        weighted = (
            self.persistent_anomaly_score_bp * profile.smoothing_previous_weight
            + raw * (profile.smoothing_denominator - profile.smoothing_previous_weight)
        ) // profile.smoothing_denominator
        self.persistent_anomaly_score_bp = min(MAX_BP, weighted)
        return self.persistent_anomaly_score_bp

    def live_vr_betti_1_skeleton(self, trajectory: TrajectoryWindow | Iterable[TorusVector | Iterable[int]]) -> dict[str, int]:
        """Compute integer live-window VR 1-skeleton Betti diagnostics.

        The radius is derived from the guard profile and remains integer-only.
        For short live windows the result is diagnostic/provenance data; only
        explicitly calibrated large windows may be blocked on beta mismatch.
        """
        if isinstance(trajectory, TrajectoryWindow):
            window = trajectory
        else:
            states = tuple(self._state(item) for item in trajectory)
            window = TrajectoryWindow(states=states)
        epsilon = max(1, self.profile.modulus // 16)
        beta0, beta1, edge_count = _vr_1_skeleton_betti(window.states, epsilon=epsilon)
        return {
            "live_beta0": beta0,
            "live_beta1": beta1,
            "live_vr_edge_count": edge_count,
            "live_vr_epsilon": epsilon,
        }

    def evaluate_transition(
        self,
        previous_state: TorusVector | Iterable[int],
        current_state: TorusVector | Iterable[int],
    ) -> TopologyDecision:
        """Evaluate a single transition and update persistent anomaly score."""

        previous = self._state(previous_state)
        current = self._state(current_state)
        delta = q_vector_sub(current.phases, previous.phases, self.profile.modulus)
        jump_bp, distance_sq = self._jump_bp(previous, current)
        shock_bp, shock_loss = self._shock_bp(delta)
        raw_score = max(jump_bp, shock_bp)
        persistent = self._update_persistent_score(raw_score)
        self._previous_delta = delta

        warnings: list[str] = []
        action = TopologyAction.PASS
        passed = True
        reason = "topology transition passed"

        if jump_bp >= self.profile.jump_block_bp:
            warnings.append("phase_jump_block")
            action = TopologyAction.BLOCK
            passed = False
            reason = "phase shock/jump blocked by local jump detector"
        elif shock_bp >= self.profile.shock_block_bp:
            warnings.append("phase_shock_block")
            action = TopologyAction.BLOCK
            passed = False
            reason = "phase shock blocked by delta-continuity detector"
        elif persistent >= self.profile.persistent_block_bp:
            warnings.append("persistent_anomaly_block")
            action = TopologyAction.BLOCK
            passed = False
            reason = "persistent anomaly score blocked transition"
        elif jump_bp >= self.profile.jump_warn_bp or shock_bp >= self.profile.shock_warn_bp or persistent >= self.profile.persistent_warn_bp:
            if jump_bp >= self.profile.jump_warn_bp:
                warnings.append("phase_jump_warn")
            if shock_bp >= self.profile.shock_warn_bp:
                warnings.append("phase_shock_warn")
            if persistent >= self.profile.persistent_warn_bp:
                warnings.append("persistent_anomaly_warn")
            action = TopologyAction.WARN
            reason = "topology transition warned but did not block"

        return TopologyDecision(
            passed=passed,
            anomaly_score_bp=max(raw_score, persistent),
            reason=reason,
            action=action,
            warnings=tuple(warnings),
            details={
                "jump_bp": jump_bp,
                "shock_bp": shock_bp,
                "persistent_score_bp": persistent,
                "jump_distance_sq": distance_sq,
                "shock_loss": shock_loss,
                "profile_name": self.profile.profile_name,
            },
        )

    def evaluate_trajectory(
        self,
        trajectory: TrajectoryWindow | Iterable[TorusVector | Iterable[int]],
        *,
        observed_beta1: int | None = None,
    ) -> TopologyDecision:
        """Evaluate a short runtime trajectory window."""

        if isinstance(trajectory, TrajectoryWindow):
            window = trajectory
        else:
            states = tuple(self._state(item) for item in trajectory)
            window = TrajectoryWindow(states=states)
        if window.count < 2:
            return TopologyDecision(
                True,
                0,
                "trajectory too short; no topology anomaly",
                details={"state_count": window.count, "profile_name": self.profile.profile_name},
            )

        self.reset()
        warning_count = 0
        block_count = 0
        max_score = 0
        for previous, current in zip(window.states, window.states[1:]):
            decision = self.evaluate_transition(previous, current)
            max_score = max(max_score, decision.anomaly_score_bp)
            if decision.action == TopologyAction.WARN:
                warning_count += 1
            if decision.action == TopologyAction.BLOCK:
                block_count += 1

        warnings: list[str] = []
        if warning_count:
            warnings.append("trajectory_warning")
        if block_count:
            warnings.append("trajectory_block")

        live_betti = self.live_vr_betti_1_skeleton(window)
        beta_checked = False
        beta_mismatch = False
        beta1_for_check = observed_beta1 if observed_beta1 is not None else live_betti["live_beta1"]
        if self.profile.expected_beta1 is not None and window.count >= self.profile.min_betti_points:
            beta_checked = True
            beta_mismatch = int(beta1_for_check) != self.profile.expected_beta1
            if beta_mismatch:
                warnings.append("calibrated_beta1_mismatch")
                max_score = max(max_score, self.profile.persistent_block_bp)

        details = {
            "state_count": window.count,
            "profile_name": self.profile.profile_name,
            "warning_count": warning_count,
            "block_count": block_count,
            "beta_checked": beta_checked,
            "beta_mismatch": beta_mismatch,
            **live_betti,
        }

        if block_count:
            return TopologyDecision(
                False,
                max_score,
                "trajectory blocked by phase shock or local jump detector",
                action=TopologyAction.BLOCK,
                warnings=tuple(warnings),
                details=details,
            )
        if beta_mismatch:
            return TopologyDecision(
                False,
                max_score,
                "calibrated topology mismatch on sufficiently large replay window",
                action=TopologyAction.BLOCK,
                warnings=tuple(warnings),
                details=details,
            )
        if warning_count >= self.profile.dirty_warning_count:
            return TopologyDecision(
                False,
                max_score,
                "dirty trajectory marked anomalous by repeated topology warnings",
                action=TopologyAction.BLOCK,
                warnings=tuple(warnings + ["dirty_trajectory"]),
                details=details,
            )
        if warning_count:
            return TopologyDecision(
                True,
                max_score,
                "trajectory warned but remains within runtime topology boundary",
                action=TopologyAction.WARN,
                warnings=tuple(warnings),
                details=details,
            )
        return TopologyDecision(
            True,
            max_score,
            "trajectory passed runtime topology guard",
            action=TopologyAction.PASS,
            details=details,
        )

    def evaluate(self, trajectory: object) -> TopologyDecision:
        if isinstance(trajectory, TrajectoryWindow):
            return self.evaluate_trajectory(trajectory)
        if isinstance(trajectory, Iterable):
            return self.evaluate_trajectory(trajectory)
        raise TopologyError("trajectory must be a TrajectoryWindow or iterable of states")


@dataclass(frozen=True)
class L3TopologyValidationReport:
    """Integer 1-skeleton Betti diagnostic for consolidated L3 state clouds."""

    beta0: int
    beta1: int
    point_count: int
    edge_count: int
    passed: bool
    reason: str
    expected_beta1: int | None = None

    def __post_init__(self) -> None:
        if self.beta0 < 0 or self.beta1 < 0 or self.point_count < 0 or self.edge_count < 0:
            raise TopologyError("L3 topology report counters must be non-negative")


def l3_betti_1_skeleton(
    states: Iterable[Iterable[int]],
    *,
    radius: int | None = None,
    modulus: int = DEFAULT_MODULUS,
) -> tuple[int, int, int, int]:
    """Compute integer (beta0, beta1, point_count, edge_count) for L3 states.

    This is a dependency-free Vietoris-Rips 1-skeleton diagnostic.  It is a
    lower-order runtime/offline guard, not a full persistent-homology backend.
    """
    points = tuple(tuple(int(value) % int(modulus) for value in state) for state in states)
    if not points:
        return 0, 0, 0, 0
    dimension = len(points[0])
    if dimension <= 0:
        raise TopologyError("L3 topology states must be non-empty vectors")
    for point in points:
        if len(point) != dimension:
            raise TopologyError("L3 topology states must share one dimension")
    n = len(points)
    parent = list(range(n))
    rank = [0 for _ in range(n)]
    components = n

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        nonlocal components
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1
        components -= 1
        return True

    eps = int(radius) if radius is not None else max(1, int(modulus) // 64)
    radius_sq = eps * eps
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0
            for coord in range(dimension):
                dist = q_distance(points[i][coord], points[j][coord], modulus)
                dist_sq += dist * dist
                if dist_sq > radius_sq:
                    break
            if dist_sq <= radius_sq:
                edge_count += 1
                union(i, j)
    beta0 = components
    beta1 = max(0, edge_count - n + beta0)
    return beta0, beta1, n, edge_count


def validate_l3_semantic_window(
    states: Iterable[Iterable[int]],
    *,
    radius: int | None = None,
    expected_beta1: int | None = 0,
    beta1_tolerance: int = 1,
    modulus: int = DEFAULT_MODULUS,
) -> L3TopologyValidationReport:
    """Validate that consolidated L3 states remain one coherent component."""
    beta0, beta1, point_count, edge_count = l3_betti_1_skeleton(states, radius=radius, modulus=modulus)
    if point_count < 4:
        return L3TopologyValidationReport(beta0, beta1, point_count, edge_count, True, "insufficient L3 points for topology block", expected_beta1)
    if beta0 != 1:
        return L3TopologyValidationReport(beta0, beta1, point_count, edge_count, False, "L3 topology fractured: beta0 != 1", expected_beta1)
    if expected_beta1 is not None and abs(beta1 - int(expected_beta1)) > int(beta1_tolerance):
        return L3TopologyValidationReport(beta0, beta1, point_count, edge_count, False, "L3 anomalous cycle count", expected_beta1)
    return L3TopologyValidationReport(beta0, beta1, point_count, edge_count, True, "L3 topology valid", expected_beta1)


__all__ = [
    "CalibrationProfile",
    "TopologyAction",
    "TopologyDecision",
    "TopologyError",
    "L3TopologyValidationReport",
    "TopologyGuard",
    "TopologyProfileKind",
    "TrajectoryWindow",
    "TransitionMetrics",
    "l3_betti_1_skeleton",
    "profile_clean_torus",
    "validate_l3_semantic_window",
    "profile_noisy_torus",
    "profile_partial_subtorus",
    "profile_phase_shock",
    "profile_short_path",
]
