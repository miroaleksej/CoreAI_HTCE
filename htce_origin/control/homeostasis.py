"""homeostasis/control layer homeostatic body state and active-inference surrogate.

Scope boundary:
- Represents bounded internal control variables as integer basis points.
- Computes setpoint deviation, viability, and an expected-free-energy surrogate.
- Applies deterministic action effects and deterministic domain-noise perturbations.
- Selects clarification/sleep/risk/exploration control recommendations.
- Does not mutate L1/L2/L3 memory, does not authorize real actions, and does not
  claim biological homeostasis or literal free-energy minimization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_distance, q_mod

MAX_BP = 10000
HOMEOSTATIC_KEYS = (
    "energy_bp",
    "sleep_pressure_bp",
    "risk_bp",
    "integrity_bp",
    "novelty_bp",
    "uncertainty_bp",
)


class HomeostasisError(ValueError):
    """Raised when homeostatic values violate bounded integer contracts."""


class ControlSignal(str, Enum):
    """Bounded control recommendation emitted by the homeostasis layer."""

    CONTINUE = "continue"
    ASK_CLARIFICATION = "ask_clarification"
    SLEEP_REQUIRED = "sleep_required"
    BLOCK_SIMULATED_ACTION = "block_simulated_action"
    EXPLORE_SIMULATION = "explore_simulation"


def clamp_bp(value: int) -> int:
    """Clamp an integer score into basis points [0, 10000]."""
    return max(0, min(MAX_BP, int(value)))


def signed_clamp_bp(value: int, magnitude_bp: int) -> int:
    """Clamp a signed integer into [-magnitude_bp, +magnitude_bp]."""
    magnitude = require_bp(magnitude_bp, "magnitude_bp")
    return max(-magnitude, min(magnitude, int(value)))


def require_bp(value: int, name: str) -> int:
    """Validate a basis-point value."""
    result = int(value)
    if not 0 <= result <= MAX_BP:
        raise HomeostasisError(f"{name} must be in [0, 10000]")
    return result


def require_non_negative_int(value: int, name: str) -> int:
    """Validate a non-negative integer value."""
    result = int(value)
    if result < 0:
        raise HomeostasisError(f"{name} must be non-negative")
    return result


def _signed_noise_from_seed(seed: str, label: str, magnitude_bp: int) -> int:
    """Return deterministic signed noise from a seed and label.

    This is a simulation helper, not stochastic runtime entropy. It makes domain
    randomisation reproducible and traceable.
    """
    magnitude = require_bp(magnitude_bp, "magnitude_bp")
    if magnitude == 0:
        return 0
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    span = 2 * magnitude + 1
    return int.from_bytes(digest[:8], "big") % span - magnitude


@dataclass(frozen=True)
class HomeostaticState:
    """Bounded integer body-state surrogate.

    All fields are basis points. This is a controller state, not a biological
    claim. High pressure values should narrow the allowed policy surface.

    B_t = (energy, sleep_pressure, risk, integrity, novelty, uncertainty).
    """

    energy_bp: int = MAX_BP
    sleep_pressure_bp: int = 0
    risk_bp: int = 0
    integrity_bp: int = MAX_BP
    novelty_bp: int = 0
    uncertainty_bp: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_bp", require_bp(self.energy_bp, "energy_bp"))
        object.__setattr__(self, "sleep_pressure_bp", require_bp(self.sleep_pressure_bp, "sleep_pressure_bp"))
        object.__setattr__(self, "risk_bp", require_bp(self.risk_bp, "risk_bp"))
        object.__setattr__(self, "integrity_bp", require_bp(self.integrity_bp, "integrity_bp"))
        object.__setattr__(self, "novelty_bp", require_bp(self.novelty_bp, "novelty_bp"))
        object.__setattr__(self, "uncertainty_bp", require_bp(self.uncertainty_bp, "uncertainty_bp"))

    def as_vector(self) -> tuple[int, int, int, int, int, int]:
        """Return B_t in canonical coordinate order."""
        return (
            self.energy_bp,
            self.sleep_pressure_bp,
            self.risk_bp,
            self.integrity_bp,
            self.novelty_bp,
            self.uncertainty_bp,
        )

    def as_mapping(self) -> dict[str, int]:
        """Return B_t as a canonical mapping."""
        return {key: value for key, value in zip(HOMEOSTATIC_KEYS, self.as_vector())}

    def viability_bp(self) -> int:
        """Return bounded legacy viability from support minus active pressures.

        Kept for compatibility with previous tests and release reports. The
        richer setpoint-aware viability is exposed by ActiveInferenceSurrogate.
        """
        support = (self.energy_bp + self.integrity_bp) // 2
        pressure = max(self.sleep_pressure_bp, self.risk_bp, self.uncertainty_bp)
        return clamp_bp(support - (pressure // 2))

    def with_delta(
        self,
        *,
        energy_delta_bp: int = 0,
        sleep_pressure_delta_bp: int = 0,
        risk_delta_bp: int = 0,
        integrity_delta_bp: int = 0,
        novelty_delta_bp: int = 0,
        uncertainty_delta_bp: int = 0,
    ) -> "HomeostaticState":
        """Return a new bounded state after signed integer deltas."""
        return HomeostaticState(
            energy_bp=clamp_bp(self.energy_bp + int(energy_delta_bp)),
            sleep_pressure_bp=clamp_bp(self.sleep_pressure_bp + int(sleep_pressure_delta_bp)),
            risk_bp=clamp_bp(self.risk_bp + int(risk_delta_bp)),
            integrity_bp=clamp_bp(self.integrity_bp + int(integrity_delta_bp)),
            novelty_bp=clamp_bp(self.novelty_bp + int(novelty_delta_bp)),
            uncertainty_bp=clamp_bp(self.uncertainty_bp + int(uncertainty_delta_bp)),
        )

    def with_observation(
        self,
        *,
        uncertainty_delta_bp: int = 0,
        novelty_delta_bp: int = 0,
        risk_delta_bp: int = 0,
        sleep_pressure_delta_bp: int = 0,
    ) -> "HomeostaticState":
        """Return a new bounded state after a simulated observation update."""
        return self.with_delta(
            uncertainty_delta_bp=uncertainty_delta_bp,
            novelty_delta_bp=novelty_delta_bp,
            risk_delta_bp=risk_delta_bp,
            sleep_pressure_delta_bp=sleep_pressure_delta_bp,
        )


@dataclass(frozen=True)
class HomeostaticSetpoint:
    """Target homeostatic vector B* in basis points.

    Pressure coordinates such as risk/sleep/uncertainty normally target zero;
    support coordinates such as energy/integrity normally target MAX_BP.
    """

    energy_bp: int = MAX_BP
    sleep_pressure_bp: int = 0
    risk_bp: int = 0
    integrity_bp: int = MAX_BP
    novelty_bp: int = 0
    uncertainty_bp: int = 0

    def __post_init__(self) -> None:
        for key in HOMEOSTATIC_KEYS:
            object.__setattr__(self, key, require_bp(getattr(self, key), key))

    def as_vector(self) -> tuple[int, int, int, int, int, int]:
        """Return B* in canonical coordinate order."""
        return (
            self.energy_bp,
            self.sleep_pressure_bp,
            self.risk_bp,
            self.integrity_bp,
            self.novelty_bp,
            self.uncertainty_bp,
        )


@dataclass(frozen=True)
class HomeostaticWeights:
    """Non-negative integer weights for setpoint deviation.

    The default weights sum to 100, so the weighted deviation is naturally in
    basis points.
    """

    energy_bp: int = 18
    sleep_pressure_bp: int = 16
    risk_bp: int = 22
    integrity_bp: int = 18
    novelty_bp: int = 8
    uncertainty_bp: int = 18

    def __post_init__(self) -> None:
        for key in HOMEOSTATIC_KEYS:
            object.__setattr__(self, key, require_non_negative_int(getattr(self, key), key))
        if self.total_weight() <= 0:
            raise HomeostasisError("homeostatic weights must have positive sum")

    def as_vector(self) -> tuple[int, int, int, int, int, int]:
        """Return weights in canonical coordinate order."""
        return (
            self.energy_bp,
            self.sleep_pressure_bp,
            self.risk_bp,
            self.integrity_bp,
            self.novelty_bp,
            self.uncertainty_bp,
        )

    def total_weight(self) -> int:
        """Return sum of all weights."""
        return sum(self.as_vector())


@dataclass(frozen=True)
class HomeostaticCalibrationReport:
    """Calibration report for integer homeostatic setpoints and weights."""

    setpoint: HomeostaticSetpoint
    weights: HomeostaticWeights
    safe_viability_min_bp: int
    unsafe_viability_max_bp: int
    uncertain_viability_max_bp: int
    uncertain_signal: ControlSignal
    safe_count: int
    unsafe_count: int
    uncertainty_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_viability_min_bp", require_bp(self.safe_viability_min_bp, "safe_viability_min_bp"))
        object.__setattr__(self, "unsafe_viability_max_bp", require_bp(self.unsafe_viability_max_bp, "unsafe_viability_max_bp"))
        object.__setattr__(self, "uncertain_viability_max_bp", require_bp(self.uncertain_viability_max_bp, "uncertain_viability_max_bp"))
        if self.safe_count <= 0 or self.unsafe_count <= 0 or self.uncertainty_count <= 0:
            raise HomeostasisError("calibration replay groups must be non-empty")

    def surrogate(self, config: ActiveInferenceConfig | None = None) -> "ActiveInferenceSurrogate":
        """Build a surrogate with calibrated setpoint and weights."""
        return ActiveInferenceSurrogate(config=config, setpoint=self.setpoint, weights=self.weights)


def _mean_bp(values: tuple[int, ...]) -> int:
    if not values:
        raise HomeostasisError("cannot compute mean of empty replay")
    return sum(values) // len(values)


def _mean_state(states: tuple[HomeostaticState, ...]) -> HomeostaticState:
    if not states:
        raise HomeostasisError("homeostasis calibration replays must be non-empty")
    columns = tuple(zip(*(state.as_vector() for state in states)))
    return HomeostaticState(*(_mean_bp(tuple(column)) for column in columns))


def _calibrated_weight(safe_value: int, unsafe_value: int, uncertain_value: int, base: int) -> int:
    return max(1, int(base) + (abs(int(unsafe_value) - int(safe_value)) // 100) + (abs(int(uncertain_value) - int(safe_value)) // 200))


def calibrate_homeostasis_weights(
    safe_replay: Iterable[HomeostaticState],
    unsafe_replay: Iterable[HomeostaticState],
    uncertainty_replay: Iterable[HomeostaticState],
) -> HomeostaticCalibrationReport:
    """Calibrate B* and integer weights from safe/unsafe/uncertain replay.

    The function is intentionally bounded and deterministic. It does not learn a
    neural model and does not authorize actions; it only returns a calibrated
    ActiveInferenceSurrogate configuration for release-gate evaluation.
    """
    safe = tuple(safe_replay)
    unsafe = tuple(unsafe_replay)
    uncertain = tuple(uncertainty_replay)
    safe_mean = _mean_state(safe)
    unsafe_mean = _mean_state(unsafe)
    uncertain_mean = _mean_state(uncertain)
    setpoint = HomeostaticSetpoint(*safe_mean.as_vector())
    weights = HomeostaticWeights(
        energy_bp=_calibrated_weight(safe_mean.energy_bp, unsafe_mean.energy_bp, uncertain_mean.energy_bp, 12),
        sleep_pressure_bp=_calibrated_weight(safe_mean.sleep_pressure_bp, unsafe_mean.sleep_pressure_bp, uncertain_mean.sleep_pressure_bp, 14),
        risk_bp=_calibrated_weight(safe_mean.risk_bp, unsafe_mean.risk_bp, uncertain_mean.risk_bp, 30),
        integrity_bp=_calibrated_weight(safe_mean.integrity_bp, unsafe_mean.integrity_bp, uncertain_mean.integrity_bp, 12),
        novelty_bp=_calibrated_weight(safe_mean.novelty_bp, unsafe_mean.novelty_bp, uncertain_mean.novelty_bp, 6),
        uncertainty_bp=_calibrated_weight(safe_mean.uncertainty_bp, unsafe_mean.uncertainty_bp, uncertain_mean.uncertainty_bp, 30),
    )
    surrogate = ActiveInferenceSurrogate(setpoint=setpoint, weights=weights)
    safe_viabilities = tuple(surrogate.viability_bp(state) for state in safe)
    unsafe_viabilities = tuple(surrogate.viability_bp(state) for state in unsafe)
    uncertain_evaluations = tuple(surrogate.evaluate(state) for state in uncertain)
    uncertain_viabilities = tuple(row.viability_bp for row in uncertain_evaluations)
    # If any uncertain replay crosses the clarification threshold, the calibrated
    # policy class is clarification/simulation rather than action.
    uncertain_signal = ControlSignal.ASK_CLARIFICATION if any(row.signal == ControlSignal.ASK_CLARIFICATION for row in uncertain_evaluations) else uncertain_evaluations[0].signal
    return HomeostaticCalibrationReport(
        setpoint=setpoint,
        weights=weights,
        safe_viability_min_bp=min(safe_viabilities),
        unsafe_viability_max_bp=max(unsafe_viabilities),
        uncertain_viability_max_bp=max(uncertain_viabilities),
        uncertain_signal=uncertain_signal,
        safe_count=len(safe),
        unsafe_count=len(unsafe),
        uncertainty_count=len(uncertain),
    )


@dataclass(frozen=True)
class DomainNoise:
    """Deterministic signed domain perturbation for simulation planning.

    This is not random runtime mutation. It is a reproducible, seed-derived
    perturbation used to test homeostatic robustness under simulated domains.
    """

    energy_noise_bp: int = 0
    sleep_pressure_noise_bp: int = 0
    risk_noise_bp: int = 0
    integrity_noise_bp: int = 0
    novelty_noise_bp: int = 0
    uncertainty_noise_bp: int = 0

    @staticmethod
    def from_seed(seed: str, magnitude_bp: int = 250) -> "DomainNoise":
        """Create deterministic signed domain noise from a seed."""
        return DomainNoise(
            energy_noise_bp=_signed_noise_from_seed(seed, "energy", magnitude_bp),
            sleep_pressure_noise_bp=_signed_noise_from_seed(seed, "sleep_pressure", magnitude_bp),
            risk_noise_bp=_signed_noise_from_seed(seed, "risk", magnitude_bp),
            integrity_noise_bp=_signed_noise_from_seed(seed, "integrity", magnitude_bp),
            novelty_noise_bp=_signed_noise_from_seed(seed, "novelty", magnitude_bp),
            uncertainty_noise_bp=_signed_noise_from_seed(seed, "uncertainty", magnitude_bp),
        )

    def as_delta_kwargs(self) -> dict[str, int]:
        """Return noise as HomeostaticState.with_delta keyword arguments."""
        return {
            "energy_delta_bp": self.energy_noise_bp,
            "sleep_pressure_delta_bp": self.sleep_pressure_noise_bp,
            "risk_delta_bp": self.risk_noise_bp,
            "integrity_delta_bp": self.integrity_noise_bp,
            "novelty_delta_bp": self.novelty_noise_bp,
            "uncertainty_delta_bp": self.uncertainty_noise_bp,
        }


@dataclass(frozen=True)
class HomeostaticActionEffect:
    """Deterministic action-effect vector for simulated control.

    Positive pressure deltas increase stress; negative pressure deltas reduce
    stress. The metadata fields are used by the expected-free-energy surrogate.
    """

    energy_delta_bp: int = 0
    sleep_pressure_delta_bp: int = 0
    risk_delta_bp: int = 0
    integrity_delta_bp: int = 0
    novelty_delta_bp: int = 0
    uncertainty_delta_bp: int = 0
    model_error_bp: int = 0
    evidence_gap_bp: int = 0
    complexity_bp: int = 0
    goal_progress_bp: int = 0
    novelty_gain_bp: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_error_bp", require_bp(self.model_error_bp, "model_error_bp"))
        object.__setattr__(self, "evidence_gap_bp", require_bp(self.evidence_gap_bp, "evidence_gap_bp"))
        object.__setattr__(self, "complexity_bp", require_bp(self.complexity_bp, "complexity_bp"))
        object.__setattr__(self, "goal_progress_bp", require_bp(self.goal_progress_bp, "goal_progress_bp"))
        object.__setattr__(self, "novelty_gain_bp", require_bp(self.novelty_gain_bp, "novelty_gain_bp"))

    @staticmethod
    def sleep_cycle(recovery_bp: int = 4000) -> "HomeostaticActionEffect":
        """Return a deterministic sleep effect reducing sleep pressure."""
        recovery = require_bp(recovery_bp, "recovery_bp")
        return HomeostaticActionEffect(
            sleep_pressure_delta_bp=-recovery,
            uncertainty_delta_bp=-(recovery // 4),
            energy_delta_bp=recovery // 3,
            complexity_bp=500,
            goal_progress_bp=1000,
        )

    def apply(self, state: HomeostaticState, domain_noise: DomainNoise | None = None) -> HomeostaticState:
        """Apply deterministic effect plus optional deterministic domain noise."""
        noise = domain_noise.as_delta_kwargs() if domain_noise else {}
        return state.with_delta(
            energy_delta_bp=self.energy_delta_bp + noise.get("energy_delta_bp", 0),
            sleep_pressure_delta_bp=self.sleep_pressure_delta_bp + noise.get("sleep_pressure_delta_bp", 0),
            risk_delta_bp=self.risk_delta_bp + noise.get("risk_delta_bp", 0),
            integrity_delta_bp=self.integrity_delta_bp + noise.get("integrity_delta_bp", 0),
            novelty_delta_bp=self.novelty_delta_bp + noise.get("novelty_delta_bp", 0),
            uncertainty_delta_bp=self.uncertainty_delta_bp + noise.get("uncertainty_delta_bp", 0),
        )



@dataclass(frozen=True)
class SensoryObservation:
    """Simulated sensory observation used for L1 intake and curiosity scoring.

    o_t = (modality, value, intensity_bp, reliability_bp, phase).

    The object is explicitly simulation-bound in v0.1: it may update L1 via
    ``L123Body.observe_simulated`` and may drive curiosity scoring, but it may
    not commit facts to L2/L3 and may not represent a real sensor authority.
    """

    modality: str
    value: str
    intensity_bp: int
    reliability_bp: int
    phase: tuple[int, ...]
    evidence_id: str = "simulated_observation"
    simulated: bool = True
    real_sensor_commit_allowed: bool = False
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        if not self.modality:
            raise HomeostasisError("observation modality must be non-empty")
        if not self.value:
            raise HomeostasisError("observation value must be non-empty")
        object.__setattr__(self, "intensity_bp", require_bp(self.intensity_bp, "intensity_bp"))
        object.__setattr__(self, "reliability_bp", require_bp(self.reliability_bp, "reliability_bp"))
        object.__setattr__(self, "phase", tuple(q_mod(value, self.modulus) for value in self.phase))
        if not self.phase:
            raise HomeostasisError("observation phase must be non-empty")
        if self.real_sensor_commit_allowed:
            raise HomeostasisError("real_sensor_commit_allowed must remain false in v0.1")

    def public_payload(self) -> dict[str, object]:
        """Return a safe payload without any authority to commit facts."""
        return {
            "evidence_id": self.evidence_id,
            "intensity_bp": self.intensity_bp,
            "modality": self.modality,
            "phase": self.phase,
            "real_sensor_commit_allowed": False,
            "reliability_bp": self.reliability_bp,
            "simulated": self.simulated,
            "value": self.value,
        }


@dataclass(frozen=True)
class CuriositySignal:
    """Bounded curiosity signal emitted from prediction error and novelty.

    curiosity_bp = a*prediction_error + b*novelty + c*reliability - d*risk,
    normalised by the positive weights and clamped to [0,10000].
    """

    prediction_error_bp: int
    novelty_bp: int
    reliability_bp: int
    risk_bp: int
    curiosity_bp: int
    explore_simulation: bool
    reason: str
    real_sensor_commit_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("prediction_error_bp", "novelty_bp", "reliability_bp", "risk_bp", "curiosity_bp"):
            object.__setattr__(self, name, require_bp(getattr(self, name), name))
        if self.real_sensor_commit_allowed:
            raise HomeostasisError("curiosity cannot authorize real sensor commits")


@dataclass(frozen=True)
class CuriosityDriveConfig:
    """Integer coefficients and thresholds for simulated curiosity."""

    prediction_error_weight: int = 40
    novelty_weight: int = 25
    reliability_weight: int = 20
    risk_penalty_weight: int = 15
    explore_threshold_bp: int = 5000
    max_exploration_risk_bp: int = 3500

    def __post_init__(self) -> None:
        for name in (
            "prediction_error_weight",
            "novelty_weight",
            "reliability_weight",
            "risk_penalty_weight",
        ):
            object.__setattr__(self, name, require_non_negative_int(getattr(self, name), name))
        if self.positive_weight_sum() <= 0:
            raise HomeostasisError("curiosity positive weights must have positive sum")
        object.__setattr__(self, "explore_threshold_bp", require_bp(self.explore_threshold_bp, "explore_threshold_bp"))
        object.__setattr__(self, "max_exploration_risk_bp", require_bp(self.max_exploration_risk_bp, "max_exploration_risk_bp"))

    def positive_weight_sum(self) -> int:
        return self.prediction_error_weight + self.novelty_weight + self.reliability_weight


class CuriosityDrive:
    """Simulation-only curiosity module.

    It converts a predicted phase and a simulated observation into a curiosity
    signal. It does not mutate memory and never promotes observations to facts.
    """

    def __init__(self, config: CuriosityDriveConfig | None = None, *, modulus: int = DEFAULT_MODULUS) -> None:
        self.config = config or CuriosityDriveConfig()
        self.modulus = int(modulus)

    def prediction_error_bp(self, predicted_phase: Sequence[int], observed_phase: Sequence[int]) -> int:
        """Return average circular prediction error in basis points."""
        predicted = tuple(q_mod(value, self.modulus) for value in predicted_phase)
        observed = tuple(q_mod(value, self.modulus) for value in observed_phase)
        if len(predicted) != len(observed):
            raise HomeostasisError("predicted and observed phase dimensions must match")
        if not predicted:
            raise HomeostasisError("phase vectors must be non-empty")
        average_distance = sum(q_distance(a, b, self.modulus) for a, b in zip(predicted, observed)) // len(predicted)
        half_modulus = self.modulus // 2
        return clamp_bp((average_distance * MAX_BP) // half_modulus)

    def evaluate(
        self,
        *,
        predicted_phase: Sequence[int],
        observation: SensoryObservation,
        novelty_bp: int | None = None,
        risk_bp: int = 0,
    ) -> CuriositySignal:
        """Compute curiosity from prediction error, novelty, reliability and risk."""
        risk = require_bp(risk_bp, "risk_bp")
        novelty = require_bp(observation.intensity_bp if novelty_bp is None else novelty_bp, "novelty_bp")
        pred_error = self.prediction_error_bp(predicted_phase, observation.phase)
        positive = (
            self.config.prediction_error_weight * pred_error
            + self.config.novelty_weight * novelty
            + self.config.reliability_weight * observation.reliability_bp
        )
        penalty = self.config.risk_penalty_weight * risk
        raw = (positive - penalty) // self.config.positive_weight_sum()
        curiosity = clamp_bp(raw)
        explore = curiosity >= self.config.explore_threshold_bp and risk <= self.config.max_exploration_risk_bp
        if explore:
            reason = "prediction_error_novelty_reliable_low_risk"
        elif risk > self.config.max_exploration_risk_bp:
            reason = "risk_blocks_curiosity_exploration"
        else:
            reason = "curiosity_below_threshold"
        return CuriositySignal(
            prediction_error_bp=pred_error,
            novelty_bp=novelty,
            reliability_bp=observation.reliability_bp,
            risk_bp=risk,
            curiosity_bp=curiosity,
            explore_simulation=explore,
            reason=reason,
        )


@dataclass(frozen=True)
class HypothesisTestingResult:
    """Result of a simulation-only curiosity probe."""

    observation: SensoryObservation
    curiosity: CuriositySignal
    next_state: HomeostaticState
    suggested_action: str
    commit_allowed: bool = False
    real_sensor_commit_allowed: bool = False

    def __post_init__(self) -> None:
        if self.commit_allowed or self.real_sensor_commit_allowed:
            raise HomeostasisError("hypothesis-testing loop cannot authorize commits in v0.1")


class HypothesisTestingLoop:
    """Simulation-only loop for deciding whether to explore a surprising signal."""

    def __init__(self, curiosity_drive: CuriosityDrive | None = None, surrogate: "ActiveInferenceSurrogate" | None = None) -> None:
        self.curiosity_drive = curiosity_drive or CuriosityDrive()
        self.surrogate = surrogate or ActiveInferenceSurrogate()

    def evaluate(
        self,
        *,
        state: HomeostaticState,
        predicted_phase: Sequence[int],
        observation: SensoryObservation,
    ) -> HypothesisTestingResult:
        """Evaluate curiosity and return a non-committing exploration recommendation."""
        signal = self.curiosity_drive.evaluate(
            predicted_phase=predicted_phase,
            observation=observation,
            novelty_bp=observation.intensity_bp,
            risk_bp=state.risk_bp,
        )
        next_state = state.with_observation(
            novelty_delta_bp=signal.curiosity_bp // 4,
            uncertainty_delta_bp=signal.prediction_error_bp // 5,
            risk_delta_bp=signal.risk_bp // 10,
        )
        suggested = "simulate_hypothesis_test" if signal.explore_simulation else "hold_observation_for_evidence"
        return HypothesisTestingResult(
            observation=observation,
            curiosity=signal,
            next_state=next_state,
            suggested_action=suggested,
        )


@dataclass(frozen=True)
class ActiveInferenceConfig:
    """Integer thresholds for the homeostasis/control layer controller."""

    uncertainty_clarify_bp: int = 7000
    sleep_required_bp: int = 8000
    risk_block_simulated_bp: int = 7000
    novelty_explore_bp: int = 6500
    exploration_max_risk_bp: int = 2500
    exploration_max_uncertainty_bp: int = 6500

    def __post_init__(self) -> None:
        object.__setattr__(self, "uncertainty_clarify_bp", require_bp(self.uncertainty_clarify_bp, "uncertainty_clarify_bp"))
        object.__setattr__(self, "sleep_required_bp", require_bp(self.sleep_required_bp, "sleep_required_bp"))
        object.__setattr__(self, "risk_block_simulated_bp", require_bp(self.risk_block_simulated_bp, "risk_block_simulated_bp"))
        object.__setattr__(self, "novelty_explore_bp", require_bp(self.novelty_explore_bp, "novelty_explore_bp"))
        object.__setattr__(self, "exploration_max_risk_bp", require_bp(self.exploration_max_risk_bp, "exploration_max_risk_bp"))
        object.__setattr__(self, "exploration_max_uncertainty_bp", require_bp(self.exploration_max_uncertainty_bp, "exploration_max_uncertainty_bp"))


@dataclass(frozen=True)
class HomeostaticEvaluation:
    """Decision-oriented evaluation of a body state."""

    signal: ControlSignal
    reason: str
    viability_bp: int
    expected_free_energy_bp: int
    allow_simulated_action: bool
    explore_simulation: bool = False
    homeostatic_deviation_bp: int = 0


class ActiveInferenceSurrogate:
    """Bounded controller approximating active-inference-style selection.

    Lower expected-free-energy surrogate is better. The computation is an
    integer policy heuristic over risk, uncertainty, model error, homeostatic
    deviation, evidence gap, complexity, novelty gain and goal progress. It is
    not a biological FEP implementation.
    """

    def __init__(
        self,
        config: ActiveInferenceConfig | None = None,
        *,
        setpoint: HomeostaticSetpoint | None = None,
        weights: HomeostaticWeights | None = None,
    ) -> None:
        self.config = config or ActiveInferenceConfig()
        self.setpoint = setpoint or HomeostaticSetpoint()
        self.weights = weights or HomeostaticWeights()

    def homeostatic_deviation_bp(
        self,
        state: HomeostaticState,
        *,
        setpoint: HomeostaticSetpoint | None = None,
        weights: HomeostaticWeights | None = None,
    ) -> int:
        """Compute D(B_t) = weighted L1 distance from B*."""
        target = setpoint or self.setpoint
        weight_vec = weights or self.weights
        weighted_distance = 0
        for value, target_value, weight in zip(state.as_vector(), target.as_vector(), weight_vec.as_vector()):
            weighted_distance += abs(value - target_value) * weight
        return clamp_bp(weighted_distance // weight_vec.total_weight())

    def viability_bp(self, state: HomeostaticState) -> int:
        """Compute setpoint-aware viability.

        V(B_t) = 10000 - max(D(B_t), risk) - uncertainty/2, clamped.
        """
        deviation = self.homeostatic_deviation_bp(state)
        return clamp_bp(MAX_BP - max(deviation, state.risk_bp) - (state.uncertainty_bp // 2))

    def apply_action_effect(
        self,
        state: HomeostaticState,
        effect: HomeostaticActionEffect,
        *,
        domain_noise: DomainNoise | None = None,
    ) -> HomeostaticState:
        """Apply a deterministic simulated action effect to B_t."""
        return effect.apply(state, domain_noise)

    def expected_free_energy_bp(
        self,
        state: HomeostaticState,
        *,
        goal_gap_bp: int = 0,
        evidence_gap_bp: int = 0,
        model_error_bp: int = 0,
        complexity_bp: int = 0,
        novelty_gain_bp: int | None = None,
        goal_progress_bp: int = 0,
        action_effect: HomeostaticActionEffect | None = None,
        domain_noise: DomainNoise | None = None,
    ) -> int:
        """Compute bounded integer EFE surrogate.

        EFE(pi) rises with risk, uncertainty, model error, homeostatic
        deviation, evidence gap and complexity, and is reduced by novelty gain
        and goal progress. `goal_gap_bp` is kept for backward compatibility.
        """
        predicted_state = state
        effect = action_effect
        if effect is not None:
            predicted_state = self.apply_action_effect(state, effect, domain_noise=domain_noise)
            model_error_bp = max(model_error_bp, effect.model_error_bp)
            evidence_gap_bp = max(evidence_gap_bp, effect.evidence_gap_bp)
            complexity_bp = max(complexity_bp, effect.complexity_bp)
            goal_progress_bp = max(goal_progress_bp, effect.goal_progress_bp)
            if novelty_gain_bp is None:
                novelty_gain_bp = effect.novelty_gain_bp
        goal_gap = require_bp(goal_gap_bp, "goal_gap_bp")
        evidence_gap = require_bp(evidence_gap_bp, "evidence_gap_bp")
        model_error = require_bp(model_error_bp, "model_error_bp")
        complexity = require_bp(complexity_bp, "complexity_bp")
        goal_progress = require_bp(goal_progress_bp, "goal_progress_bp")
        novelty_gain = require_bp(predicted_state.novelty_bp if novelty_gain_bp is None else novelty_gain_bp, "novelty_gain_bp")
        deviation = self.homeostatic_deviation_bp(predicted_state)
        adverse = (
            predicted_state.risk_bp
            + predicted_state.uncertainty_bp
            + model_error
            + deviation
            + evidence_gap
            + complexity
            + goal_gap
        )
        beneficial = novelty_gain + goal_progress
        # Integer-only bounded surrogate. The denominator is a fixed BP-range
        # normalization factor: the adverse term aggregates the main bounded
        # cost channels, and // 6 keeps ordinary EFE values inside [0, 10000]
        # while preserving monotonicity and avoiding any floating-point path.
        return clamp_bp((adverse - beneficial) // 6)

    def evaluate(
        self,
        state: HomeostaticState,
        *,
        goal_gap_bp: int = 0,
        evidence_gap_bp: int = 0,
        model_error_bp: int = 0,
        complexity_bp: int = 0,
        novelty_gain_bp: int | None = None,
        goal_progress_bp: int = 0,
        action_effect: HomeostaticActionEffect | None = None,
        domain_noise: DomainNoise | None = None,
    ) -> HomeostaticEvaluation:
        """Select the next safe control recommendation."""
        predicted_state = self.apply_action_effect(state, action_effect, domain_noise=domain_noise) if action_effect else state
        efe = self.expected_free_energy_bp(
            state,
            goal_gap_bp=goal_gap_bp,
            evidence_gap_bp=evidence_gap_bp,
            model_error_bp=model_error_bp,
            complexity_bp=complexity_bp,
            novelty_gain_bp=novelty_gain_bp,
            goal_progress_bp=goal_progress_bp,
            action_effect=action_effect,
            domain_noise=domain_noise,
        )
        viability = self.viability_bp(predicted_state)
        deviation = self.homeostatic_deviation_bp(predicted_state)
        if predicted_state.risk_bp >= self.config.risk_block_simulated_bp:
            return HomeostaticEvaluation(
                signal=ControlSignal.BLOCK_SIMULATED_ACTION,
                reason="risk pressure blocks simulated action",
                viability_bp=viability,
                expected_free_energy_bp=efe,
                allow_simulated_action=False,
                homeostatic_deviation_bp=deviation,
            )
        if predicted_state.sleep_pressure_bp >= self.config.sleep_required_bp:
            return HomeostaticEvaluation(
                signal=ControlSignal.SLEEP_REQUIRED,
                reason="sleep pressure exceeds threshold",
                viability_bp=viability,
                expected_free_energy_bp=efe,
                allow_simulated_action=False,
                homeostatic_deviation_bp=deviation,
            )
        if predicted_state.uncertainty_bp >= self.config.uncertainty_clarify_bp:
            return HomeostaticEvaluation(
                signal=ControlSignal.ASK_CLARIFICATION,
                reason="uncertainty exceeds clarification threshold",
                viability_bp=viability,
                expected_free_energy_bp=efe,
                allow_simulated_action=False,
                homeostatic_deviation_bp=deviation,
            )
        if (
            predicted_state.novelty_bp >= self.config.novelty_explore_bp
            and predicted_state.risk_bp <= self.config.exploration_max_risk_bp
            and predicted_state.uncertainty_bp <= self.config.exploration_max_uncertainty_bp
        ):
            return HomeostaticEvaluation(
                signal=ControlSignal.EXPLORE_SIMULATION,
                reason="novelty can be explored in simulation",
                viability_bp=viability,
                expected_free_energy_bp=efe,
                allow_simulated_action=True,
                explore_simulation=True,
                homeostatic_deviation_bp=deviation,
            )
        return HomeostaticEvaluation(
            signal=ControlSignal.CONTINUE,
            reason="homeostatic state within bounded control thresholds",
            viability_bp=viability,
            expected_free_energy_bp=efe,
            allow_simulated_action=True,
            homeostatic_deviation_bp=deviation,
        )
