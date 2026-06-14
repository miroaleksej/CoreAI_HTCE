"""world model Q256 predictive world model for HTCE-Origin.

Scope boundary:
- Predicts bounded next toroidal state from an explicit Q256 state + explicit action delta.
- Measures observed-vs-predicted error with integer Q256 loss.
- Updates a compact self-model uncertainty estimate.
- Supports imagined rollouts over explicit actions.
- Does not create, infer, or commit facts. Fact memory belongs to the L1/L2/L3 body runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, TorusVector, active_state_digest, hash_to_phase
from htce_origin.kernel.q16 import (
    DEFAULT_MODULUS,
    LOSS_SCALE,
    Q16Error,
    q_add,
    q_mod,
    q_toroidal_loss_vector,
    q_vector,
    q_vector_add,
)

MAX_BP = 10000


class WorldModelError(ValueError):
    """Raised when the world model receives an invalid state/action contract."""


def _require_bp(value: int, name: str) -> int:
    bp = int(value)
    if not 0 <= bp <= MAX_BP:
        raise WorldModelError(f"{name} must be in [0, 10000]")
    return bp


def _require_nonnegative_raw(value: int, name: str) -> int:
    raw = int(value)
    if raw < 0:
        raise WorldModelError(f"{name} must be non-negative raw integer")
    return raw


def _signed_raw_to_report_bp(value: int, *, denominator: int) -> int:
    """Render-only conversion from raw score to basis points.

    P13 invariant: this helper is forbidden in action/gate selection.  It is
    used only to keep human-facing reports and legacy fields compatible.
    """
    if value <= 0:
        return 0
    denom = max(1, int(denominator))
    return max(0, min(MAX_BP, (int(value) * MAX_BP) // denom))


def signed_delta_n(a: int, b: int, modulus: int = DEFAULT_MODULUS) -> int:
    """Return signed circular delta from phase ``a`` toward phase ``b``.

    Output is in [-modulus//2, modulus//2]. This is used for bounded
    prediction-error updates and never leaves the integer torus contract.
    """
    if modulus <= 0:
        raise Q16Error("modulus must be positive")
    forward = (int(b) - int(a)) % int(modulus)
    half = int(modulus) // 2
    if forward <= half:
        return forward
    return forward - int(modulus)


def q256_saturating_signed(value: int, *, limit: int = DEFAULT_MODULUS // 4) -> int:
    """Integer-only saturating nonlinearity for adaptive Q256 dynamics.

    It is not a real tanh and does not use floating-point. It maps any signed
    accumulator into a bounded signed interval by the rational integer form
    ``value * limit // (abs(value) + limit)``.
    """
    x = int(value)
    cap = int(limit)
    if cap <= 0:
        raise WorldModelError("limit must be positive")
    denom = abs(x) + cap
    if denom == 0:
        return 0
    return (x * cap) // denom


def _signed_phase(phase: int, modulus: int) -> int:
    value = q_mod(phase, modulus)
    half = int(modulus) // 2
    return value if value <= half else value - int(modulus)


def _phase_from_signed(value: int, modulus: int) -> int:
    return q_mod(value, modulus)


def _tri_coeff(label: str, modulus: int = DEFAULT_MODULUS) -> int:
    phase = hash_to_phase(label, dimension=1, modulus=modulus, namespace="adaptive_coeff")[0]
    bucket = phase % 3
    if bucket == 0:
        return -1
    if bucket == 1:
        return 0
    return 1


@dataclass(frozen=True)
class AdaptivePredictionDetails:
    """Audit payload for one adaptive world-model prediction.

    The details contain only transition-model identifiers and digests. They do
    not contain subject/relation/object fact fields and cannot authorize a fact
    commit.
    """

    action_key: str
    context_key: str
    adaptive_delta: tuple[int, ...]
    hidden1_digest: str
    hidden2_digest: str
    correction_before: tuple[int, ...]


@dataclass
class AdaptiveQ256Dynamics:
    """Optional integer-only adaptive toroidal dynamics module.

    This is a bounded Q256 surrogate for a learned world model:

    z1 = sigma_Q256(A_a h + B_c c + b1)
    z2 = sigma_Q256(W2 z1 + b2)
    delta_hat = W_o z2 + p_a + p_c + correction mod N
    h_hat_next = h + delta_hat mod N

    All parameters are deterministic integer coefficients in {-1, 0, +1} or
    Q256 phases in Z_N. The module predicts transitions only; it cannot create,
    infer, or commit facts.
    """

    dimension: int = DEFAULT_TORUS_DIMENSION
    hidden_width: int = 8
    modulus: int = DEFAULT_MODULUS
    learning_rate_phase: int = 1
    correction_memory: dict[tuple[str, str], tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise WorldModelError("dimension must be positive")
        if self.hidden_width <= 0:
            raise WorldModelError("hidden_width must be positive")
        if self.modulus <= 0:
            raise Q16Error("modulus must be positive")
        self.learning_rate_phase = q_mod(self.learning_rate_phase, self.modulus)
        if self.learning_rate_phase == 0:
            self.learning_rate_phase = 1

    def _context_vector(self, context: str | Mapping[str, str | int | bool] | None) -> tuple[int, ...]:
        if context is None:
            key = "default"
        elif isinstance(context, Mapping):
            items = tuple(sorted((str(k), str(v)) for k, v in context.items()))
            key = "|".join(f"{k}={v}" for k, v in items) or "default"
        else:
            key = str(context)
        return hash_to_phase(key, dimension=self.dimension, modulus=self.modulus, namespace="adaptive_context")

    def _context_key(self, context: str | Mapping[str, str | int | bool] | None) -> str:
        return active_state_digest(TorusVector(self._context_vector(context), self.modulus))

    def _hidden_layer(
        self,
        values: tuple[int, ...],
        context_values: tuple[int, ...],
        *,
        action_key: str,
        layer: str,
    ) -> tuple[int, ...]:
        signed_values = tuple(_signed_phase(v, self.modulus) for v in values)
        signed_context = tuple(_signed_phase(v, self.modulus) for v in context_values)
        result: list[int] = []
        for j in range(self.hidden_width):
            acc = _signed_phase(hash_to_phase(f"{layer}:bias:{action_key}:{j}", dimension=1, modulus=self.modulus, namespace="adaptive_bias")[0], self.modulus)
            for i, value in enumerate(signed_values):
                coeff = _tri_coeff(f"{layer}:A:{action_key}:{j}:{i}", self.modulus)
                acc += coeff * value
            for i, value in enumerate(signed_context):
                coeff = _tri_coeff(f"{layer}:B:{action_key}:{j}:{i}", self.modulus)
                # Context is deliberately weaker than active state so that an
                # accidental context hash cannot dominate the explicit action.
                acc += coeff * (value // max(1, self.dimension))
            result.append(_phase_from_signed(q256_saturating_signed(acc), self.modulus))
        return tuple(result)

    def _output_delta(
        self,
        hidden2: tuple[int, ...],
        action: Q256WorldAction,
        context_vector: tuple[int, ...],
        correction: tuple[int, ...],
    ) -> tuple[int, ...]:
        signed_hidden = tuple(_signed_phase(v, self.modulus) for v in hidden2)
        signed_context = tuple(_signed_phase(v, self.modulus) for v in context_vector)
        output: list[int] = []
        for j in range(self.dimension):
            acc = 0
            for i, value in enumerate(signed_hidden):
                coeff = _tri_coeff(f"out:W:{action.name}:{j}:{i}", self.modulus)
                acc += coeff * value
            acc = acc // max(1, self.hidden_width)
            context_push = signed_context[j] // max(1, self.dimension * 4)
            adaptive_push = _phase_from_signed(q256_saturating_signed(acc + context_push, limit=self.modulus // 8), self.modulus)
            output.append(q_add(q_add(action.delta[j], adaptive_push, self.modulus), correction[j], self.modulus))
        return tuple(output)

    def predict(
        self,
        state: TorusVector,
        action: Q256WorldAction,
        *,
        context: str | Mapping[str, str | int | bool] | None = None,
    ) -> tuple[TorusVector, AdaptivePredictionDetails]:
        if state.dimension != self.dimension or action.dimension != self.dimension:
            raise WorldModelError("state/action dimension must match adaptive dynamics dimension")
        context_vector = self._context_vector(context if context is not None else action.metadata)
        context_key = self._context_key(context if context is not None else action.metadata)
        action_key = str(action.name)
        correction = self.correction_memory.get((action_key, context_key), tuple(0 for _ in range(self.dimension)))
        hidden1 = self._hidden_layer(state.phases, context_vector, action_key=action_key, layer="z1")
        hidden2 = self._hidden_layer(hidden1, context_vector, action_key=action_key, layer="z2")
        delta_hat = self._output_delta(hidden2, action, context_vector, correction)
        predicted = TorusVector(q_vector_add(state.phases, delta_hat, self.modulus), self.modulus)
        details = AdaptivePredictionDetails(
            action_key=action_key,
            context_key=context_key,
            adaptive_delta=delta_hat,
            hidden1_digest=active_state_digest(TorusVector(hidden1, self.modulus)),
            hidden2_digest=active_state_digest(TorusVector(hidden2, self.modulus)),
            correction_before=correction,
        )
        return predicted, details

    def update_from_error(self, prediction: "PredictionResult", observed_state: TorusVector) -> tuple[int, ...]:
        details = prediction.adaptive_details
        if details is None:
            raise WorldModelError("adaptive prediction details are required for adaptive update")
        previous = self.correction_memory.get((details.action_key, details.context_key), tuple(0 for _ in range(self.dimension)))
        updated: list[int] = []
        signed_lr = int(self.learning_rate_phase)
        for prev, predicted_phase, observed_phase in zip(previous, prediction.predicted_state.phases, observed_state.phases):
            error = signed_delta_n(predicted_phase, observed_phase, self.modulus)
            if error > 0:
                updated.append(q_add(prev, signed_lr, self.modulus))
            elif error < 0:
                updated.append(q_add(prev, -signed_lr, self.modulus))
            else:
                updated.append(q_mod(prev, self.modulus))
        correction = tuple(updated)
        self.correction_memory[(details.action_key, details.context_key)] = correction
        return correction


@dataclass(frozen=True)
class Q256WorldAction:
    """Explicit bounded action delta for simulation/prediction.

    The action is a state-transition proposal, not a factual claim. It carries only
    an action name, a Q256 delta vector, optional evidence/metadata, and a bounded
    confidence score. It deliberately has no subject/relation/object fact fields.
    """

    name: str
    delta: tuple[int, ...]
    evidence_id: str | None = None
    confidence_bp: int = MAX_BP
    metadata: Mapping[str, str | int | bool] = field(default_factory=dict)
    modulus: int = DEFAULT_MODULUS

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise WorldModelError("action name must be non-empty")
        object.__setattr__(self, "delta", q_vector(self.delta, self.modulus))
        object.__setattr__(self, "confidence_bp", _require_bp(self.confidence_bp, "confidence_bp"))

    @property
    def dimension(self) -> int:
        return len(self.delta)


@dataclass(frozen=True)
class PredictionError:
    """Observed-vs-predicted integer error report."""

    loss: int
    error_bp: int
    predicted_digest: str
    observed_digest: str
    matched: bool


@dataclass(frozen=True)
class PredictionResult:
    """Result of one Q256 next-state prediction."""

    action_name: str
    start_state: TorusVector
    predicted_state: TorusVector
    predicted_digest: str
    action_confidence_bp: int
    confidence_bp: int
    uncertainty_bp: int
    error: PredictionError | None = None
    adaptive_details: AdaptivePredictionDetails | None = None


@dataclass(frozen=True)
class WorldActionEvaluation:
    """P13 raw-integer EFE evaluation for one candidate action.

    The decision field is ``expected_free_energy_raw``.  Basis-point fields are
    report-only compatibility fields and must not be used for selection.  Lower
    raw EFE is better.
    """

    action_name: str
    predicted_state: TorusVector
    predicted_digest: str
    risk_raw: int
    uncertainty_raw: int
    complexity_raw: int
    novelty_gain_raw: int
    goal_progress_raw: int
    expected_free_energy_raw: int
    risk_bp: int
    uncertainty_bp: int
    complexity_bp: int
    novelty_gain_bp: int
    goal_progress_bp: int
    expected_free_energy_bp: int
    confidence_bp: int

    def __post_init__(self) -> None:
        _require_nonnegative_raw(self.risk_raw, "risk_raw")
        _require_nonnegative_raw(self.uncertainty_raw, "uncertainty_raw")
        _require_nonnegative_raw(self.complexity_raw, "complexity_raw")
        _require_nonnegative_raw(self.novelty_gain_raw, "novelty_gain_raw")
        _require_nonnegative_raw(self.goal_progress_raw, "goal_progress_raw")
        _require_bp(self.risk_bp, "risk_bp")
        _require_bp(self.uncertainty_bp, "uncertainty_bp")
        _require_bp(self.complexity_bp, "complexity_bp")
        _require_bp(self.novelty_gain_bp, "novelty_gain_bp")
        _require_bp(self.goal_progress_bp, "goal_progress_bp")
        _require_bp(self.expected_free_energy_bp, "expected_free_energy_bp")
        _require_bp(self.confidence_bp, "confidence_bp")


@dataclass(frozen=True)
class PerceptionActionLoopResult:
    """One grounded simulated perception-action loop result.

    This closes the local loop: evaluate candidate actions, select the minimum
    EFE simulated transition, measure observed-vs-predicted surprise, and update
    only the world-model uncertainty/correction state. It never commits facts or
    grants real action authority.
    """

    selected_action_name: str
    selected_evaluation: WorldActionEvaluation
    prediction: PredictionResult
    observed_prediction: PredictionResult
    surprise_raw: int
    self_model_uncertainty_raw: int
    surprise_bp: int
    self_model_uncertainty_bp: int
    adaptive_updated: bool

    def __post_init__(self) -> None:
        _require_nonnegative_raw(self.surprise_raw, "surprise_raw")
        _require_nonnegative_raw(self.self_model_uncertainty_raw, "self_model_uncertainty_raw")
        _require_bp(self.surprise_bp, "surprise_bp")
        _require_bp(self.self_model_uncertainty_bp, "self_model_uncertainty_bp")


@dataclass(frozen=True)
class WorldModelAblationRow:
    """One transition row for adaptive-world ablation.

    row_i = (h_t, action_t, h_observed_{t+1}).  It contains only toroidal
    transition state and never contains subject/relation/object fact fields.
    """

    start_state: TorusVector
    action: Q256WorldAction
    observed_state: TorusVector

    def __post_init__(self) -> None:
        if self.start_state.dimension != self.action.dimension:
            raise WorldModelError("ablation start/action dimension mismatch")
        if self.start_state.dimension != self.observed_state.dimension:
            raise WorldModelError("ablation start/observed dimension mismatch")
        if self.start_state.modulus != self.observed_state.modulus:
            raise WorldModelError("ablation row modulus mismatch")


@dataclass(frozen=True)
class WorldModelAblationReport:
    """Integer report comparing deterministic vs adaptive Q256 world loss.

    baseline_loss = L(deterministic_world, sequence)
    adaptive_loss_after_k = L(adaptive_world_after_k_updates, sequence)
    improvement_bp = 10000 * (baseline_loss - adaptive_loss_after_k) // baseline_loss
    """

    baseline_loss: int
    adaptive_loss_before: int
    adaptive_loss_after_k: int
    improvement_bp: int
    update_count: int
    transition_count: int
    facts_created: bool = False
    integer_only: bool = True

    def __post_init__(self) -> None:
        if self.baseline_loss < 0 or self.adaptive_loss_before < 0 or self.adaptive_loss_after_k < 0:
            raise WorldModelError("ablation losses must be non-negative")
        if self.update_count < 0 or self.transition_count < 0:
            raise WorldModelError("ablation counts must be non-negative")
        if self.facts_created:
            raise WorldModelError("world-model ablation cannot create facts")


@dataclass(frozen=True)
class ImaginedStep:
    """One step inside an imagined rollout."""

    index: int
    action_name: str
    start_digest: str
    predicted_digest: str
    confidence_bp: int
    uncertainty_bp: int


@dataclass(frozen=True)
class ImaginedRollout:
    """Sequence of predictions over explicit action deltas.

    This object is a simulated trajectory only. It does not authorize action and
    does not produce facts.
    """

    initial_digest: str
    final_state: TorusVector
    final_digest: str
    steps: tuple[ImaginedStep, ...]
    confidence_bp: int
    uncertainty_bp: int


@dataclass(frozen=True)
class SelfModelState:
    """Compact predictor self-state with separated raw/report uncertainty.

    Raw fields are the P13 decision state.  BP fields are report-only legacy
    renderings used by UI/tests/export.
    """

    observations: int = 0
    last_error_bp: int = 0
    mean_error_bp: int = 0
    uncertainty_bp: int = 0
    confidence_bp: int = MAX_BP
    high_error_count: int = 0
    last_error_raw: int = 0
    accumulated_error_raw: int = 0
    uncertainty_raw: int = 0

    def update(self, error_bp: int, *, high_error_threshold_bp: int, error_raw: int = 0) -> "SelfModelState":
        error = _require_bp(error_bp, "error_bp")
        raw_error = _require_nonnegative_raw(error_raw, "error_raw")
        observations = self.observations + 1
        if observations == 1:
            mean_error = error
        else:
            # Report-only integer smoothing.  It is not used by P13 selectors.
            mean_error = ((self.mean_error_bp * 3) + error) // 4
        high_count = self.high_error_count + (1 if error >= high_error_threshold_bp else 0)
        uncertainty = min(MAX_BP, (mean_error + error) // 2 + high_count * 100)
        confidence = max(0, MAX_BP - uncertainty)
        accumulated_raw = self.accumulated_error_raw + raw_error
        uncertainty_raw = accumulated_raw + raw_error
        return SelfModelState(
            observations=observations,
            last_error_bp=error,
            mean_error_bp=mean_error,
            uncertainty_bp=uncertainty,
            confidence_bp=confidence,
            high_error_count=high_count,
            last_error_raw=raw_error,
            accumulated_error_raw=accumulated_raw,
            uncertainty_raw=uncertainty_raw,
        )


@dataclass
class Q256WorldModel:
    """Integer-only toroidal world predictor with optional adaptive dynamics.

    Default mode remains the deterministic contract ``h' = h + delta mod N``.
    Adaptive mode is opt-in and adds a bounded Q256 transition predictor without
    giving the world model authority to create facts.
    """

    dimension: int = DEFAULT_TORUS_DIMENSION
    modulus: int = DEFAULT_MODULUS
    high_error_threshold_bp: int = 2500
    self_model: SelfModelState = field(default_factory=SelfModelState)
    adaptive_enabled: bool = False
    adaptive_dynamics: AdaptiveQ256Dynamics | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise WorldModelError("dimension must be positive")
        if self.modulus <= 0:
            raise Q16Error("modulus must be positive")
        self.high_error_threshold_bp = _require_bp(self.high_error_threshold_bp, "high_error_threshold_bp")
        if self.adaptive_enabled and self.adaptive_dynamics is None:
            self.adaptive_dynamics = AdaptiveQ256Dynamics(dimension=self.dimension, modulus=self.modulus)
        if self.adaptive_dynamics is not None:
            if self.adaptive_dynamics.dimension != self.dimension:
                raise WorldModelError("adaptive dynamics dimension does not match world model dimension")
            if self.adaptive_dynamics.modulus != self.modulus:
                raise WorldModelError("adaptive dynamics modulus does not match world model modulus")

    def _state(self, state: TorusVector | Iterable[int]) -> TorusVector:
        if isinstance(state, TorusVector):
            result = state
        else:
            result = TorusVector(tuple(state), self.modulus)
        if result.dimension != self.dimension:
            raise WorldModelError("state dimension does not match world model dimension")
        if result.modulus != self.modulus:
            raise WorldModelError("state modulus does not match world model modulus")
        return result

    def _action(self, action: Q256WorldAction | Iterable[int]) -> Q256WorldAction:
        if isinstance(action, Q256WorldAction):
            result = action
        else:
            result = Q256WorldAction(name="explicit_delta", delta=tuple(action), modulus=self.modulus)
        if result.dimension != self.dimension:
            raise WorldModelError("action delta dimension does not match world model dimension")
        if result.modulus != self.modulus:
            raise WorldModelError("action modulus does not match world model modulus")
        return result

    def _adaptive(self) -> AdaptiveQ256Dynamics:
        if self.adaptive_dynamics is None:
            self.adaptive_dynamics = AdaptiveQ256Dynamics(dimension=self.dimension, modulus=self.modulus)
        return self.adaptive_dynamics

    def _prediction_result(
        self,
        *,
        start: TorusVector,
        action: Q256WorldAction,
        predicted: TorusVector,
        adaptive_details: AdaptivePredictionDetails | None = None,
    ) -> PredictionResult:
        confidence = min(action.confidence_bp, self.self_model.confidence_bp)
        uncertainty = max(MAX_BP - action.confidence_bp, self.self_model.uncertainty_bp)
        return PredictionResult(
            action_name=action.name,
            start_state=start,
            predicted_state=predicted,
            predicted_digest=active_state_digest(predicted),
            action_confidence_bp=action.confidence_bp,
            confidence_bp=confidence,
            uncertainty_bp=uncertainty,
            adaptive_details=adaptive_details,
        )

    def predict_next_state(self, state: TorusVector | Iterable[int], action: Q256WorldAction | Iterable[int]) -> PredictionResult:
        """Predict next state. Default is state + action.delta mod N; adaptive mode is opt-in."""
        if self.adaptive_enabled:
            return self.predict_adaptive(state, action)
        start = self._state(state)
        act = self._action(action)
        predicted = TorusVector(q_vector_add(start.phases, act.delta, self.modulus), self.modulus)
        return self._prediction_result(start=start, action=act, predicted=predicted)

    def predict_adaptive(
        self,
        state: TorusVector | Iterable[int],
        action: Q256WorldAction | Iterable[int],
        *,
        context: str | Mapping[str, str | int | bool] | None = None,
    ) -> PredictionResult:
        """Predict with optional adaptive integer/Q256 hidden dynamics.

        This method predicts toroidal transitions only. It cannot create facts and
        does not commit predicted states into L2/L3 memory.
        """
        start = self._state(state)
        act = self._action(action)
        predicted, details = self._adaptive().predict(start, act, context=context)
        return self._prediction_result(start=start, action=act, predicted=predicted, adaptive_details=details)

    def predict_next(self, state: TorusVector | Iterable[int], action: Q256WorldAction | Iterable[int]) -> PredictionResult:
        return self.predict_next_state(state, action)

    def measure_prediction_error(self, prediction: PredictionResult, observed_state: TorusVector | Iterable[int]) -> PredictionError:
        """Measure Q256 observed-vs-predicted error with LUT-backed toroidal loss."""
        observed = self._state(observed_state)
        loss = q_toroidal_loss_vector(prediction.predicted_state.phases, observed.phases, self.modulus)
        max_loss = self.dimension * LOSS_SCALE
        error_bp = (loss * MAX_BP) // max_loss if max_loss else MAX_BP
        return PredictionError(
            loss=loss,
            error_bp=error_bp,
            predicted_digest=prediction.predicted_digest,
            observed_digest=active_state_digest(observed),
            matched=(error_bp == 0),
        )

    def observe(self, prediction: PredictionResult, observed_state: TorusVector | Iterable[int]) -> PredictionResult:
        """Attach observed error to a prediction and update self-model uncertainty."""
        error = self.measure_prediction_error(prediction, observed_state)
        self.self_model = self.self_model.update(
            error.error_bp,
            high_error_threshold_bp=self.high_error_threshold_bp,
            error_raw=error.loss,
        )
        confidence = min(prediction.action_confidence_bp, self.self_model.confidence_bp)
        uncertainty = max(MAX_BP - prediction.action_confidence_bp, self.self_model.uncertainty_bp)
        return PredictionResult(
            action_name=prediction.action_name,
            start_state=prediction.start_state,
            predicted_state=prediction.predicted_state,
            predicted_digest=prediction.predicted_digest,
            action_confidence_bp=prediction.action_confidence_bp,
            confidence_bp=confidence,
            uncertainty_bp=uncertainty,
            error=error,
            adaptive_details=prediction.adaptive_details,
        )

    def update_from_observation(
        self,
        prediction: PredictionResult,
        observed_state: TorusVector | Iterable[int],
    ) -> PredictionResult:
        """Observe a prediction and, when adaptive details exist, update Q256 dynamics.

        The update is a bounded circular-error correction over transition
        parameters only. It cannot create or commit facts.
        """
        observed = self._state(observed_state)
        updated_prediction = self.observe(prediction, observed)
        if prediction.adaptive_details is not None:
            self._adaptive().update_from_error(prediction, observed)
        return updated_prediction

    def _ablation_row(self, row: WorldModelAblationRow | tuple[object, object, object]) -> WorldModelAblationRow:
        if isinstance(row, WorldModelAblationRow):
            result = row
        else:
            start_raw, action_raw, observed_raw = row
            result = WorldModelAblationRow(
                start_state=self._state(start_raw),
                action=self._action(action_raw),
                observed_state=self._state(observed_raw),
            )
        if result.start_state.dimension != self.dimension:
            raise WorldModelError("ablation row dimension does not match world model")
        return result

    def deterministic_sequence_loss(self, sequence: Iterable[WorldModelAblationRow | tuple[object, object, object]]) -> int:
        """Return L(deterministic_world, sequence) with integer Q256 loss."""
        rows = tuple(self._ablation_row(row) for row in sequence)
        deterministic = Q256WorldModel(dimension=self.dimension, modulus=self.modulus)
        total_loss = 0
        for row in rows:
            prediction = deterministic.predict_next_state(row.start_state, row.action)
            total_loss += deterministic.measure_prediction_error(prediction, row.observed_state).loss
        return int(total_loss)

    def adaptive_sequence_loss(self, sequence: Iterable[WorldModelAblationRow | tuple[object, object, object]]) -> int:
        """Return L(adaptive_world, sequence) without updating parameters."""
        rows = tuple(self._ablation_row(row) for row in sequence)
        total_loss = 0
        for row in rows:
            prediction = self.predict_adaptive(row.start_state, row.action)
            total_loss += self.measure_prediction_error(prediction, row.observed_state).loss
        return int(total_loss)

    def adaptive_ablation(
        self,
        sequence: Iterable[WorldModelAblationRow | tuple[object, object, object]],
        *,
        update_passes: int = 1,
    ) -> WorldModelAblationReport:
        """Train/update adaptive Q256 dynamics on a transition sequence and report loss.

        This is a calibration diagnostic only. It never commits predicted states
        and never creates facts. The deterministic baseline remains h' = h + Δ.
        """
        rows = tuple(self._ablation_row(row) for row in sequence)
        if not rows:
            raise WorldModelError("ablation sequence must not be empty")
        passes = int(update_passes)
        if passes <= 0:
            raise WorldModelError("update_passes must be positive")
        baseline_loss = self.deterministic_sequence_loss(rows)
        adaptive_before = self.adaptive_sequence_loss(rows)
        update_count = 0
        for _ in range(passes):
            for row in rows:
                prediction = self.predict_adaptive(row.start_state, row.action)
                self.update_from_observation(prediction, row.observed_state)
                update_count += 1
        adaptive_after = self.adaptive_sequence_loss(rows)
        if baseline_loss == 0:
            improvement_bp = MAX_BP if adaptive_after == 0 else 0
        else:
            improvement_bp = ((baseline_loss - adaptive_after) * MAX_BP) // baseline_loss
        return WorldModelAblationReport(
            baseline_loss=baseline_loss,
            adaptive_loss_before=adaptive_before,
            adaptive_loss_after_k=adaptive_after,
            improvement_bp=int(improvement_bp),
            update_count=update_count,
            transition_count=len(rows),
            facts_created=False,
            integer_only=True,
        )

    def _report_bp_from_loss_raw(self, loss_raw: int) -> int:
        return _signed_raw_to_report_bp(loss_raw, denominator=self.dimension * LOSS_SCALE)

    def _legacy_bp_to_raw(self, bp: int) -> int:
        # Compatibility bridge only.  P13 callers should pass raw values.
        return _require_bp(bp, "bp") * LOSS_SCALE

    def evaluate_expected_free_energy_raw(
        self,
        predicted_state: TorusVector | Iterable[int],
        target_state: TorusVector | Iterable[int] | None = None,
        *,
        complexity_raw: int = 0,
        novelty_gain_raw: int = 0,
        goal_progress_raw: int = 0,
    ) -> int:
        """Return raw Q256 EFE decision score with no normalization.

        EFE_raw = Risk_raw + Uncertainty_raw + Complexity_raw
                  - Novelty_raw - GoalProgress_raw.
        This is the only score intended for action selection.
        """
        predicted = self._state(predicted_state)
        risk_raw = 0
        if target_state is not None:
            target = self._state(target_state)
            risk_raw = q_toroidal_loss_vector(predicted.phases, target.phases, self.modulus)
        uncertainty_raw = self.self_model.uncertainty_raw
        complexity = _require_nonnegative_raw(complexity_raw, "complexity_raw")
        novelty = _require_nonnegative_raw(novelty_gain_raw, "novelty_gain_raw")
        progress = _require_nonnegative_raw(goal_progress_raw, "goal_progress_raw")
        return risk_raw + uncertainty_raw + complexity - novelty - progress

    def evaluate_expected_free_energy(
        self,
        predicted_state: TorusVector | Iterable[int],
        target_state: TorusVector | Iterable[int] | None = None,
        *,
        complexity_bp: int = 0,
        novelty_gain_bp: int = 0,
        goal_progress_bp: int = 0,
    ) -> int:
        """Render-only legacy BP EFE wrapper.

        P13 invariant: selectors must call ``evaluate_expected_free_energy_raw``
        or compare ``WorldActionEvaluation.expected_free_energy_raw``.
        """
        raw = self.evaluate_expected_free_energy_raw(
            predicted_state,
            target_state,
            complexity_raw=self._legacy_bp_to_raw(complexity_bp),
            novelty_gain_raw=self._legacy_bp_to_raw(novelty_gain_bp),
            goal_progress_raw=self._legacy_bp_to_raw(goal_progress_bp),
        )
        return self._report_bp_from_loss_raw(max(0, raw))

    def evaluate_action_expected_free_energy(
        self,
        state: TorusVector | Iterable[int],
        action: Q256WorldAction | Iterable[int],
        *,
        target_state: TorusVector | Iterable[int] | None = None,
        context: str | Mapping[str, str | int | bool] | None = None,
        complexity_bp: int = 0,
        novelty_gain_bp: int = 0,
        goal_progress_bp: int = 0,
        complexity_raw: int | None = None,
        novelty_gain_raw: int | None = None,
        goal_progress_raw: int | None = None,
    ) -> WorldActionEvaluation:
        """Predict one candidate action and score it by raw integer EFE.

        Basis-point inputs are accepted only for compatibility and are converted
        into raw report-equivalent integers before the decision score is built.
        New runtime callers pass the ``*_raw`` arguments directly.
        """
        start = self._state(state)
        act = self._action(action)
        if self.adaptive_enabled:
            prediction = self.predict_adaptive(start, act, context=context)
        else:
            prediction = self.predict_next_state(start, act)
        risk_raw = 0
        if target_state is not None:
            target = self._state(target_state)
            risk_raw = q_toroidal_loss_vector(prediction.predicted_state.phases, target.phases, self.modulus)
        complexity_r = _require_nonnegative_raw(
            self._legacy_bp_to_raw(complexity_bp) if complexity_raw is None else complexity_raw,
            "complexity_raw",
        )
        novelty_r = _require_nonnegative_raw(
            self._legacy_bp_to_raw(novelty_gain_bp) if novelty_gain_raw is None else novelty_gain_raw,
            "novelty_gain_raw",
        )
        progress_r = _require_nonnegative_raw(
            self._legacy_bp_to_raw(goal_progress_bp) if goal_progress_raw is None else goal_progress_raw,
            "goal_progress_raw",
        )
        uncertainty_raw = self.self_model.uncertainty_raw
        efe_raw = risk_raw + uncertainty_raw + complexity_r - novelty_r - progress_r
        return WorldActionEvaluation(
            action_name=prediction.action_name,
            predicted_state=prediction.predicted_state,
            predicted_digest=prediction.predicted_digest,
            risk_raw=risk_raw,
            uncertainty_raw=uncertainty_raw,
            complexity_raw=complexity_r,
            novelty_gain_raw=novelty_r,
            goal_progress_raw=progress_r,
            expected_free_energy_raw=efe_raw,
            risk_bp=self._report_bp_from_loss_raw(risk_raw),
            uncertainty_bp=self.self_model.uncertainty_bp,
            complexity_bp=_require_bp(complexity_bp, "complexity_bp"),
            novelty_gain_bp=_require_bp(novelty_gain_bp, "novelty_gain_bp"),
            goal_progress_bp=_require_bp(goal_progress_bp, "goal_progress_bp"),
            expected_free_energy_bp=self._report_bp_from_loss_raw(max(0, efe_raw)),
            confidence_bp=prediction.confidence_bp,
        )

    def select_min_expected_free_energy_action(
        self,
        state: TorusVector | Iterable[int],
        actions: Iterable[Q256WorldAction | Iterable[int]],
        *,
        target_state: TorusVector | Iterable[int] | None = None,
        context: str | Mapping[str, str | int | bool] | None = None,
        complexity_bp: int = 0,
        novelty_gain_bp: int = 0,
        goal_progress_bp: int = 0,
        complexity_raw: int | None = None,
        novelty_gain_raw: int | None = None,
        goal_progress_raw: int | None = None,
    ) -> WorldActionEvaluation:
        """Return candidate action with minimal raw Q256 EFE.

        Ties are resolved by input order to keep the selection deterministic.
        """
        evaluations = []
        for action in actions:
            evaluations.append(
                self.evaluate_action_expected_free_energy(
                    state,
                    action,
                    target_state=target_state,
                    context=context,
                    complexity_bp=complexity_bp,
                    novelty_gain_bp=novelty_gain_bp,
                    goal_progress_bp=goal_progress_bp,
                    complexity_raw=complexity_raw,
                    novelty_gain_raw=novelty_gain_raw,
                    goal_progress_raw=goal_progress_raw,
                )
            )
        if not evaluations:
            raise WorldModelError("candidate actions must not be empty")
        return min(evaluations, key=lambda item: item.expected_free_energy_raw)

    def grounded_perception_action_step(
        self,
        state: TorusVector | Iterable[int],
        actions: Iterable[Q256WorldAction | Iterable[int]],
        observed_state: TorusVector | Iterable[int],
        *,
        target_state: TorusVector | Iterable[int] | None = None,
        context: str | Mapping[str, str | int | bool] | None = None,
        complexity_bp: int = 0,
        novelty_gain_bp: int = 0,
        goal_progress_bp: int = 0,
        complexity_raw: int | None = None,
        novelty_gain_raw: int | None = None,
        goal_progress_raw: int | None = None,
    ) -> PerceptionActionLoopResult:
        """Run one simulated grounded perception-action update.

        Action choice is based only on raw Q256 EFE.  BP fields are produced only
        after selection for trace/report compatibility.
        """
        raw_actions = tuple(actions)
        selected = self.select_min_expected_free_energy_action(
            state,
            raw_actions,
            target_state=target_state,
            context=context,
            complexity_bp=complexity_bp,
            novelty_gain_bp=novelty_gain_bp,
            goal_progress_bp=goal_progress_bp,
            complexity_raw=complexity_raw,
            novelty_gain_raw=novelty_gain_raw,
            goal_progress_raw=goal_progress_raw,
        )
        start = self._state(state)
        selected_action = None
        for action in raw_actions:
            coerced = self._action(action)
            if coerced.name == selected.action_name:
                selected_action = coerced
                break
        if selected_action is None:
            raise WorldModelError("selected action disappeared during loop evaluation")
        if self.adaptive_enabled:
            prediction = self.predict_adaptive(start, selected_action, context=context)
        else:
            prediction = self.predict_next_state(start, selected_action)
        observed_prediction = self.update_from_observation(prediction, observed_state)
        surprise_raw = observed_prediction.error.loss if observed_prediction.error else 0
        surprise_bp = observed_prediction.error.error_bp if observed_prediction.error else MAX_BP
        return PerceptionActionLoopResult(
            selected_action_name=selected.action_name,
            selected_evaluation=selected,
            prediction=prediction,
            observed_prediction=observed_prediction,
            surprise_raw=surprise_raw,
            self_model_uncertainty_raw=self.self_model.uncertainty_raw,
            surprise_bp=surprise_bp,
            self_model_uncertainty_bp=self.self_model.uncertainty_bp,
            adaptive_updated=prediction.adaptive_details is not None and self.adaptive_enabled,
        )

    def imagined_rollout(
        self,
        initial_state: TorusVector | Iterable[int],
        actions: Iterable[Q256WorldAction | Iterable[int]],
    ) -> ImaginedRollout:
        """Run a simulated rollout over explicit Q256 action deltas."""
        current = self._state(initial_state)
        initial_digest = active_state_digest(current)
        steps: list[ImaginedStep] = []
        confidence = self.self_model.confidence_bp
        uncertainty = self.self_model.uncertainty_bp
        for index, action in enumerate(actions):
            prediction = self.predict_next_state(current, action)
            steps.append(
                ImaginedStep(
                    index=index,
                    action_name=prediction.action_name,
                    start_digest=active_state_digest(current),
                    predicted_digest=prediction.predicted_digest,
                    confidence_bp=prediction.confidence_bp,
                    uncertainty_bp=prediction.uncertainty_bp,
                )
            )
            current = prediction.predicted_state
            confidence = min(confidence, prediction.confidence_bp)
            uncertainty = max(uncertainty, prediction.uncertainty_bp)
        return ImaginedRollout(
            initial_digest=initial_digest,
            final_state=current,
            final_digest=active_state_digest(current),
            steps=tuple(steps),
            confidence_bp=confidence,
            uncertainty_bp=uncertainty,
        )

    def can_confidently_answer(self, *, max_uncertainty_bp: int = 2500) -> bool:
        """Return whether the current self-model uncertainty permits confidence."""
        threshold = _require_bp(max_uncertainty_bp, "max_uncertainty_bp")
        return self.self_model.uncertainty_bp <= threshold

# Backward-compatible public aliases for older integrations. The default modulus
# is Q256; these names do not switch the runtime back to a 16-bit profile.
Q16WorldAction = Q256WorldAction
Q16WorldModel = Q256WorldModel
AdaptiveQ16Dynamics = AdaptiveQ256Dynamics
q16_tanh_like = q256_saturating_signed
