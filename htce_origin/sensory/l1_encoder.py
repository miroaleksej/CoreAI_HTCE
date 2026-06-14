"""Exact L1 sensory encoder for HTCE-Origin Q256 runtime.

Mathematical contract
---------------------
The protected L1 path maps driver-quantized sensor samples into a bounded
large-range toroidal phase without floating-point arithmetic:

    x_q[j] = floor((clip(x[j], lo, hi) - lo) * (N - 1) / (hi - lo))
    u_obs[i] = b[i] + sum_j W[i,j] * x_q[j] mod N,
    W[i,j] in {-1, 0, +1}.

For commit to the L1 body, the encoder returns the delta that makes the active
L1 state equal to the observed phase:

    Delta_L1 = u_obs - h_L1 mod N.

Raw analogue/float values are deliberately outside this module.  A sensor
adapter or ADC driver must provide bounded integer samples plus an integer
calibration interval.  This preserves the HTCE protected-path invariant:
source and runtime contain no floating-point constants or operations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from htce_origin.cognition.world import PredictionError
from htce_origin.control.homeostasis import CuriosityDrive, CuriositySignal, SensoryObservation
from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, TorusVector, active_state_digest, hash_to_phase
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_add, q_distance, q_mod, q_sub, q_vector

MAX_BP = 10000
DEFAULT_L1_INPUT_DIM = 256


class SensoryError(ValueError):
    """Raised when a sensory packet violates the L1 boundary contract."""


def _require_bp(value: int, name: str) -> int:
    bp = int(value)
    if not 0 <= bp <= MAX_BP:
        raise SensoryError(f"{name} must be in [0, 10000]")
    return bp


@dataclass(frozen=True)
class RawSensorPacket:
    """Driver-quantized sensor packet at the L1 boundary.

    ``samples`` are integers emitted by an ADC, simulator, parser, vision
    frontend, audio frontend, proprioceptive driver, or API adapter.  The pair
    ``sample_min``/``sample_max`` defines the integer calibration interval that
    is mapped onto ``Z_N``.  Values outside the interval are clipped before
    quantization, which prevents topological tearing at the boundary.
    """

    modality: str
    samples: Sequence[int]
    sample_min: int = 0
    sample_max: int = (1 << 16) - 1
    reliability_bp: int = MAX_BP
    evidence_id: str = "l1_sensor_boundary"
    metadata: Mapping[str, str | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.modality).strip():
            raise SensoryError("modality must be non-empty")
        if not self.samples:
            raise SensoryError("samples sequence must not be empty")
        if int(self.sample_max) <= int(self.sample_min):
            raise SensoryError("sample_max must be greater than sample_min")
        object.__setattr__(self, "reliability_bp", _require_bp(self.reliability_bp, "reliability_bp"))
        if not str(self.evidence_id).strip():
            raise SensoryError("evidence_id must be non-empty")


@dataclass(frozen=True)
class QuantizedSensorState:
    """Strict integer representation of an L1 sensor packet in ``Z_N``."""

    modality: str
    q_values: tuple[int, ...]
    reliability_bp: int
    evidence_id: str
    sample_count: int
    modulus: int = DEFAULT_MODULUS
    metadata: Mapping[str, str | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.q_values:
            raise SensoryError("q_values must be non-empty")
        object.__setattr__(self, "q_values", q_vector(self.q_values, self.modulus))
        object.__setattr__(self, "reliability_bp", _require_bp(self.reliability_bp, "reliability_bp"))
        if self.sample_count <= 0:
            raise SensoryError("sample_count must be positive")


@dataclass(frozen=True)
class L1ProjectionWeights:
    """Deterministic ternary projection matrix and Q256 bias."""

    dimension: int
    input_dim: int
    modulus: int
    weights: tuple[tuple[int, ...], ...]
    bias: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.input_dim <= 0:
            raise SensoryError("dimensions must be positive")
        if len(self.weights) != self.dimension:
            raise SensoryError("weight matrix row count must match dimension")
        for row in self.weights:
            if len(row) != self.input_dim:
                raise SensoryError("weight matrix column count must match input_dim")
            if any(value not in (-1, 0, 1) for value in row):
                raise SensoryError("projection weights must be ternary")
        if len(self.bias) != self.dimension:
            raise SensoryError("bias length must match dimension")
        object.__setattr__(self, "bias", q_vector(self.bias, self.modulus))


@dataclass(frozen=True)
class L1EncodedObservation:
    """Complete output of the exact L1 pipeline."""

    quantized: QuantizedSensorState
    observed_phase: tuple[int, ...]
    delta: tuple[int, ...]
    prediction_error: PredictionError
    curiosity: CuriositySignal
    projection_digest: str

    def as_payload(self) -> dict[str, object]:
        return {
            "curiosity": {
                "curiosity_bp": self.curiosity.curiosity_bp,
                "explore_simulation": self.curiosity.explore_simulation,
                "reason": self.curiosity.reason,
            },
            "delta_digest": active_state_digest(TorusVector(self.delta, self.quantized.modulus)),
            "evidence_id": self.quantized.evidence_id,
            "modality": self.quantized.modality,
            "observed_digest": active_state_digest(TorusVector(self.observed_phase, self.quantized.modulus)),
            "prediction_error_bp": self.prediction_error.error_bp,
            "projection_digest": self.projection_digest,
            "sample_count": self.quantized.sample_count,
        }


class L1SensoryEncoder:
    """Exact ternary sensory projection into the HTCE L1 torus."""

    def __init__(
        self,
        *,
        torus_dimension: int = DEFAULT_TORUS_DIMENSION,
        input_dim: int = DEFAULT_L1_INPUT_DIM,
        modulus: int = DEFAULT_MODULUS,
        seed: str = "HTCE-L1-SENSORY-Q256-v1",
        curiosity_drive: CuriosityDrive | None = None,
    ) -> None:
        if torus_dimension <= 0 or input_dim <= 0:
            raise SensoryError("dimensions must be positive")
        if modulus <= 1:
            raise SensoryError("modulus must be greater than one")
        self.dimension = int(torus_dimension)
        self.input_dim = int(input_dim)
        self.modulus = int(modulus)
        self.seed = str(seed)
        self.projection = L1ProjectionWeights(
            dimension=self.dimension,
            input_dim=self.input_dim,
            modulus=self.modulus,
            weights=self._generate_ternary_projection(),
            bias=self._generate_bias(),
        )
        self.curiosity_drive = curiosity_drive or CuriosityDrive(modulus=self.modulus)

    def _generate_ternary_projection(self) -> tuple[tuple[int, ...], ...]:
        rows: list[tuple[int, ...]] = []
        for row in range(self.dimension):
            values: list[int] = []
            for col in range(self.input_dim):
                digest = hashlib.blake2b(
                    f"{self.seed}|W|{row}|{col}".encode("utf-8"),
                    digest_size=1,
                    person=b"L1Ternry",
                ).digest()[0]
                bucket = digest % 3
                if bucket == 0:
                    values.append(-1)
                elif bucket == 1:
                    values.append(0)
                else:
                    values.append(1)
            rows.append(tuple(values))
        return tuple(rows)

    def _generate_bias(self) -> tuple[int, ...]:
        return hash_to_phase(f"{self.seed}|bias", dimension=self.dimension, modulus=self.modulus, namespace="l1_bias")

    def quantize_packet(self, packet: RawSensorPacket) -> QuantizedSensorState:
        lower = int(packet.sample_min)
        upper = int(packet.sample_max)
        span = upper - lower
        limit = self.modulus - 1
        values: list[int] = []
        identity_q256_interval = lower == 0 and upper == self.modulus - 1
        for raw in packet.samples:
            value = int(raw)
            if value < lower:
                value = lower
            if value > upper:
                value = upper
            if identity_q256_interval:
                # Exact fast path for the native Q256 driver interval [0, N-1]:
                # floor((x-0)*(N-1)/(N-1)) = x.
                values.append(q_mod(value, self.modulus))
            else:
                values.append(((value - lower) * limit) // span)
        sample_count = len(values)
        if len(values) < self.input_dim:
            values.extend(0 for _ in range(self.input_dim - len(values)))
        else:
            values = values[: self.input_dim]
        return QuantizedSensorState(
            modality=str(packet.modality).strip().lower(),
            q_values=tuple(values),
            reliability_bp=packet.reliability_bp,
            evidence_id=str(packet.evidence_id),
            sample_count=sample_count,
            modulus=self.modulus,
            metadata=dict(packet.metadata),
        )

    def project_to_torus(self, q_state: QuantizedSensorState) -> tuple[int, ...]:
        if len(q_state.q_values) != self.input_dim:
            raise SensoryError("quantized state dimension mismatch")
        projected: list[int] = []
        for row_index, row in enumerate(self.projection.weights):
            acc = self.projection.bias[row_index]
            for weight, sample in zip(row, q_state.q_values):
                if weight == 1:
                    acc = q_add(acc, sample, self.modulus)
                elif weight == -1:
                    acc = q_sub(acc, sample, self.modulus)
            projected.append(q_mod(acc, self.modulus))
        return tuple(projected)

    def observation_delta(self, current_l1_phase: Iterable[int], observed_phase: Iterable[int]) -> tuple[int, ...]:
        current = q_vector(current_l1_phase, self.modulus)
        observed = q_vector(observed_phase, self.modulus)
        if len(current) != self.dimension or len(observed) != self.dimension:
            raise SensoryError("L1 phase dimension mismatch")
        return tuple(q_sub(obs, cur, self.modulus) for cur, obs in zip(current, observed))

    def prediction_error(self, predicted_phase: Iterable[int], observed_phase: Iterable[int]) -> PredictionError:
        predicted = q_vector(predicted_phase, self.modulus)
        observed = q_vector(observed_phase, self.modulus)
        if len(predicted) != self.dimension or len(observed) != self.dimension:
            raise SensoryError("prediction/observation dimension mismatch")
        half = self.modulus // 2
        max_loss = self.dimension * half * half
        actual_loss = sum(q_distance(obs, pred, self.modulus) ** 2 for obs, pred in zip(observed, predicted))
        error_bp = 0 if max_loss == 0 else min(MAX_BP, (actual_loss * MAX_BP) // max_loss)
        return PredictionError(
            loss=actual_loss,
            error_bp=error_bp,
            predicted_digest=active_state_digest(TorusVector(predicted, self.modulus)),
            observed_digest=active_state_digest(TorusVector(observed, self.modulus)),
            matched=actual_loss == 0,
        )

    def encode(
        self,
        packet: RawSensorPacket,
        *,
        current_l1_phase: Iterable[int],
        predicted_phase: Iterable[int] | None = None,
        current_risk_bp: int = 0,
    ) -> L1EncodedObservation:
        q_state = self.quantize_packet(packet)
        observed_phase = self.project_to_torus(q_state)
        predicted = tuple(predicted_phase) if predicted_phase is not None else tuple(current_l1_phase)
        error = self.prediction_error(predicted, observed_phase)
        sensory_obs = SensoryObservation(
            modality=q_state.modality,
            value=q_state.evidence_id,
            intensity_bp=q_state.reliability_bp,
            reliability_bp=q_state.reliability_bp,
            phase=observed_phase,
            evidence_id=q_state.evidence_id,
            simulated=True,
            real_sensor_commit_allowed=False,
            modulus=self.modulus,
        )
        curiosity = self.curiosity_drive.evaluate(
            predicted_phase=predicted,
            observation=sensory_obs,
            risk_bp=_require_bp(current_risk_bp, "current_risk_bp"),
        )
        return L1EncodedObservation(
            quantized=q_state,
            observed_phase=observed_phase,
            delta=self.observation_delta(current_l1_phase, observed_phase),
            prediction_error=error,
            curiosity=curiosity,
            projection_digest=active_state_digest(
                {
                    "bias": self.projection.bias,
                    "dimension": self.dimension,
                    "input_dim": self.input_dim,
                    "modulus": self.modulus,
                    "seed": self.seed,
                }
            ),
        )

    def get_projection_matrix_for_hardware(self) -> dict[str, object]:
        """Return a deterministic hardware/export payload for ROM generation."""
        return {
            "bias": self.projection.bias,
            "dimension": self.dimension,
            "input_dim": self.input_dim,
            "modulus": self.modulus,
            "seed": self.seed,
            "weights": self.projection.weights,
        }
