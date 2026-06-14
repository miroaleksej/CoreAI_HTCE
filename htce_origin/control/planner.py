"""simulation planner proof-guided planner and simulation habitat boundary.

Scope boundary:
- Extends the homeostasis/control layer homeostatic planner with a small simulated skill registry.
- Executes no real-world action and mutates no L1/L2/L3 memory.
- Can produce traceable simulation-only plan decisions.
- Real actions remain blocked because signed-manifest execution is outside simulation planner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from htce_origin.governance.evidence import HashChain
from htce_origin.control.homeostasis import ActiveInferenceSurrogate, HomeostaticEvaluation, HomeostaticState, ControlSignal, HomeostaticActionEffect, MAX_BP, clamp_bp, require_bp
from htce_origin.governance.policy import DecisionKind
from htce_origin.governance.proof import ProofObject, Statement, TheoremLayer, normalize_statement


class PlannerError(ValueError):
    """Raised when the planner receives an invalid simulated-action contract."""


@dataclass(frozen=True)
class PlanStep:
    """One simulation-only plan step."""

    action_name: str
    simulated: bool = True

    def __post_init__(self) -> None:
        if not str(self.action_name).strip():
            raise PlannerError("action_name must be non-empty")
        if not self.simulated:
            raise PlannerError("simulation planner PlanStep accepts simulated steps only")


@dataclass(frozen=True)
class PlanResult:
    """Bounded planner result.

    The result is a recommendation. It does not execute or authorize a real
    action and it does not commit facts to memory.
    """

    steps: tuple[PlanStep, ...]
    reason: str
    decision: DecisionKind = DecisionKind.ACT_SIMULATED
    viability_bp: int = 10000
    expected_free_energy_bp: int = 0
    allow_simulated_action: bool = True
    habitat_policy_allowed: bool = True
    habitat_gate_reason: str = "simulation habitat gate not evaluated"
    explore_simulation: bool = False
    rollout_score_bp: int = 0
    domain_randomized: bool = False
    domain_count: int = 0
    worst_domain_seed: str | None = None
    proof_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class SimulatedSkill:
    """Registered simulation-habitat skill.

    ``simulated_only=False`` is allowed only so the planner can prove that real
    actions are blocked at the simulation planner boundary. Such a skill can never produce
    executable steps in this release.
    """

    name: str
    ensures: Statement | str
    steps: tuple[PlanStep, ...] = ()
    simulated_only: bool = True
    metadata: Mapping[str, int | str | bool] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise PlannerError("skill name must be non-empty")
        object.__setattr__(self, "ensures", normalize_statement(self.ensures))
        object.__setattr__(self, "steps", tuple(self.steps))
        if self.simulated_only and not self.steps:
            object.__setattr__(self, "steps", (PlanStep(self.name),))
        if self.simulated_only and any(not step.simulated for step in self.steps):
            raise PlannerError("simulated skill contains a non-simulated step")
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class SkillRegistry:
    """Small in-memory skill registry for simulation-habitat planning."""

    def __init__(self) -> None:
        self._skills: dict[str, SimulatedSkill] = {}

    def register(self, skill: SimulatedSkill) -> SimulatedSkill:
        if skill.name in self._skills:
            raise PlannerError(f"skill already registered: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> SimulatedSkill | None:
        return self._skills.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))


@dataclass(frozen=True)
class RolloutScore:
    """Integer rollout score for a candidate simulated plan."""

    utility_bp: int
    step_count: int
    expected_free_energy_bp: int


@dataclass(frozen=True)
class SimulationDomain:
    """Seeded domain-randomized simulator parameters.

    This structure is not a real sensor model and does not execute actions. It is
    a deterministic set of perturbations used to score simulated plans across
    multiple plausible domains.

    D_s = H(seed) -> (energy_decay, observation_noise, actuator_noise,
    sensor_drop, terrain_resistance, risk_noise, uncertainty_noise).
    """

    seed: str
    domain_hash: str
    energy_decay_bp: int
    observation_noise_bp: int
    actuator_noise_bp: int
    sensor_drop_bp: int
    terrain_resistance_bp: int
    risk_noise_bp: int
    uncertainty_noise_bp: int

    def __post_init__(self) -> None:
        if not self.seed:
            raise PlannerError("domain seed must be non-empty")
        object.__setattr__(self, "energy_decay_bp", require_bp(self.energy_decay_bp, "energy_decay_bp"))
        object.__setattr__(self, "observation_noise_bp", require_bp(self.observation_noise_bp, "observation_noise_bp"))
        object.__setattr__(self, "actuator_noise_bp", require_bp(self.actuator_noise_bp, "actuator_noise_bp"))
        object.__setattr__(self, "sensor_drop_bp", require_bp(self.sensor_drop_bp, "sensor_drop_bp"))
        object.__setattr__(self, "terrain_resistance_bp", require_bp(self.terrain_resistance_bp, "terrain_resistance_bp"))
        object.__setattr__(self, "risk_noise_bp", require_bp(self.risk_noise_bp, "risk_noise_bp"))
        object.__setattr__(self, "uncertainty_noise_bp", require_bp(self.uncertainty_noise_bp, "uncertainty_noise_bp"))

    def as_trace_payload(self) -> dict[str, int | str]:
        """Return a canonical trace payload for this simulated domain."""
        return {
            "seed": self.seed,
            "domain_hash": self.domain_hash,
            "energy_decay_bp": self.energy_decay_bp,
            "observation_noise_bp": self.observation_noise_bp,
            "actuator_noise_bp": self.actuator_noise_bp,
            "sensor_drop_bp": self.sensor_drop_bp,
            "terrain_resistance_bp": self.terrain_resistance_bp,
            "risk_noise_bp": self.risk_noise_bp,
            "uncertainty_noise_bp": self.uncertainty_noise_bp,
        }


@dataclass(frozen=True)
class DomainRollout:
    """One rollout result under a specific sampled simulation domain."""

    domain: SimulationDomain
    final_state: HomeostaticState
    score_bp: int
    expected_free_energy_bp: int
    model_error_bp: int
    step_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_bp", require_bp(self.score_bp, "score_bp"))
        object.__setattr__(self, "expected_free_energy_bp", require_bp(self.expected_free_energy_bp, "expected_free_energy_bp"))
        object.__setattr__(self, "model_error_bp", require_bp(self.model_error_bp, "model_error_bp"))
        if self.step_count < 0:
            raise PlannerError("step_count must be non-negative")


@dataclass(frozen=True)
class RobustRolloutScore:
    """Worst-case score across deterministic domain-randomized rollouts."""

    utility_bp: int
    worst_domain_seed: str
    domain_count: int
    expected_free_energy_bp: int
    step_count: int
    rollouts: tuple[DomainRollout, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "utility_bp", require_bp(self.utility_bp, "utility_bp"))
        object.__setattr__(self, "expected_free_energy_bp", require_bp(self.expected_free_energy_bp, "expected_free_energy_bp"))
        if self.domain_count != len(self.rollouts):
            raise PlannerError("domain_count must match rollout count")


@dataclass(frozen=True)
class CandidatePlan:
    """A named simulated candidate policy π for robust selection."""

    name: str
    steps: tuple[PlanStep, ...]
    goal_progress_bp: int = 8000
    evidence_gap_bp: int = 0
    domain_fragility_bp: int = 0
    stability_bonus_bp: int = 0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PlannerError("candidate plan name must be non-empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise PlannerError("candidate plan must contain at least one simulated step")
        if any(not step.simulated for step in self.steps):
            raise PlannerError("candidate plan accepts simulated steps only")
        object.__setattr__(self, "goal_progress_bp", require_bp(self.goal_progress_bp, "goal_progress_bp"))
        object.__setattr__(self, "evidence_gap_bp", require_bp(self.evidence_gap_bp, "evidence_gap_bp"))
        object.__setattr__(self, "domain_fragility_bp", require_bp(self.domain_fragility_bp, "domain_fragility_bp"))
        object.__setattr__(self, "stability_bonus_bp", require_bp(self.stability_bonus_bp, "stability_bonus_bp"))


@dataclass(frozen=True)
class CandidatePlanScore:
    """Robust and average score for one candidate policy π."""

    plan: CandidatePlan
    robust_score: RobustRolloutScore
    average_score_bp: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "average_score_bp", require_bp(self.average_score_bp, "average_score_bp"))


@dataclass(frozen=True)
class RobustPlanSelection:
    """Selection report implementing π* = argmax_π min_s score(π, D_s)."""

    selected_plan: CandidatePlan
    selected_score: CandidatePlanScore
    candidate_scores: tuple[CandidatePlanScore, ...]
    selected_index: int
    compared_plan_count: int
    shortest_plan_name: str
    best_average_plan_name: str

    def __post_init__(self) -> None:
        if self.compared_plan_count != len(self.candidate_scores):
            raise PlannerError("compared_plan_count must match candidate_scores")
        if self.compared_plan_count < 2:
            raise PlannerError("robust plan selection requires at least two candidates")
        if self.selected_plan != self.selected_score.plan:
            raise PlannerError("selected plan/score mismatch")


class DomainRandomizedSimulatorWorld:
    """Deterministic domain-randomized simulation scorer.

    The simulator evaluates the same simulated action plan against multiple
    seeded domains and returns the pessimistic/worst-case score. It is a
    planning diagnostic only: it never observes real sensors, never commits
    facts, and never authorizes real-world actions.
    """

    DEFAULT_SEEDS: tuple[str, ...] = ("nominal", "noisy", "resistant")

    def __init__(self, surrogate: ActiveInferenceSurrogate | None = None) -> None:
        self.surrogate = surrogate or ActiveInferenceSurrogate()

    def sample_domain(self, seed: str) -> SimulationDomain:
        """Sample D_s deterministically from a seed using SHA-256."""
        seed_text = str(seed)
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return SimulationDomain(
            seed=seed_text,
            domain_hash=digest,
            energy_decay_bp=_bounded_hash_bp(seed_text, "energy_decay", 100, 900),
            observation_noise_bp=_bounded_hash_bp(seed_text, "observation_noise", 0, 1500),
            actuator_noise_bp=_bounded_hash_bp(seed_text, "actuator_noise", 0, 1200),
            sensor_drop_bp=_bounded_hash_bp(seed_text, "sensor_drop", 0, 1000),
            terrain_resistance_bp=_bounded_hash_bp(seed_text, "terrain_resistance", 0, 1500),
            risk_noise_bp=_bounded_hash_bp(seed_text, "risk_noise", 0, 1500),
            uncertainty_noise_bp=_bounded_hash_bp(seed_text, "uncertainty_noise", 0, 1500),
        )

    def rollout(
        self,
        actions: Iterable[PlanStep],
        seed: str,
        *,
        initial_state: HomeostaticState | None = None,
        goal_progress_bp: int = 8000,
        evidence_gap_bp: int = 0,
    ) -> DomainRollout:
        """Roll out simulated actions under one deterministic domain.

        h_hat_{t+1} = W(h_t, a_t, D_s) is implemented as deterministic updates
        to the homeostatic control vector. This is not a sensor model and not a
        world-fact generator.
        """
        step_tuple = tuple(actions)
        for step in step_tuple:
            if not step.simulated:
                raise PlannerError("domain-randomized rollout accepts simulated steps only")
        domain = self.sample_domain(seed)
        state = initial_state or HomeostaticState()
        if not step_tuple:
            efe = self.surrogate.expected_free_energy_bp(state, evidence_gap_bp=evidence_gap_bp)
            return DomainRollout(domain, state, 0, efe, 0, 0)

        target_goal_progress = require_bp(goal_progress_bp, "goal_progress_bp")
        evidence_gap = require_bp(evidence_gap_bp, "evidence_gap_bp")
        per_step_goal = target_goal_progress // len(step_tuple)
        total_goal_progress = 0
        total_model_error = 0
        total_complexity = 0
        for index, step in enumerate(step_tuple):
            action_cost = _bounded_hash_bp(domain.seed, f"action:{index}:{step.action_name}", 0, 500)
            model_error = clamp_bp((
                domain.observation_noise_bp
                + domain.actuator_noise_bp
                + domain.sensor_drop_bp
                + domain.terrain_resistance_bp
            ) // 4)
            complexity = clamp_bp(((index + 1) * 100) + (domain.terrain_resistance_bp // 4) + (action_cost // 2))
            step_goal_progress = clamp_bp(max(0, per_step_goal - (domain.sensor_drop_bp // 8) - (action_cost // 10)))
            effect = HomeostaticActionEffect(
                energy_delta_bp=-(domain.energy_decay_bp + (domain.terrain_resistance_bp // 4) + (action_cost // 4)),
                risk_delta_bp=domain.risk_noise_bp + (domain.actuator_noise_bp // 2) + (action_cost // 4),
                uncertainty_delta_bp=domain.uncertainty_noise_bp + (domain.observation_noise_bp // 2) + (domain.sensor_drop_bp // 2),
                novelty_delta_bp=max(0, (domain.observation_noise_bp // 4) - (domain.sensor_drop_bp // 8)),
                model_error_bp=model_error,
                evidence_gap_bp=evidence_gap,
                complexity_bp=complexity,
                goal_progress_bp=step_goal_progress,
                novelty_gain_bp=domain.observation_noise_bp // 8,
            )
            state = self.surrogate.apply_action_effect(state, effect)
            total_goal_progress += step_goal_progress
            total_model_error += model_error
            total_complexity += complexity

        assert len(step_tuple) > 0
        average_model_error = clamp_bp(total_model_error // len(step_tuple))
        average_complexity = clamp_bp(total_complexity // len(step_tuple))
        goal_progress = clamp_bp(total_goal_progress)
        deviation = self.surrogate.homeostatic_deviation_bp(state)
        efe = self.surrogate.expected_free_energy_bp(
            state,
            evidence_gap_bp=evidence_gap,
            model_error_bp=average_model_error,
            complexity_bp=average_complexity,
            goal_progress_bp=goal_progress,
            novelty_gain_bp=state.novelty_bp,
        )
        length_penalty = min(2000, len(step_tuple) * 100)
        adverse = state.risk_bp + state.uncertainty_bp + average_model_error + deviation + evidence_gap + length_penalty
        score = clamp_bp(goal_progress - (adverse // 5))
        return DomainRollout(domain, state, score, efe, average_model_error, len(step_tuple))

    def robust_rollout(
        self,
        actions: Iterable[PlanStep],
        seeds: Iterable[str] = (),
        *,
        initial_state: HomeostaticState | None = None,
        goal_progress_bp: int = 8000,
        evidence_gap_bp: int = 0,
    ) -> RobustRolloutScore:
        """Return RobustScore(pi) = min_s score(pi, D_s)."""
        seed_tuple = tuple(str(seed) for seed in seeds) or self.DEFAULT_SEEDS
        rollouts = tuple(
            self.rollout(
                actions,
                seed,
                initial_state=initial_state,
                goal_progress_bp=goal_progress_bp,
                evidence_gap_bp=evidence_gap_bp,
            )
            for seed in seed_tuple
        )
        worst = min(rollouts, key=lambda row: row.score_bp)
        return RobustRolloutScore(
            utility_bp=worst.score_bp,
            worst_domain_seed=worst.domain.seed,
            domain_count=len(rollouts),
            expected_free_energy_bp=worst.expected_free_energy_bp,
            step_count=worst.step_count,
            rollouts=rollouts,
        )

    def score_candidate_plan(
        self,
        plan: CandidatePlan,
        seeds: Iterable[str] = (),
        *,
        initial_state: HomeostaticState | None = None,
    ) -> CandidatePlanScore:
        """Score a candidate policy π across sampled domains.

        Domain fragility is an explicit simulation-only penalty that makes
        optimistic plans lose under noisy/resistant domains. This keeps the
        selector robust rather than merely shortest-plan or average-score driven.
        """
        seed_tuple = tuple(str(seed) for seed in seeds) or self.DEFAULT_SEEDS
        adjusted_rollouts: list[DomainRollout] = []
        for seed in seed_tuple:
            raw = self.rollout(
                plan.steps,
                seed,
                initial_state=initial_state,
                goal_progress_bp=plan.goal_progress_bp,
                evidence_gap_bp=plan.evidence_gap_bp,
            )
            domain_pressure = (
                raw.domain.observation_noise_bp
                + raw.domain.actuator_noise_bp
                + raw.domain.sensor_drop_bp
                + raw.domain.terrain_resistance_bp
                + raw.domain.risk_noise_bp
                + raw.domain.uncertainty_noise_bp
            ) // 6
            fragility_penalty = (domain_pressure * plan.domain_fragility_bp) // MAX_BP
            adjusted_score = clamp_bp(raw.score_bp + plan.stability_bonus_bp - fragility_penalty)
            adjusted_rollouts.append(
                DomainRollout(
                    domain=raw.domain,
                    final_state=raw.final_state,
                    score_bp=adjusted_score,
                    expected_free_energy_bp=raw.expected_free_energy_bp,
                    model_error_bp=raw.model_error_bp,
                    step_count=raw.step_count,
                )
            )
        rollouts = tuple(adjusted_rollouts)
        worst = min(rollouts, key=lambda row: row.score_bp)
        robust = RobustRolloutScore(
            utility_bp=worst.score_bp,
            worst_domain_seed=worst.domain.seed,
            domain_count=len(rollouts),
            expected_free_energy_bp=worst.expected_free_energy_bp,
            step_count=worst.step_count,
            rollouts=rollouts,
        )
        average = sum(row.score_bp for row in rollouts) // len(rollouts)
        return CandidatePlanScore(plan=plan, robust_score=robust, average_score_bp=average)


def _bounded_hash_bp(seed: str, label: str, lower_bp: int, upper_bp: int) -> int:
    """Return deterministic integer in [lower_bp, upper_bp]."""
    lower = require_bp(lower_bp, "lower_bp")
    upper = require_bp(upper_bp, "upper_bp")
    if lower > upper:
        raise PlannerError("lower_bp must be <= upper_bp")
    span = upper - lower + 1
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return lower + (int.from_bytes(digest[:8], "big") % span)


@dataclass(frozen=True)
class HabitatGateInput:
    """Simulation habitat gate input.

    The gate is the mathematical boundary between simulated action planning and
    anything that would look like a real actuator path.  It is intentionally
    integer/bool-only and is evaluated before the planner emits ACT_SIMULATED.

    AllowedSimAction = proof_bp >= tau_proof and topology_bp >= tau_topology and
    model_error_bp <= tau_model_error and policy_ok and trace_ok and
    action_class == "simulated".

    In v0.1 AllowedRealAction is always false, even if a caller sets a real-action
    permission flag elsewhere in the runtime.
    """

    proof_bp: int = 10000
    topology_bp: int = 10000
    model_error_bp: int = 0
    model_error_raw: int = 0
    max_model_error_raw: int | None = None
    proof_ok: bool = True
    topology_ok: bool = True
    policy_ok: bool = True
    trace_ok: bool = True
    action_class: str = "simulated"
    external_sensor_only: bool = True
    signed_manifest: bool = False
    operator_enable: bool = False
    safety_case: bool = False
    dry_run_pass: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "proof_bp", require_bp(self.proof_bp, "proof_bp"))
        object.__setattr__(self, "topology_bp", require_bp(self.topology_bp, "topology_bp"))
        object.__setattr__(self, "model_error_bp", require_bp(self.model_error_bp, "model_error_bp"))
        if int(self.model_error_raw) < 0:
            raise PlannerError("model_error_raw must be non-negative")
        object.__setattr__(self, "model_error_raw", int(self.model_error_raw))
        if self.max_model_error_raw is not None and int(self.max_model_error_raw) < 0:
            raise PlannerError("max_model_error_raw must be non-negative")
        object.__setattr__(self, "max_model_error_raw", None if self.max_model_error_raw is None else int(self.max_model_error_raw))
        action_class = str(self.action_class).strip().lower()
        if action_class not in {"simulated", "real"}:
            raise PlannerError("action_class must be simulated or real")
        object.__setattr__(self, "action_class", action_class)

    def as_trace_payload(self) -> dict[str, bool | int | str]:
        return {
            "action_class": self.action_class,
            "dry_run_pass": self.dry_run_pass,
            "external_sensor_only": self.external_sensor_only,
            "max_model_error_raw": self.max_model_error_raw if self.max_model_error_raw is not None else -1,
            "model_error_bp": self.model_error_bp,
            "model_error_raw": self.model_error_raw,
            "operator_enable": self.operator_enable,
            "policy_ok": self.policy_ok,
            "proof_bp": self.proof_bp,
            "proof_ok": self.proof_ok,
            "safety_case": self.safety_case,
            "signed_manifest": self.signed_manifest,
            "topology_bp": self.topology_bp,
            "topology_ok": self.topology_ok,
            "trace_ok": self.trace_ok,
        }


@dataclass(frozen=True)
class HabitatGateDecision:
    """Decision produced by the simulation habitat policy gate."""

    allowed: bool
    decision: DecisionKind
    reason: str
    allowed_real_action: bool = False

    def as_trace_payload(self) -> dict[str, bool | str]:
        return {
            "allowed": self.allowed,
            "allowed_real_action": self.allowed_real_action,
            "decision": self.decision.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SimulationHabitatPolicy:
    """Gate for simulation-only planning.

    This policy deliberately does not implement signed-manifest real action mode.
    It only records the necessary real-action predicates and returns
    BLOCK_REAL_ACTION for any real actuator path in v0.1.
    """

    min_proof_bp: int = 7000
    min_topology_bp: int = 7000
    max_model_error_bp: int = 3000

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_proof_bp", require_bp(self.min_proof_bp, "min_proof_bp"))
        object.__setattr__(self, "min_topology_bp", require_bp(self.min_topology_bp, "min_topology_bp"))
        object.__setattr__(self, "max_model_error_bp", require_bp(self.max_model_error_bp, "max_model_error_bp"))

    def evaluate(self, gate: HabitatGateInput) -> HabitatGateDecision:
        """Evaluate AllowedSimAction and force AllowedRealAction = 0."""
        if gate.action_class == "real":
            return HabitatGateDecision(
                False,
                DecisionKind.BLOCK_REAL_ACTION,
                "real action blocked: AllowedRealAction=0 in v0.1; signed manifest mode is not implemented",
                allowed_real_action=False,
            )
        if not gate.external_sensor_only:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: external sensor-only boundary violated")
        if not gate.proof_ok or gate.proof_bp < self.min_proof_bp:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: proof confidence below habitat threshold")
        if not gate.topology_ok or gate.topology_bp < self.min_topology_bp:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: topology confidence below habitat threshold")
        if gate.max_model_error_raw is not None:
            if gate.model_error_raw > gate.max_model_error_raw:
                return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: raw model error exceeds habitat threshold")
        elif gate.model_error_bp > self.max_model_error_bp:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: model error exceeds habitat threshold")
        if not gate.policy_ok:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: policy gate failed")
        if not gate.trace_ok:
            return HabitatGateDecision(False, DecisionKind.REFUSE, "simulation blocked: trace gate failed")
        return HabitatGateDecision(True, DecisionKind.ACT_SIMULATED, "simulation habitat gate passed; simulated action only")

    def allowed_real_action(self, gate: HabitatGateInput) -> bool:
        """Return the formal real-action gate. In v0.1 this is hard false."""
        _ = gate
        return False


class ExpectedFreeEnergyScorer:
    """Integer wrapper around the homeostatic EFE surrogate."""

    def __init__(self, surrogate: ActiveInferenceSurrogate | None = None) -> None:
        self.surrogate = surrogate or ActiveInferenceSurrogate()

    def score(
        self,
        state: HomeostaticState,
        *,
        goal_gap_bp: int = 0,
        evidence_gap_bp: int = 0,
    ) -> int:
        return self.surrogate.expected_free_energy_bp(
            state,
            goal_gap_bp=require_bp(goal_gap_bp, "goal_gap_bp"),
            evidence_gap_bp=require_bp(evidence_gap_bp, "evidence_gap_bp"),
        )


class RiskAwareUtility:
    """Integer utility helper for ranking simulation-only plan candidates."""

    def utility_bp(
        self,
        *,
        goal_progress_bp: int,
        risk_bp: int,
        uncertainty_bp: int,
        homeostatic_deviation_bp: int,
        evidence_gap_bp: int,
    ) -> int:
        goal_progress = require_bp(goal_progress_bp, "goal_progress_bp")
        risk = require_bp(risk_bp, "risk_bp")
        uncertainty = require_bp(uncertainty_bp, "uncertainty_bp")
        deviation = require_bp(homeostatic_deviation_bp, "homeostatic_deviation_bp")
        evidence_gap = require_bp(evidence_gap_bp, "evidence_gap_bp")
        penalty = (risk + uncertainty + deviation + evidence_gap) // 4
        return max(0, goal_progress - penalty)


class ProofGuidedPlanner:
    """simulation planner bounded planner.

    It can run verified simulated skills in a simulation habitat and emit a
    protected trace. It cannot execute real-world actions, cannot fabricate
    proof objects, and cannot commit facts.
    """

    def __init__(self, surrogate: ActiveInferenceSurrogate | None = None) -> None:
        self.surrogate = surrogate or ActiveInferenceSurrogate()
        self.utility = RiskAwareUtility()
        self.simulator = DomainRandomizedSimulatorWorld(self.surrogate)
        self.habitat_policy = SimulationHabitatPolicy()

    # Backward-compatible homeostasis/control layer helper.
    def plan(self, goal: object, *, proof_verified: bool = False) -> PlanResult:
        if not proof_verified:
            return PlanResult(
                steps=(),
                reason="simulated skill requires ENSURES proof before planning",
                decision=DecisionKind.REFUSE,
                allow_simulated_action=False,
            )
        return PlanResult((PlanStep(action_name=str(goal)),), "simulation planner simulated plan candidate only")

    # Backward-compatible homeostasis/control layer helper.
    def plan_with_homeostasis(
        self,
        goal: object,
        state: HomeostaticState,
        *,
        candidate_steps: Iterable[PlanStep] = (),
        goal_gap_bp: int = 0,
        evidence_gap_bp: int = 0,
        proof_verified: bool = True,
        domain_seeds: Iterable[str] = (),
    ) -> PlanResult:
        evaluation = self.surrogate.evaluate(state, goal_gap_bp=goal_gap_bp, evidence_gap_bp=evidence_gap_bp)
        decision = _decision_from_evaluation(evaluation)
        if decision != DecisionKind.ACT_SIMULATED:
            return PlanResult(
                steps=(),
                reason=evaluation.reason,
                decision=decision,
                viability_bp=evaluation.viability_bp,
                expected_free_energy_bp=evaluation.expected_free_energy_bp,
                allow_simulated_action=False,
                explore_simulation=evaluation.explore_simulation,
            )
        if not proof_verified:
            return PlanResult(
                steps=(),
                reason="simulated skill requires ENSURES proof before planning",
                decision=DecisionKind.REFUSE,
                viability_bp=evaluation.viability_bp,
                expected_free_energy_bp=evaluation.expected_free_energy_bp,
                allow_simulated_action=False,
                explore_simulation=evaluation.explore_simulation,
            )
        steps = tuple(candidate_steps) or (PlanStep(action_name=str(goal)),)
        seed_tuple = tuple(domain_seeds)
        if seed_tuple:
            robust = self.score_domain_randomized_rollout(
                steps,
                state,
                seeds=seed_tuple,
                evidence_gap_bp=evidence_gap_bp,
            )
            return PlanResult(
                steps=steps,
                reason=f"{evaluation.reason}; domain-randomized robust rollout scored",
                decision=DecisionKind.ACT_SIMULATED,
                viability_bp=evaluation.viability_bp,
                expected_free_energy_bp=robust.expected_free_energy_bp,
                allow_simulated_action=evaluation.allow_simulated_action,
                explore_simulation=evaluation.explore_simulation,
                rollout_score_bp=robust.utility_bp,
                domain_randomized=True,
                domain_count=robust.domain_count,
                worst_domain_seed=robust.worst_domain_seed,
            )
        rollout = self.score_rollout(steps, state, evaluation=evaluation, evidence_gap_bp=evidence_gap_bp)
        return PlanResult(
            steps=steps,
            reason=evaluation.reason,
            decision=DecisionKind.ACT_SIMULATED,
            viability_bp=evaluation.viability_bp,
            expected_free_energy_bp=evaluation.expected_free_energy_bp,
            allow_simulated_action=evaluation.allow_simulated_action,
            explore_simulation=evaluation.explore_simulation,
            rollout_score_bp=rollout.utility_bp,
        )

    def score_rollout(
        self,
        steps: Iterable[PlanStep],
        state: HomeostaticState,
        *,
        evaluation: HomeostaticEvaluation | None = None,
        goal_progress_bp: int = 8000,
        evidence_gap_bp: int = 0,
    ) -> RolloutScore:
        step_tuple = tuple(steps)
        eval_result = evaluation or self.surrogate.evaluate(state, evidence_gap_bp=evidence_gap_bp)
        step_penalty = min(2000, len(step_tuple) * 100)
        goal_progress = max(0, require_bp(goal_progress_bp, "goal_progress_bp") - step_penalty)
        utility = self.utility.utility_bp(
            goal_progress_bp=goal_progress,
            risk_bp=state.risk_bp,
            uncertainty_bp=state.uncertainty_bp,
            homeostatic_deviation_bp=10000 - state.viability_bp(),
            evidence_gap_bp=require_bp(evidence_gap_bp, "evidence_gap_bp"),
        )
        return RolloutScore(
            utility_bp=utility,
            step_count=len(step_tuple),
            expected_free_energy_bp=eval_result.expected_free_energy_bp,
        )

    def score_domain_randomized_rollout(
        self,
        steps: Iterable[PlanStep],
        state: HomeostaticState,
        *,
        seeds: Iterable[str] = (),
        goal_progress_bp: int = 8000,
        evidence_gap_bp: int = 0,
    ) -> RobustRolloutScore:
        """Score a simulated plan under multiple deterministic domains.

        RobustScore(pi) = min_s score(pi, D_s).
        """
        return self.simulator.robust_rollout(
            steps,
            seeds,
            initial_state=state,
            goal_progress_bp=goal_progress_bp,
            evidence_gap_bp=evidence_gap_bp,
        )

    def choose_robust_plan(
        self,
        candidate_plans: Iterable[CandidatePlan],
        state: HomeostaticState,
        *,
        seeds: Iterable[str] = (),
    ) -> RobustPlanSelection:
        """Choose π* = argmax_π min_s score(π, D_s) across candidates."""
        plans = tuple(candidate_plans)
        if len(plans) < 2:
            raise PlannerError("robust plan selection requires at least two candidate plans")
        scores = tuple(self.simulator.score_candidate_plan(plan, seeds, initial_state=state) for plan in plans)
        selected_index, selected_score = max(
            enumerate(scores),
            key=lambda item: (
                item[1].robust_score.utility_bp,
                item[1].average_score_bp,
                -len(item[1].plan.steps),
                item[1].plan.name,
            ),
        )
        shortest_plan = min(plans, key=lambda plan: (len(plan.steps), plan.name))
        best_average = max(scores, key=lambda score: (score.average_score_bp, score.robust_score.utility_bp, score.plan.name))
        return RobustPlanSelection(
            selected_plan=selected_score.plan,
            selected_score=selected_score,
            candidate_scores=scores,
            selected_index=selected_index,
            compared_plan_count=len(scores),
            shortest_plan_name=shortest_plan.name,
            best_average_plan_name=best_average.plan.name,
        )

    def plan_skill(
        self,
        skill_name: str,
        registry: SkillRegistry,
        theorem_layer: TheoremLayer,
        *,
        state: HomeostaticState | None = None,
        goal_gap_bp: int = 0,
        evidence_gap_bp: int = 0,
        allow_real_actions: bool = False,
        trace: HashChain | None = None,
        domain_seeds: Iterable[str] = (),
        habitat_gate_input: HabitatGateInput | None = None,
    ) -> PlanResult:
        skill = registry.get(skill_name)
        body_state = state or HomeostaticState()
        if skill is None:
            return self._with_trace(
                PlanResult((), f"skill is not registered: {skill_name}", DecisionKind.REFUSE, allow_simulated_action=False),
                trace,
                skill_name=skill_name,
                proof=None,
            )
        if not skill.simulated_only:
            real_gate = self.habitat_policy.evaluate(HabitatGateInput(action_class="real"))
            return self._with_trace(
                PlanResult(
                    (),
                    real_gate.reason,
                    real_gate.decision,
                    allow_simulated_action=False,
                    habitat_policy_allowed=real_gate.allowed,
                    habitat_gate_reason=real_gate.reason,
                ),
                trace,
                skill_name=skill_name,
                proof=None,
                habitat_gate=HabitatGateInput(action_class="real"),
                habitat_decision=real_gate,
            )
        if allow_real_actions:
            real_gate = self.habitat_policy.evaluate(HabitatGateInput(action_class="real"))
            return self._with_trace(
                PlanResult(
                    (),
                    "real action permission flag ignored: simulation planner is simulation-only; " + real_gate.reason,
                    DecisionKind.BLOCK_REAL_ACTION,
                    allow_simulated_action=False,
                    habitat_policy_allowed=False,
                    habitat_gate_reason=real_gate.reason,
                ),
                trace,
                skill_name=skill_name,
                proof=None,
                habitat_gate=HabitatGateInput(action_class="real"),
                habitat_decision=real_gate,
            )

        proof = theorem_layer.verify_skill(skill.name)
        if not proof.valid or proof.quarantined:
            return self._with_trace(
                PlanResult(
                    (),
                    f"simulated skill blocked: {proof.reason}",
                    DecisionKind.REFUSE,
                    allow_simulated_action=False,
                    proof_id=proof.proof_id,
                ),
                trace,
                skill_name=skill_name,
                proof=proof,
            )


        gate_input = habitat_gate_input or HabitatGateInput(
            proof_bp=10000,
            topology_bp=int(skill.metadata.get("topology_bp", 10000)) if skill.metadata else 10000,
            model_error_bp=int(skill.metadata.get("model_error_bp", 0)) if skill.metadata else 0,
            policy_ok=bool(skill.metadata.get("policy_ok", True)) if skill.metadata else True,
            trace_ok=trace.verify() if trace is not None else True,
            action_class="simulated",
            external_sensor_only=bool(skill.metadata.get("external_sensor_only", True)) if skill.metadata else True,
        )
        gate_decision = self.habitat_policy.evaluate(gate_input)
        if not gate_decision.allowed:
            return self._with_trace(
                PlanResult(
                    (),
                    gate_decision.reason,
                    gate_decision.decision,
                    allow_simulated_action=False,
                    habitat_policy_allowed=gate_decision.allowed,
                    habitat_gate_reason=gate_decision.reason,
                    proof_id=proof.proof_id,
                ),
                trace,
                skill_name=skill_name,
                proof=proof,
                habitat_gate=gate_input,
                habitat_decision=gate_decision,
            )
        result = self.plan_with_homeostasis(
            skill.name,
            body_state,
            candidate_steps=skill.steps,
            goal_gap_bp=goal_gap_bp,
            evidence_gap_bp=evidence_gap_bp,
            proof_verified=True,
            domain_seeds=domain_seeds,
        )
        result = PlanResult(
            steps=result.steps,
            reason=result.reason,
            decision=result.decision,
            viability_bp=result.viability_bp,
            expected_free_energy_bp=result.expected_free_energy_bp,
            allow_simulated_action=result.allow_simulated_action,
            habitat_policy_allowed=gate_decision.allowed,
            habitat_gate_reason=gate_decision.reason,
            explore_simulation=result.explore_simulation,
            rollout_score_bp=result.rollout_score_bp,
            domain_randomized=result.domain_randomized,
            domain_count=result.domain_count,
            worst_domain_seed=result.worst_domain_seed,
            proof_id=proof.proof_id,
            trace_id=result.trace_id,
        )
        return self._with_trace(
            result,
            trace,
            skill_name=skill_name,
            proof=proof,
            habitat_gate=gate_input,
            habitat_decision=gate_decision,
        )

    def _with_trace(
        self,
        result: PlanResult,
        trace: HashChain | None,
        *,
        skill_name: str,
        proof: ProofObject | None,
        habitat_gate: HabitatGateInput | None = None,
        habitat_decision: HabitatGateDecision | None = None,
    ) -> PlanResult:
        if trace is None:
            return result
        event = trace.append("plan_decision", {
            "decision": result.decision.value,
            "domain_count": result.domain_count,
            "domain_randomized": result.domain_randomized,
            "expected_free_energy_bp": result.expected_free_energy_bp,
            "habitat_gate": habitat_gate.as_trace_payload() if habitat_gate is not None else {},
            "habitat_policy": habitat_decision.as_trace_payload() if habitat_decision is not None else {"allowed": result.habitat_policy_allowed, "reason": result.habitat_gate_reason},
            "habitat_policy_allowed": result.habitat_policy_allowed,
            "proof_id": proof.proof_id if proof is not None else result.proof_id,
            "reason": result.reason,
            "rollout_score_bp": result.rollout_score_bp,
            "simulated_only": True,
            "skill_name": skill_name,
            "release_line": "final_math",
            "step_count": len(result.steps),
            "viability_bp": result.viability_bp,
            "worst_domain_seed": result.worst_domain_seed or "",
        })
        return PlanResult(
            steps=result.steps,
            reason=result.reason,
            decision=result.decision,
            viability_bp=result.viability_bp,
            expected_free_energy_bp=result.expected_free_energy_bp,
            allow_simulated_action=result.allow_simulated_action,
            habitat_policy_allowed=result.habitat_policy_allowed,
            habitat_gate_reason=result.habitat_gate_reason,
            explore_simulation=result.explore_simulation,
            rollout_score_bp=result.rollout_score_bp,
            domain_randomized=result.domain_randomized,
            domain_count=result.domain_count,
            worst_domain_seed=result.worst_domain_seed,
            proof_id=result.proof_id,
            trace_id=event.event_hash(),
        )


def _decision_from_evaluation(evaluation: HomeostaticEvaluation) -> DecisionKind:
    if evaluation.signal == ControlSignal.ASK_CLARIFICATION:
        return DecisionKind.ASK_CLARIFICATION
    if evaluation.signal == ControlSignal.SLEEP_REQUIRED:
        return DecisionKind.SLEEP_REQUIRED
    if evaluation.signal == ControlSignal.BLOCK_SIMULATED_ACTION:
        return DecisionKind.REFUSE
    return DecisionKind.ACT_SIMULATED


__all__ = [
    "CandidatePlan",
    "CandidatePlanScore",
    "DomainRandomizedSimulatorWorld",
    "DomainRollout",
    "ExpectedFreeEnergyScorer",
    "HabitatGateDecision",
    "HabitatGateInput",
    "PlanResult",
    "PlanStep",
    "PlannerError",
    "ProofGuidedPlanner",
    "RiskAwareUtility",
    "RobustPlanSelection",
    "RobustRolloutScore",
    "RolloutScore",
    "SimulationDomain",
    "SimulationHabitatPolicy",
    "SimulatedSkill",
    "SkillRegistry",
]
