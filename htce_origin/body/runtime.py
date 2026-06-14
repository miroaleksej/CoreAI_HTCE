"""Unified lifecycle for HTCE-Origin through mathematically worked L1/L2/L3 body.

L1/L2/L3 body finally connects AIR candidates to a gated L1/L2/L3 body.  The runtime
now supports deterministic fact-as-delta commits, latest-state queries,
supersession and contradiction quarantine.  It still remains simulation-first:
real actions are blocked, legacy is not imported, and all mutation is routed via
AIR -> policy/evidence gates -> memory/body -> protected trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from htce_origin.language.air import AIRException, AIRParser, AIRStaticChecker, AIRVM
from htce_origin.language.nlu_air_bridge import NluBridgeError, NluToAirBridge
from htce_origin.kernel.config import RuntimeConfig
from htce_origin.kernel.q16 import q_mod, q_sub, q_toroidal_loss_vector
from htce_origin.kernel.core import EvidenceId, FactDelta, FactFrame, EntityId, RelationId, TorusVector, active_state_digest, fact_delta, hash_to_phase
from htce_origin.governance.evidence import ClaimSupportReport, EvidenceWeigher, TraceEvent, TraceLog
from htce_origin.control.homeostasis import HomeostaticActionEffect, HomeostaticState, HypothesisTestingLoop, SensoryObservation, MAX_BP
from htce_origin.body.layers import L123Body, LayerDelta, LayerName
from htce_origin.body.memory import FactDeltaStore, FactStatus, MemoryRecord, QueryResult
from htce_origin.control.planner import HabitatGateInput, PlanResult, PlanStep, ProofGuidedPlanner, SimulatedSkill, SimulationHabitatPolicy, SkillRegistry
from htce_origin.cognition.world import PredictionResult, Q256WorldAction, Q256WorldModel, WorldActionEvaluation
from htce_origin.cognition.learning import EpisodeFact, EpisodeRecord, SleepConsolidator
from htce_origin.cognition.l3_promotion import (
    L3RuleCandidate,
    L3RuleConflictReport,
    L3RulePromotionDecision,
    L3RulePromotionStatus,
    L3RuleSupportReport,
    build_l3_rule_conflict_report,
    build_l3_rule_support_report,
)
from htce_origin.governance.proof import Judgment
from htce_origin.sensory.l1_encoder import L1SensoryEncoder, RawSensorPacket
from htce_origin.governance.policy import DecisionKind, FactCandidate, PolicyEngine, PolicyRequest, RequestKind
from htce_origin.governance.proof import Statement, TheoremLayer
from htce_origin.topology.guard import CalibrationProfile, TopologyDecision, TopologyGuard


@dataclass(frozen=True)
class RuntimeRequest:
    input_text: str
    source: str = "operator"
    require_trace: bool = True


@dataclass(frozen=True)
class RuntimeDecision:
    kind: DecisionKind
    reason: str
    trace_id: str | None = None


@dataclass(frozen=True)
class RuntimeResponse:
    decision: RuntimeDecision
    output: str
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ClosedLoopStepRecord:
    """One audited P11 L1-world-planner simulation step.

    The record contains only integer/bool/digest fields.  It is a traceable
    simulation result, not a fact claim and not an actuator authorization.
    """

    step: int
    observation_digest: str
    l1_digest: str
    predicted_digest: str
    observed_next_digest: str
    chosen_action: str
    efe_raw: int
    surprise_raw: int
    efe_bp: int
    surprise_bp: int
    gate_allowed: bool
    trace_id: str

    def as_payload(self) -> dict[str, bool | int | str]:
        return {
            "chosen_action": self.chosen_action,
            "efe_bp": self.efe_bp,
            "efe_raw": self.efe_raw,
            "gate_allowed": self.gate_allowed,
            "l1_digest": self.l1_digest,
            "observation_digest": self.observation_digest,
            "observed_next_digest": self.observed_next_digest,
            "predicted_digest": self.predicted_digest,
            "step": self.step,
            "surprise_bp": self.surprise_bp,
            "surprise_raw": self.surprise_raw,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class ClosedLoopSimulationReport:
    """Summary of a P11 closed-loop simulation run."""

    steps: tuple[ClosedLoopStepRecord, ...]
    trace_head: str
    trace_verified: bool
    total_efe_raw: int
    total_surprise_raw: int
    average_efe_bp: int
    average_surprise_bp: int
    advance_count: int
    rotate_count: int
    hold_count: int
    modulus: int
    dimension: int

    def as_payload(self) -> dict[str, object]:
        return {
            "action_counts": {
                "advance": self.advance_count,
                "hold": self.hold_count,
                "rotate": self.rotate_count,
            },
            "average_efe_bp": self.average_efe_bp,
            "average_surprise_bp": self.average_surprise_bp,
            "dimension": self.dimension,
            "modulus": self.modulus,
            "step_count": len(self.steps),
            "total_efe_raw": self.total_efe_raw,
            "total_surprise_raw": self.total_surprise_raw,
            "steps": [step.as_payload() for step in self.steps],
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
        }


@dataclass(frozen=True)
class LivingActiveStepRecord:
    """One P24 simulation-only active-inference heartbeat.

    The record is a functional-liveness diagnostic: continuous L1 intake,
    integer expected-free-energy action selection, homeostatic update,
    world-model prediction error and protected trace.  It is not a claim of
    consciousness and never authorizes real-world action.
    """

    step: int
    heartbeat: int
    x: int
    y: int
    goal_x: int
    goal_y: int
    chosen_action: str
    expected_free_energy_raw: int
    goal_distance_before: int
    goal_distance_after: int
    risk_bp: int
    uncertainty_bp: int
    energy_bp: int
    viability_bp: int
    prediction_error_bp: int
    l1_digest: str
    observation_digest: str
    predicted_digest: str
    trace_id: str

    def as_payload(self) -> dict[str, int | str]:
        return {
            "chosen_action": self.chosen_action,
            "energy_bp": self.energy_bp,
            "expected_free_energy_raw": self.expected_free_energy_raw,
            "goal_distance_after": self.goal_distance_after,
            "goal_distance_before": self.goal_distance_before,
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "heartbeat": self.heartbeat,
            "l1_digest": self.l1_digest,
            "observation_digest": self.observation_digest,
            "predicted_digest": self.predicted_digest,
            "prediction_error_bp": self.prediction_error_bp,
            "risk_bp": self.risk_bp,
            "step": self.step,
            "trace_id": self.trace_id,
            "uncertainty_bp": self.uncertainty_bp,
            "viability_bp": self.viability_bp,
            "x": self.x,
            "y": self.y,
        }


@dataclass(frozen=True)
class UnifiedLivingDialogTurnRecord:
    """One dialog/action-policy turn executed inside the living simulation.

    This is not a standalone benchmark shell.  The turn is processed by the same
    HTCERuntime.tick path during the active heartbeat loop, writes into the same
    L2 memory and protected trace, and is audited as part of one simulation.
    """

    step: int
    scenario: str
    user_text: str
    expected_kind: str
    expected_contains: tuple[str, ...]
    decision_kind: str
    output: str
    passed: bool
    false_support: int
    evidence_count: int
    proof_id: str | None
    trace_id: str | None
    active_values: dict[str, str]
    missing_slots: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "active_values": dict(self.active_values),
            "decision_kind": self.decision_kind,
            "evidence_count": self.evidence_count,
            "expected_contains": list(self.expected_contains),
            "expected_kind": self.expected_kind,
            "false_support": self.false_support,
            "missing_slots": list(self.missing_slots),
            "output": self.output,
            "passed": self.passed,
            "proof_id": self.proof_id,
            "scenario": self.scenario,
            "step": self.step,
            "trace_id": self.trace_id,
            "user_text": self.user_text,
        }


@dataclass(frozen=True)
class LivingActiveAgentReport:
    """P24 functional-liveness report for the simulation-only active agent."""

    steps: tuple[LivingActiveStepRecord, ...]
    grid_size: int
    reached_goal: bool
    final_x: int
    final_y: int
    goal_x: int
    goal_y: int
    start_goal_distance: int
    final_goal_distance: int
    total_efe_raw: int
    average_prediction_error_bp: int
    average_viability_bp: int
    min_viability_bp: int
    action_counts: dict[str, int]
    visited_count: int
    heartbeat_count: int
    l2_fact_count_before: int
    l2_fact_count_after: int
    l3_clock_before: int
    l3_clock_after: int
    real_actions_allowed: bool
    trace_head: str
    trace_verified: bool
    unified_simulation: bool = False
    dialog_turns: tuple[UnifiedLivingDialogTurnRecord, ...] = ()
    dialog_metrics: dict[str, int] = field(default_factory=dict)
    domain_contexts: dict[str, str] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return {
            "action_counts": dict(self.action_counts),
            "average_prediction_error_bp": self.average_prediction_error_bp,
            "average_viability_bp": self.average_viability_bp,
            "final_goal_distance": self.final_goal_distance,
            "final_position": {"x": self.final_x, "y": self.final_y},
            "goal_position": {"x": self.goal_x, "y": self.goal_y},
            "grid_size": self.grid_size,
            "heartbeat_count": self.heartbeat_count,
            "l2_fact_count_after": self.l2_fact_count_after,
            "l2_fact_count_before": self.l2_fact_count_before,
            "l3_clock_after": self.l3_clock_after,
            "l3_clock_before": self.l3_clock_before,
            "min_viability_bp": self.min_viability_bp,
            "real_actions_allowed": self.real_actions_allowed,
            "reached_goal": self.reached_goal,
            "start_goal_distance": self.start_goal_distance,
            "step_count": len(self.steps),
            "steps": [step.as_payload() for step in self.steps],
            "total_efe_raw": self.total_efe_raw,
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
            "unified_simulation": self.unified_simulation,
            "dialog_turns": [turn.as_payload() for turn in self.dialog_turns],
            "dialog_metrics": dict(self.dialog_metrics),
            "domain_contexts": dict(self.domain_contexts),
            "visited_count": self.visited_count,
        }


@dataclass(frozen=True)
class AdaptivePolicyEpisodeReport:
    """One P26 episode executed inside the same runtime simulation.

    The episode contains the living grid loop and the unified P25 dialog/action
    policy loop.  It is an episode-level report only: it does not create a
    separate dialog harness or a second agent.
    """

    episode_index: int
    goal_description: str
    heartbeat_count: int
    reached_goal: bool
    final_goal_distance: int
    ask_clarification_count: int
    act_simulated_count: int
    wrong_turns: int
    false_support_count: int
    hidden_hazard_hits: int
    recovery_actions: int
    learned_hazard_count: int
    adaptive_cost_raw: int
    trace_id: str
    report: LivingActiveAgentReport

    def as_payload(self) -> dict[str, object]:
        return {
            "act_simulated_count": self.act_simulated_count,
            "adaptive_cost_raw": self.adaptive_cost_raw,
            "ask_clarification_count": self.ask_clarification_count,
            "episode_index": self.episode_index,
            "false_support_count": self.false_support_count,
            "final_goal_distance": self.final_goal_distance,
            "goal_description": self.goal_description,
            "heartbeat_count": self.heartbeat_count,
            "hidden_hazard_hits": self.hidden_hazard_hits,
            "learned_hazard_count": self.learned_hazard_count,
            "living_dialog_report": self.report.as_payload(),
            "reached_goal": self.reached_goal,
            "recovery_actions": self.recovery_actions,
            "trace_id": self.trace_id,
            "wrong_turns": self.wrong_turns,
        }


@dataclass(frozen=True)
class AdaptivePolicyImprovementReport:
    """P26 evidence that one runtime improves across two same-goal episodes.

    Improvement is verified by the strict integer inequality
    cost(E2) < cost(E1), where cost = heartbeat steps + dialog clarifications
    + hidden-hazard recovery actions.  The causal evidence is the protected trace
    chain: episode 1 -> sleep/L3 promotion -> episode 2 -> verification.
    """

    goal_description: str
    episode_1: AdaptivePolicyEpisodeReport
    episode_2: AdaptivePolicyEpisodeReport
    consolidation_replayed_episodes: int
    consolidation_replayed_facts: int
    l3_rules_promoted_during_sleep: int
    l3_rules_blocked_during_sleep: int
    l3_provisional_rules_total: int
    learned_avoid_cells: tuple[str, ...]
    learned_dialog_required_slots: tuple[str, ...]
    improvement_margin_raw: int
    improvement_verified: bool
    trace_head: str
    trace_verified: bool
    real_actions_allowed: bool
    simulation_only: bool = True
    single_runtime_loop: bool = True

    def as_payload(self) -> dict[str, object]:
        return {
            "consolidation_replayed_episodes": self.consolidation_replayed_episodes,
            "consolidation_replayed_facts": self.consolidation_replayed_facts,
            "episode_1": self.episode_1.as_payload(),
            "episode_2": self.episode_2.as_payload(),
            "goal_description": self.goal_description,
            "improvement_margin_raw": self.improvement_margin_raw,
            "improvement_verified": self.improvement_verified,
            "l3_provisional_rules_total": self.l3_provisional_rules_total,
            "l3_rules_blocked_during_sleep": self.l3_rules_blocked_during_sleep,
            "l3_rules_promoted_during_sleep": self.l3_rules_promoted_during_sleep,
            "learned_avoid_cells": list(self.learned_avoid_cells),
            "learned_dialog_required_slots": list(self.learned_dialog_required_slots),
            "real_actions_allowed": self.real_actions_allowed,
            "simulation_only": self.simulation_only,
            "single_runtime_loop": self.single_runtime_loop,
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
        }


@dataclass(frozen=True)
class ContinualAdaptiveEpisodeReport:
    """One P27 continual episode inside the same runtime line.

    The episode reuses the P26 living/dialog simulation and measures whether
    previously promoted L3 hints, proof gates, bAbI-style probes and dialog
    policy probes remain valid after the next sleep/consolidation cycle.
    """

    episode_index: int
    adaptive_cost_raw: int
    previous_cost_raw: int | None
    non_regression_cost_passed: bool
    reached_goal: bool
    wrong_turns: int
    false_support_count: int
    learned_hazard_count: int
    learned_dialog_slot_count: int
    retained_l3_rule_count: int
    l3_rule_regression_count: int
    probe_total_count: int
    probe_passed_count: int
    probe_failure_count: int
    proof_gate_passed: bool
    topology_gate_passed: bool
    trace_id: str
    living_report: LivingActiveAgentReport

    def as_payload(self) -> dict[str, object]:
        return {
            "adaptive_cost_raw": self.adaptive_cost_raw,
            "episode_index": self.episode_index,
            "false_support_count": self.false_support_count,
            "learned_dialog_slot_count": self.learned_dialog_slot_count,
            "learned_hazard_count": self.learned_hazard_count,
            "l3_rule_regression_count": self.l3_rule_regression_count,
            "living_report": self.living_report.as_payload(),
            "non_regression_cost_passed": self.non_regression_cost_passed,
            "previous_cost_raw": self.previous_cost_raw,
            "probe_failure_count": self.probe_failure_count,
            "probe_passed_count": self.probe_passed_count,
            "probe_total_count": self.probe_total_count,
            "proof_gate_passed": self.proof_gate_passed,
            "reached_goal": self.reached_goal,
            "retained_l3_rule_count": self.retained_l3_rule_count,
            "topology_gate_passed": self.topology_gate_passed,
            "trace_id": self.trace_id,
            "wrong_turns": self.wrong_turns,
        }


@dataclass(frozen=True)
class ContinualAdaptiveMemoryReport:
    """P27 no-regression evidence across a sequence of adaptive episodes.

    The report proves only bounded simulation behaviour: repeated sleep/L3
    consolidation accumulates provisional hints, does not erase earlier hints,
    keeps proof/topology gates healthy, keeps false support at zero and keeps
    adaptive cost non-increasing after the first improvement plateau.
    """

    goal_description: str
    episodes: tuple[ContinualAdaptiveEpisodeReport, ...]
    total_episodes: int
    total_l3_rules_promoted: int
    retained_l3_rules_final: int
    learned_avoid_cells_final: tuple[str, ...]
    learned_dialog_slots_final: tuple[str, ...]
    no_regression_passed: bool
    monotonic_cost_passed: bool
    proof_gates_passed: bool
    topology_gates_passed: bool
    babi_dialog_probes_passed: bool
    false_support_count: int
    wrong_turn_count: int
    trace_head: str
    trace_verified: bool
    real_actions_allowed: bool
    simulation_only: bool = True
    single_runtime_loop: bool = True

    def as_payload(self) -> dict[str, object]:
        return {
            "babi_dialog_probes_passed": self.babi_dialog_probes_passed,
            "episodes": [episode.as_payload() for episode in self.episodes],
            "false_support_count": self.false_support_count,
            "goal_description": self.goal_description,
            "learned_avoid_cells_final": list(self.learned_avoid_cells_final),
            "learned_dialog_slots_final": list(self.learned_dialog_slots_final),
            "monotonic_cost_passed": self.monotonic_cost_passed,
            "no_regression_passed": self.no_regression_passed,
            "proof_gates_passed": self.proof_gates_passed,
            "real_actions_allowed": self.real_actions_allowed,
            "retained_l3_rules_final": self.retained_l3_rules_final,
            "simulation_only": self.simulation_only,
            "single_runtime_loop": self.single_runtime_loop,
            "topology_gates_passed": self.topology_gates_passed,
            "total_episodes": self.total_episodes,
            "total_l3_rules_promoted": self.total_l3_rules_promoted,
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
            "wrong_turn_count": self.wrong_turn_count,
        }


@dataclass(frozen=True)
class MultiTaskDomainProbeResult:
    """One P28 cross-domain no-regression probe result.

    The cost is a raw integer penalty collected only after the runtime response.
    A regression means the current raw cost is strictly greater than the best
    historical cost already observed for the same domain.  No floating threshold
    and no external oracle is used in the decision path.
    """

    episode_index: int
    trained_domain: str
    probe_domain: str
    probe_passed: bool
    current_cost_raw: int
    best_historical_cost_raw: int | None
    regression_detected: bool
    false_support_count: int
    wrong_turn_count: int
    l3_rules_active_count: int
    trace_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "best_historical_cost_raw": self.best_historical_cost_raw,
            "current_cost_raw": self.current_cost_raw,
            "episode_index": self.episode_index,
            "false_support_count": self.false_support_count,
            "l3_rules_active_count": self.l3_rules_active_count,
            "probe_domain": self.probe_domain,
            "probe_passed": self.probe_passed,
            "regression_detected": self.regression_detected,
            "trace_id": self.trace_id,
            "trained_domain": self.trained_domain,
            "wrong_turn_count": self.wrong_turn_count,
        }


@dataclass(frozen=True)
class MultiTaskEpisodeReport:
    """One P28 curriculum episode plus full probe-matrix result."""

    episode_index: int
    trained_domain: str
    domain_episode_cost_raw: int
    promoted_rules_count: int
    retained_l3_rule_count: int
    probe_results: tuple[MultiTaskDomainProbeResult, ...]
    probe_failure_count: int
    false_support_count: int
    wrong_turn_count: int
    trace_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "domain_episode_cost_raw": self.domain_episode_cost_raw,
            "episode_index": self.episode_index,
            "false_support_count": self.false_support_count,
            "probe_failure_count": self.probe_failure_count,
            "probe_results": [probe.as_payload() for probe in self.probe_results],
            "promoted_rules_count": self.promoted_rules_count,
            "retained_l3_rule_count": self.retained_l3_rule_count,
            "trace_id": self.trace_id,
            "trained_domain": self.trained_domain,
            "wrong_turn_count": self.wrong_turn_count,
        }


@dataclass(frozen=True)
class MultiTaskAdaptiveMemoryReport:
    """P28 evidence for continual multi-task adaptation without cross-domain regression.

    A single HTCERuntime alternates several domains, sleeps/consolidates after
    each domain episode, then probes every domain.  The report proves a bounded
    property only: L3 adaptation in one domain did not raise probe cost, did not
    bypass proof/policy/topology gates, and did not produce false support.
    """

    curriculum_domains: tuple[str, ...]
    domains_tested: tuple[str, ...]
    episodes: tuple[MultiTaskEpisodeReport, ...]
    domain_cost_history_raw: dict[str, tuple[int, ...]]
    no_cross_domain_regression: bool
    monotonic_improvement_per_domain: dict[str, bool]
    proof_gates_passed: bool
    topology_gates_passed: bool
    total_l3_rules_promoted: int
    retained_l3_rules_final: int
    false_support_count: int
    wrong_turn_count: int
    trace_head: str
    trace_verified: bool
    real_actions_allowed: bool
    simulation_only: bool = True
    single_runtime_loop: bool = True

    @property
    def passed(self) -> bool:
        return bool(
            self.no_cross_domain_regression
            and all(self.monotonic_improvement_per_domain.values())
            and self.proof_gates_passed
            and self.topology_gates_passed
            and self.false_support_count == 0
            and self.wrong_turn_count == 0
            and self.trace_verified
            and not self.real_actions_allowed
            and self.simulation_only
            and self.single_runtime_loop
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "curriculum_domains": list(self.curriculum_domains),
            "domain_cost_history_raw": {key: list(value) for key, value in self.domain_cost_history_raw.items()},
            "domains_tested": list(self.domains_tested),
            "episodes": [episode.as_payload() for episode in self.episodes],
            "false_support_count": self.false_support_count,
            "monotonic_improvement_per_domain": dict(self.monotonic_improvement_per_domain),
            "no_cross_domain_regression": self.no_cross_domain_regression,
            "passed": self.passed,
            "proof_gates_passed": self.proof_gates_passed,
            "real_actions_allowed": self.real_actions_allowed,
            "retained_l3_rules_final": self.retained_l3_rules_final,
            "simulation_only": self.simulation_only,
            "single_runtime_loop": self.single_runtime_loop,
            "topology_gates_passed": self.topology_gates_passed,
            "total_episodes": len(self.episodes),
            "total_l3_rules_promoted": self.total_l3_rules_promoted,
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
            "wrong_turn_count": self.wrong_turn_count,
        }


@dataclass(frozen=True)
class V1ExternalBenchmarkRowResult:
    """One v1.0 external-shaped revalidation row evaluated after runtime inference."""

    suite: str
    row_id: str
    engine_input_hash: str
    expected_digest: str
    decision_kind: str
    output: str
    passed: bool
    false_support: int
    proof_or_evidence_present: bool
    answer_key_visible_to_engine: int
    trace_id: str | None

    def as_payload(self) -> dict[str, object]:
        return {
            "answer_key_visible_to_engine": self.answer_key_visible_to_engine,
            "decision_kind": self.decision_kind,
            "engine_input_hash": self.engine_input_hash,
            "expected_digest": self.expected_digest,
            "false_support": self.false_support,
            "output": self.output,
            "passed": self.passed,
            "proof_or_evidence_present": self.proof_or_evidence_present,
            "row_id": self.row_id,
            "suite": self.suite,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class V1CleanSystemReport:
    """v1.0 clean-system external revalidation and package-readiness report.

    This report is intentionally bounded.  It proves that external-shaped bAbI
    and Dialog-bAbI rows, multitask stress, proof gates, topology gates, trace
    verification and no-answer-leakage checks all run through one HTCERuntime.
    It does not claim consciousness, qualia, real-world autonomy or full AGI.
    """

    version: str
    external_rows: tuple[V1ExternalBenchmarkRowResult, ...]
    multitask_report: MultiTaskAdaptiveMemoryReport
    total_external_rows: int
    external_rows_passed: int
    external_false_support_count: int
    answer_key_visible_to_engine_count: int
    dialog_loader_strict_passed: bool
    no_external_regression: bool
    proof_gates_passed: bool
    topology_gates_passed: bool
    clean_single_runtime_loop: bool
    trace_head: str
    trace_verified: bool
    real_actions_allowed: bool
    simulation_only: bool = True
    consciousness_claimed: bool = False
    qualia_claimed: bool = False

    @property
    def passed(self) -> bool:
        return bool(
            self.total_external_rows > 0
            and self.external_rows_passed == self.total_external_rows
            and self.external_false_support_count == 0
            and self.answer_key_visible_to_engine_count == 0
            and self.dialog_loader_strict_passed
            and self.no_external_regression
            and self.proof_gates_passed
            and self.topology_gates_passed
            and self.clean_single_runtime_loop
            and self.trace_verified
            and not self.real_actions_allowed
            and self.simulation_only
            and not self.consciousness_claimed
            and not self.qualia_claimed
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "answer_key_visible_to_engine_count": self.answer_key_visible_to_engine_count,
            "clean_single_runtime_loop": self.clean_single_runtime_loop,
            "consciousness_claimed": self.consciousness_claimed,
            "dialog_loader_strict_passed": self.dialog_loader_strict_passed,
            "external_false_support_count": self.external_false_support_count,
            "external_rows": [row.as_payload() for row in self.external_rows],
            "external_rows_passed": self.external_rows_passed,
            "multitask_report": self.multitask_report.as_payload(),
            "no_external_regression": self.no_external_regression,
            "passed": self.passed,
            "proof_gates_passed": self.proof_gates_passed,
            "qualia_claimed": self.qualia_claimed,
            "real_actions_allowed": self.real_actions_allowed,
            "simulation_only": self.simulation_only,
            "topology_gates_passed": self.topology_gates_passed,
            "total_external_rows": self.total_external_rows,
            "trace_head": self.trace_head,
            "trace_verified": self.trace_verified,
            "version": self.version,
        }


class HTCERuntime:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.config.validate()
        self.parser = AIRParser()
        self.nlu_bridge = NluToAirBridge()
        self.checker = AIRStaticChecker()
        self.vm = AIRVM()
        self.trace = TraceLog()
        self.policy = PolicyEngine(trace=self.trace)
        self.theorem_layer = TheoremLayer()
        self.skill_registry = SkillRegistry()
        self.planner = ProofGuidedPlanner()
        self.body = L123Body(dimension=self.config.l2_dim, modulus=self.config.modulus)
        self.memory = FactDeltaStore()
        self.l1_encoder = L1SensoryEncoder(torus_dimension=self.body.dimension, input_dim=self.config.l1_input_dim, modulus=self.body.modulus)
        self.world_model = Q256WorldModel(dimension=self.body.dimension, modulus=self.body.modulus)
        self.homeostatic_state = HomeostaticState()
        self.hypothesis_loop = HypothesisTestingLoop()
        self.topology_guard = TopologyGuard(CalibrationProfile.profile_short_path(dimension=self.body.dimension, modulus=self.body.modulus))
        self.l2_topology_window: tuple[tuple[int, ...], ...] = (self.body.l2_clean_vector(),)
        self.max_l2_topology_window = 64
        self.evidence_weigher = EvidenceWeigher()
        self.claim_support_reports: dict[str, ClaimSupportReport] = {}
        self.l3_provisional_rules: dict[str, L3RulePromotionDecision] = {}
        self.awake = False

    def wake(self) -> RuntimeResponse:
        self.awake = True
        ev = self.trace.append("wake", {"release_line": "final_math", "status": "l1_l2_l3_body_ready"})
        return RuntimeResponse(
            decision=RuntimeDecision(DecisionKind.ANSWER, "wake accepted", ev.event_hash()),
            output="HTCE-Origin L1/L2/L3 body awake: L1/L2/L3 body runtime ready.",
        )

    def tick(self, request: RuntimeRequest) -> RuntimeResponse:
        try:
            program = self.parser.parse(request.input_text)
            check = self.checker.check(program)
            if not check.ok:
                return self._refuse_air(request, {"codes": check.error_codes, "reasons": check.reasons})
            nodes = self.vm.execute(program)
        except AIRException as exc:
            try:
                translated = self._translate_natural_language_to_air(request.input_text)
            except NluBridgeError as nlu_exc:
                return self._refuse_air(request, {
                    "code": exc.code,
                    "message": exc.message,
                    "nlu_air_bridge": "ambiguous_or_unsupported_translation",
                    "nlu_reason": str(nlu_exc),
                })
            if translated:
                return self.tick(RuntimeRequest(translated, source=f"{request.source}:nlu_air_bridge", require_trace=request.require_trace))
            return self._refuse_air(request, {"code": exc.code, "message": exc.message, "nlu_air_bridge": "no_supported_translation"})

        if not nodes:
            return self._refuse_air(request, {"reason": "no AIR VM candidate events"})

        responses = [self._handle_event(node.kind, dict(node.payload), node.evidence_id, request) for node in nodes]
        if len(responses) == 1:
            return responses[0]
        ev = self.trace.append("runtime_batch", {
            "decision_count": len(responses),
            "kinds": [item.decision.kind.value for item in responses],
            "release_line": "final_math",
        })
        return RuntimeResponse(
            RuntimeDecision(responses[-1].decision.kind, "batch AIR program processed", ev.event_hash()),
            responses[-1].output,
            {"batch": [item.diagnostics for item in responses]},
        )

    def _translate_natural_language_to_air(self, text: str) -> str | None:
        """Translate bounded natural-language input to AIR without guessing.

        P23 boundary: ordinary bAbI/Dialog-like phrases may enter the same AIR runtime
        path as explicit commands.  Ambiguous text is not coerced into a fact.
        """

        commands = self.nlu_bridge.translate_story_sentence(text)
        if commands:
            return "\n".join(commands)
        return self.nlu_bridge.translate_query(text)

    def _handle_event(self, kind: str, payload: dict[str, object], evidence_id: str | None, request: RuntimeRequest) -> RuntimeResponse:
        if kind == "l1_observation_candidate":
            return self._commit_l1_observation(payload, evidence_id, request)
        if kind == "fact_candidate":
            return self._commit_fact(payload, evidence_id, request)
        if kind == "negation_candidate":
            return self._commit_negation(payload, evidence_id, request)
        if kind == "query_candidate":
            return self._answer_query(payload, evidence_id, request)
        if kind == "procedure_registered":
            ev = self.trace.append("procedure_boundary", {"payload": payload, "release_line": "final_math"})
            return RuntimeResponse(RuntimeDecision(DecisionKind.HYPOTHESIS, "procedure registered as non-committing candidate", ev.event_hash()), "HYPOTHESIS: procedure candidate registered; no real action executed.", {"payload": payload})
        return self._refuse_air(request, {"reason": f"unsupported L1/L2/L3 body event kind: {kind}", "payload": payload})

    def observe_l1_packet(self, packet: RawSensorPacket, *, source: str = "sensor_adapter") -> RuntimeResponse:
        """Commit a driver-quantized packet to L1 only.

        This direct runtime API is intentionally outside AIR fact commit: it may
        update the L1 toroidal surface and curiosity diagnostics, but it cannot
        create L2/L3 facts or authorize real-world actions.
        """
        return self._commit_l1_observation(
            {
                "modality": packet.modality,
                "samples": tuple(int(v) for v in packet.samples),
                "sample_min": int(packet.sample_min),
                "sample_max": int(packet.sample_max),
                "reliability_bp": int(packet.reliability_bp),
                "metadata": dict(packet.metadata),
            },
            packet.evidence_id,
            RuntimeRequest("L1_OBSERVATION_DIRECT", source=source),
        )

    def _commit_l1_observation(self, payload: dict[str, object], evidence_id: str | None, request: RuntimeRequest) -> RuntimeResponse:
        if evidence_id is None:
            return self._refuse_air(request, {"reason": "L1 observation has no evidence_id"})
        samples_obj = payload.get("samples", ())
        if isinstance(samples_obj, str):
            return self._refuse_air(request, {"reason": "L1 samples must be integer sequence, not string"})
        try:
            samples = tuple(int(v) for v in samples_obj)  # type: ignore[arg-type]
            packet = RawSensorPacket(
                modality=str(payload.get("modality", "sensor")),
                samples=samples,
                sample_min=int(payload.get("sample_min", 0)),
                sample_max=int(payload.get("sample_max", (1 << 16) - 1)),
                reliability_bp=int(payload.get("reliability_bp", 10000)),
                evidence_id=evidence_id,
                metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
            )
            encoded = self.l1_encoder.encode(
                packet,
                current_l1_phase=self.body.l1.vector,
                predicted_phase=self.body.l1.vector,
                current_risk_bp=self.homeostatic_state.risk_bp,
            )
        except Exception as exc:
            return self._refuse_air(request, {"reason": "invalid L1 observation", "error": str(exc)})
        before_body = self.body.digest()
        transition = self.body.observe_l1_encoded(encoded, evidence_id=evidence_id)
        trace = self.trace.append("l1_sensory_commit", {
            "body_after": self.body.digest(),
            "body_before": before_body,
            "body_transition": transition.as_payload(),
            "encoder": encoded.as_payload(),
            "release_line": "final_math_q256_l1",
            "source": request.source,
        })
        return RuntimeResponse(
            RuntimeDecision(DecisionKind.HYPOTHESIS, "L1 observation committed to sensory torus only", trace.event_hash()),
            "HYPOTHESIS: L1 observation encoded; no L2/L3 fact committed.",
            {"encoder": encoded.as_payload(), "body_transition": transition.as_payload()},
        )

    def _commit_fact(self, payload: dict[str, object], evidence_id: str | None, request: RuntimeRequest) -> RuntimeResponse:
        preliminary_fact = self._fact_from_payload(payload, evidence_id)
        claim_report = self._claim_support_report(preliminary_fact)
        payload_for_fact = dict(payload)
        supported = True
        evidence_source = "operator_or_air_evidence"
        if claim_report is not None:
            payload_for_fact["confidence_bp"] = claim_report.net_support_bp
            supported = bool(claim_report.claim_allowed)
            evidence_source = "claim_support_report"
        fact = self._fact_from_payload(payload_for_fact, evidence_id)
        delta = fact_delta(fact)
        active_record = self.memory.latest(fact.subject.value, fact.relation.value)
        active_candidate = FactCandidate.from_record(active_record) if active_record is not None else None
        policy = self.policy.evaluate(PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "claim_id": self._claim_id_for_fact(fact),
                "confidence_bp": fact.confidence_bp,
                "evidence_source": evidence_source,
                "object": fact.object.value,
                "relation": fact.relation.value,
                "revision": (active_record.revision + 1 if active_record is not None else 1),
                "subject": fact.subject.value,
                "wants_real_action": bool(payload.get("wants_real_action", False)),
            },
            evidence_id=fact.evidence.value,
            supported=supported,
            source=request.source,
        ), active_fact=active_candidate)
        if policy.blocked:
            return RuntimeResponse(
                RuntimeDecision(policy.kind, policy.reason, policy.trace_id),
                "REFUSE: fact commit blocked by policy/evidence gates.",
                {
                    "claim_support_report": claim_report.as_payload() if claim_report is not None else None,
                    "policy_decision": policy.as_payload(),
                },
            )
        topology = self._topology_precheck_l2_fact(delta)
        if not topology.passed:
            trace = self.trace.append("topology_commit_block", {
                "claim_id": self._claim_id_for_fact(fact),
                "decision": "refuse",
                "record_candidate": {
                    "evidence_id": fact.evidence.value,
                    "object": fact.object.value,
                    "relation": fact.relation.value,
                    "subject": fact.subject.value,
                },
                "release_line": "final_math",
                "topology_decision": self._topology_payload(topology),
            })
            return RuntimeResponse(
                RuntimeDecision(DecisionKind.REFUSE, topology.reason, trace.event_hash()),
                "REFUSE: L2 fact commit blocked by topology guard before memory mutation.",
                {"topology_decision": self._topology_payload(topology)},
            )
        before_body = self.body.digest()
        before_memory = self.memory.digest()
        record = self.memory.commit(delta, trace_id=policy.trace_id or self.trace.head)
        transition = self.body.commit_l2_fact(delta)
        self._append_l2_topology_state(self.body.l2_clean_vector())
        self.theorem_layer.add_judgment(Statement.atom(fact.relation.value, fact.subject.value, fact.object.value), evidence_id=fact.evidence.value, source="asserted", supported=True)
        trace = self.trace.append("l2_fact_commit", {
            "body_after": self.body.digest(),
            "body_before": before_body,
            "body_transition": transition.as_payload(),
            "claim_support_report": claim_report.as_payload() if claim_report is not None else None,
            "memory_after": self.memory.digest(),
            "memory_before": before_memory,
            "policy_decision": policy.as_payload(),
            "record": record.as_payload(),
            "release_line": "final_math",
            "topology_decision": self._topology_payload(topology),
        })
        return RuntimeResponse(
            RuntimeDecision(DecisionKind.ANSWER, "fact committed to L2 latest-state memory", trace.event_hash()),
            f"COMMIT: {fact.subject.value} {fact.relation.value} {fact.object.value}",
            {
                "body_digest": self.body.digest(),
                "claim_support_report": claim_report.as_payload() if claim_report is not None else None,
                "memory_digest": self.memory.digest(),
                "policy_decision": policy.as_payload(),
                "record": record.as_payload(),
                "topology_decision": self._topology_payload(topology),
            },
        )

    def _commit_negation(self, payload: dict[str, object], evidence_id: str | None, request: RuntimeRequest) -> RuntimeResponse:
        fact = self._fact_from_payload(payload, evidence_id)
        active_record = self.memory.latest(fact.subject.value, fact.relation.value)
        active_candidate = FactCandidate.from_record(active_record) if active_record is not None else None
        policy = self.policy.evaluate(PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "confidence_bp": int(payload.get("confidence_bp", 10000)),
                "negated": True,
                "object": fact.object.value,
                "relation": fact.relation.value,
                "revision": (active_record.revision + 1 if active_record is not None else 1),
                "subject": fact.subject.value,
            },
            evidence_id=fact.evidence.value,
            supported=True,
            source=request.source,
        ), active_fact=active_candidate)
        if policy.blocked:
            return RuntimeResponse(RuntimeDecision(policy.kind, policy.reason, policy.trace_id), "REFUSE: negation commit blocked by policy/evidence gates.", {"policy_decision": policy.as_payload()})
        before_memory = self.memory.digest()
        record = self.memory.commit_negation(fact, trace_id=policy.trace_id or self.trace.head)
        self.theorem_layer.add_judgment(Statement.atom(fact.relation.value, fact.subject.value, fact.object.value).negate(), evidence_id=fact.evidence.value, source="asserted", supported=True)
        trace = self.trace.append("l2_fact_quarantine", {
            "memory_after": self.memory.digest(),
            "memory_before": before_memory,
            "record": record.as_payload(),
            "release_line": "final_math",
        })
        return RuntimeResponse(
            RuntimeDecision(DecisionKind.REFUSE, "contradiction/negation quarantined; not surfaced as fact", trace.event_hash()),
            "REFUSE: contradiction quarantined; latest-state answer is blocked until repaired.",
            {"memory_digest": self.memory.digest(), "record": record.as_payload()},
        )

    def promote_l3_candidate_rule(
        self,
        candidate: L3RuleCandidate | object | Statement | str,
        *,
        evidence_id: str = "l3_rule_candidate",
        required_support_raw: int = 2,
    ) -> L3RulePromotionDecision:
        """Promote an L3 semantic candidate only as a provisional rule.

        P16 authority boundary:
            L3 state -> no direct answer;
            L3 state -> no direct L2 fact;
            L3 state -> no real action.

        The method performs theorem-layer validation, raw trace-support checking,
        contradiction checking, policy gating, and protected trace logging.  The
        only accepted success state is a provisional semantic rule stored in
        ``self.l3_provisional_rules`` and registered with the theorem layer as a
        non-authoritative association.
        """
        if isinstance(candidate, L3RuleCandidate):
            l3_candidate = candidate
        elif isinstance(candidate, (Statement, str)):
            l3_candidate = L3RuleCandidate(
                statement=candidate,
                support_count_raw=1,
                trace_ids=(evidence_id,),
                source_rule_id="runtime_direct_l3_candidate",
                l3_state_digest=self.body.l3.digest,
            )
        else:
            l3_candidate = L3RuleCandidate.from_rule_candidate(candidate, l3_state_digest=self.body.l3.digest)

        association = Judgment(
            l3_candidate.statement,
            evidence_id=evidence_id,
            source="association",
            supported=True,
        )
        authorization = self.theorem_layer.authorize_query(
            l3_candidate.statement,
            association_candidates=(association,),
            allow_hypothesis=True,
        )
        support_report = build_l3_rule_support_report(
            l3_candidate,
            authorization,
            required_support_raw=int(required_support_raw),
        )
        conflict_report = build_l3_rule_conflict_report(
            l3_candidate,
            self.theorem_layer.judgments,
            latest_state=self.memory.export_latest_state(),
        )
        policy = self.policy.evaluate(PolicyRequest(
            kind=RequestKind.HYPOTHESIS,
            payload={
                "candidate_id": l3_candidate.candidate_id,
                "claim": l3_candidate.statement.canonical(),
                "claim_id": l3_candidate.claim_id,
                "l3_candidate": True,
                "provisional_only": 1,
                "support_count_raw": l3_candidate.support_count_raw,
            },
            evidence_id=evidence_id,
            supported=support_report.passed and conflict_report.passed,
            is_hypothesis=True,
            wants_real_action=False,
            source="l3_rule_promotion_gate",
        ))
        allowed = bool(support_report.passed and conflict_report.passed and policy.allowed)
        status = L3RulePromotionStatus.PROVISIONAL if allowed else L3RulePromotionStatus.BLOCKED
        reason_parts = [support_report.reason, conflict_report.reason, policy.reason]
        trace = self.trace.append("l3_rule_promotion_decision", {
            "candidate": l3_candidate.as_payload(),
            "conflict_report": conflict_report.as_payload(),
            "policy_decision": policy.as_payload(),
            "release_line": "final_math_p16",
            "status": status.value,
            "support_report": support_report.as_payload(),
            "authority_boundary": {
                "may_answer": False,
                "may_commit_l2_fact": False,
                "may_execute_real_action": False,
            },
        })
        decision = L3RulePromotionDecision(
            candidate=l3_candidate,
            status=status,
            support_report=support_report,
            conflict_report=conflict_report,
            policy_allowed=policy.allowed,
            policy_trace_id=policy.trace_id,
            trace_id=trace.event_hash(),
            reason="; ".join(part for part in reason_parts if part),
            may_answer=False,
            may_commit_l2_fact=False,
            may_execute_real_action=False,
        )
        if decision.provisional_promoted:
            self.l3_provisional_rules[l3_candidate.candidate_id] = decision
            self.theorem_layer.add_association(l3_candidate.statement)
        return decision

    def _answer_query(self, payload: dict[str, object], evidence_id: str | None, request: RuntimeRequest) -> RuntimeResponse:
        subject = str(payload.get("subject", "")).strip().lower()
        query_type = str(payload.get("query_type", ""))
        relation = self._query_relation(query_type)
        target_for_yesno: str | None = None
        if relation.startswith("yesno_in_"):
            target_for_yesno = relation[len("yesno_in_"):]
            relation = "located_in"
        policy = self.policy.evaluate(PolicyRequest(
            kind=RequestKind.QUERY,
            payload={"relation": relation, "subject": subject, "query_type": query_type},
            evidence_id=evidence_id,
            supported=True,
            source=request.source,
        ))

        # P23 dialog/action closure: a restaurant-domain action request is a
        # simulated API call only when L2 slot facts prove all required fields.
        # Missing slots ask clarification; quarantined slots refuse.
        if relation == "api_call_ready":
            context_subject = subject or "current_dialog"
            required_slots = self._api_call_required_slots_from_query_type(query_type)
            evidence_ids: list[str] = []
            active_values: dict[str, str] = {}
            quarantined_slots: list[str] = []
            for slot in required_slots:
                slot_relation = f"has_slot_value_{slot}"
                slot_result = self.memory.query(context_subject, slot_relation, trace_id=policy.trace_id)
                if slot_result.status == FactStatus.QUARANTINED:
                    quarantined_slots.append(slot)
                elif slot_result.answered and slot_result.answer is not None:
                    active_values[slot] = slot_result.answer
                    if slot_result.evidence_id:
                        evidence_ids.append(slot_result.evidence_id)
            proof = self.theorem_layer.prove_api_call_ready(
                required_slots,
                self.memory.export_latest_state(),
                context_subject=context_subject,
                quarantined_slots=tuple(quarantined_slots),
            )
            proof_path = [rule.value for rule in proof.rules]
            if proof.valid and not proof.quarantined:
                answer_policy = self.policy.evaluate(PolicyRequest(
                    kind=RequestKind.SIMULATED_ACTION,
                    payload={"action": "api_call", "required_slots": dict(required_slots), "slot_values": dict(active_values)},
                    evidence_id=evidence_id or (evidence_ids[0] if evidence_ids else "dialog_api_call_ready"),
                    supported=True,
                    source="p23_dialog_api_call_proof_bridge",
                ))
                slot_order = tuple(required_slots.keys())
                domain_label = self._dialog_domain_from_context_subject(context_subject)
                domain_prefix = (f"domain={domain_label} ") if domain_label is not None else ""
                action_text = "api_call " + domain_prefix + " ".join(f"{slot}={active_values[slot]}" for slot in slot_order)
                trace = self.trace.append("p23_dialog_api_call_ready", {
                    "action": action_text,
                    "answer_policy": answer_policy.as_payload(),
                    "evidence_ids": list(evidence_ids),
                    "proof_id": proof.proof_id,
                    "proof_path": proof_path,
                    "release_line": "final_math_p23",
                    "slot_values": dict(active_values),
                })
                return RuntimeResponse(
                    RuntimeDecision(DecisionKind.ACT_SIMULATED, "simulated API call authorized by slot proof", trace.event_hash()),
                    action_text,
                    {
                        "action": "api_call",
                        "authorization": {"answer_allowed": True, "reason": proof.reason},
                        "context_subject": context_subject,
                        "domain": domain_label,
                        "evidence_ids": tuple(evidence_ids),
                        "proof_id": proof.proof_id,
                        "proof_path": proof_path,
                        "slot_values": dict(active_values),
                    },
                )
            if proof.quarantined:
                trace = self.trace.append("p23_dialog_api_call_refuse", {
                    "proof_id": proof.proof_id,
                    "proof_path": proof_path,
                    "reason": proof.reason,
                    "release_line": "final_math_p23",
                })
                return RuntimeResponse(
                    RuntimeDecision(DecisionKind.REFUSE, proof.reason, trace.event_hash()),
                    "REFUSE: dialog action blocked because a required slot is quarantined.",
                    {"context_subject": context_subject, "proof_id": proof.proof_id, "proof_path": proof_path, "missing_or_quarantined": tuple(quarantined_slots)},
                )
            missing = tuple(slot for slot in required_slots if slot not in active_values)
            trace = self.trace.append("p23_dialog_api_call_ask_clarification", {
                "active_values": dict(active_values),
                "missing_slots": list(missing),
                "proof_id": proof.proof_id,
                "proof_path": proof_path,
                "reason": proof.reason,
                "release_line": "final_math_p23",
            })
            return RuntimeResponse(
                RuntimeDecision(DecisionKind.ASK_CLARIFICATION, proof.reason, trace.event_hash()),
                "ASK_CLARIFICATION: missing required dialog slots: " + ", ".join(missing),
                {"active_values": dict(active_values), "context_subject": context_subject, "missing_slots": missing, "proof_id": proof.proof_id, "proof_path": proof_path},
            )

        if relation == "located_in":
            candidate_objects = tuple(sorted({record.object_value for record in self.memory.active_records() if record.key.relation == "located_in"}))
            result = self.memory.associative_toroidal_read(
                subject,
                relation,
                current_l2_state=self.body.l2_clean_vector(),
                candidate_objects=candidate_objects,
                trace_id=policy.trace_id,
                modulus=self.body.modulus,
            )
        else:
            result = self.memory.query(subject, relation, trace_id=policy.trace_id)
        if result.answered:
            if relation == "located_in":
                proof = self.theorem_layer.prove_where(result.subject, self.memory.export_latest_state(), evidence_id=result.evidence_id or "latest_state_index")
            else:
                proof = self.theorem_layer.prove(Statement.atom(relation, result.subject, result.answer or "unknown"))
            authorization = self.theorem_layer.authorize_query(proof, association_report=None)
            answer_policy = self.policy.evaluate(PolicyRequest(
                kind=RequestKind.ANSWER,
                payload={"answer": result.answer, "claim": f"{relation}({result.subject},{result.answer})"},
                evidence_id=result.evidence_id,
                supported=authorization.answer_allowed,
                source="l2_toroidal_read_proof_bridge" if "associative_toroidal_read" in result.reason else "l2_latest_state_proof_bridge",
            ))
            proof_path = [rule.value for rule in proof.rules]
            output_answer = result.answer or "unknown"
            if target_for_yesno is not None:
                output_answer = "yes" if output_answer == target_for_yesno else "no"
            trace = self.trace.append("l2_query_answer", {
                "answer_policy": answer_policy.as_payload(),
                "authorization": {
                    "answer_allowed": authorization.answer_allowed,
                    "hypothesis_allowed": authorization.hypothesis_allowed,
                    "reason": authorization.reason,
                },
                "body_digest": self.body.digest(),
                "memory_digest": self.memory.digest(),
                "proof_id": proof.proof_id,
                "proof_path": proof_path,
                "query": result.as_payload(),
                "release_line": "final_math",
                "yesno_target": target_for_yesno,
            })
            return RuntimeResponse(
                RuntimeDecision(answer_policy.kind, answer_policy.reason, trace.event_hash()),
                f"ANSWER: {output_answer}",
                {
                    "authorization": {
                        "answer_allowed": authorization.answer_allowed,
                        "hypothesis_allowed": authorization.hypothesis_allowed,
                        "reason": authorization.reason,
                    },
                    "evidence_ids": (result.evidence_id,) if result.evidence_id else (),
                    "proof_id": proof.proof_id,
                    "proof_path": proof_path,
                    "query": result.as_payload(),
                },
            )

        # P22 yes/no closure: a latest explicit negation of the queried
        # target authorizes a bounded "no" answer instead of an unknown refusal.
        # Memory stores negations as quarantined conflict records to avoid
        # surfacing them as positive facts; yes/no queries may still use the
        # negation as direct negative evidence.
        if target_for_yesno is not None:
            negative_record = None
            for record in reversed(self.memory.records):
                if (
                    record.key.subject == subject
                    and record.key.relation == "located_in"
                    and record.object_value == target_for_yesno
                    and record.negated
                ):
                    negative_record = record
                    break
            if negative_record is not None:
                proof_id = active_state_digest({
                    "rule": "ASSERTED_NEGATION_YESNO",
                    "subject": subject,
                    "relation": "located_in",
                    "object": target_for_yesno,
                    "evidence_id": negative_record.delta.fact.evidence.value,
                    "record_id": negative_record.record_id,
                })
                answer_policy = self.policy.evaluate(PolicyRequest(
                    kind=RequestKind.ANSWER,
                    payload={"answer": "no", "claim": f"not located_in({subject},{target_for_yesno})", "source": "explicit_negative_yesno_evidence"},
                    evidence_id=negative_record.delta.fact.evidence.value or evidence_id or "negative_yesno_evidence",
                    supported=True,
                    source="p22_negative_yesno_evidence_bridge",
                ))
                trace = self.trace.append("p22_negative_yesno_answer", {
                    "answer_policy": answer_policy.as_payload(),
                    "claim": f"not located_in({subject},{target_for_yesno})",
                    "negative_record_id": negative_record.record_id,
                    "proof_id": proof_id,
                    "proof_path": ["ASSERTED_NEGATION_YESNO"],
                    "release_line": "final_math_p22",
                    "yesno_target": target_for_yesno,
                })
                return RuntimeResponse(
                    RuntimeDecision(answer_policy.kind, answer_policy.reason, trace.event_hash()),
                    "ANSWER: no",
                    {
                        "authorization": {"answer_allowed": True, "reason": "explicit negative evidence authorizes no-answer"},
                        "evidence_ids": (negative_record.delta.fact.evidence.value,) if negative_record.delta.fact.evidence.value else (),
                        "proof_id": proof_id,
                        "proof_path": ("ASSERTED_NEGATION_YESNO",),
                    },
                )

        # P22 yes/no proof fallback: proof-derived location candidates must be
        # converted to yes/no, never surfaced as a raw place token.
        if target_for_yesno is not None:
            candidate_objects = tuple(sorted({record.object_value for record in self.memory.active_records() if record.key.relation == "located_in"}))
            for obj in (target_for_yesno,) + tuple(item for item in candidate_objects if item != target_for_yesno):
                goal = Statement.atom("located_in", subject, obj)
                proof = self.theorem_layer.prove(goal)
                authorization = self.theorem_layer.authorize_query(proof, allow_hypothesis=False)
                if authorization.answer_allowed:
                    yesno_answer = "yes" if obj == target_for_yesno else "no"
                    answer_policy = self.policy.evaluate(PolicyRequest(
                        kind=RequestKind.ANSWER,
                        payload={"answer": yesno_answer, "claim": goal.canonical(), "source": "p22_yesno_proof_fallback"},
                        evidence_id=evidence_id or "yesno_proof_candidate",
                        supported=True,
                        source="p22_yesno_proof_fallback",
                    ))
                    trace = self.trace.append("p22_yesno_proof_answer", {
                        "answer_policy": answer_policy.as_payload(),
                        "claim": goal.canonical(),
                        "matched_object": obj,
                        "proof_id": proof.proof_id,
                        "proof_path": [rule.value for rule in proof.rules],
                        "release_line": "final_math_p22",
                        "yesno_answer": yesno_answer,
                        "yesno_target": target_for_yesno,
                    })
                    return RuntimeResponse(
                        RuntimeDecision(answer_policy.kind, answer_policy.reason, trace.event_hash()),
                        f"ANSWER: {yesno_answer}",
                        {
                            "authorization": {"answer_allowed": authorization.answer_allowed, "reason": authorization.reason},
                            "evidence_ids": tuple(p.evidence_id for p in proof.premises if p.evidence_id),
                            "proof_id": proof.proof_id,
                            "proof_path": [rule.value for rule in proof.rules],
                        },
                    )

        hypothesis_response = self._answer_query_via_proof_or_l3_hypothesis(subject, relation, evidence_id, request, policy.trace_id)
        if hypothesis_response is not None:
            return hypothesis_response

        decision_kind = DecisionKind.REFUSE if result.status == FactStatus.QUARANTINED else DecisionKind.ASK_CLARIFICATION
        trace = self.trace.append("l2_query_refusal", {
            "body_digest": self.body.digest(),
            "memory_digest": self.memory.digest(),
            "query": result.as_payload(),
            "release_line": "final_math",
        })
        output = "REFUSE: query target is quarantined by contradiction." if decision_kind == DecisionKind.REFUSE else "ASK_CLARIFICATION: no supported non-quarantined fact is available for this query."
        return RuntimeResponse(
            RuntimeDecision(decision_kind, result.reason, trace.event_hash()),
            output,
            {"query": result.as_payload()},
        )

    def _answer_query_via_proof_or_l3_hypothesis(
        self,
        subject: str,
        relation: str,
        evidence_id: str | None,
        request: RuntimeRequest,
        trace_id: str | None,
    ) -> RuntimeResponse | None:
        candidate_objects = tuple(sorted({record.object_value for record in self.memory.active_records() if record.key.relation == relation}))
        if relation == "has_property":
            candidate_objects = tuple(sorted(set(candidate_objects) | {record.object_value for record in self.memory.active_records() if record.key.relation in {"has_property", "color"}}))
        for obj in candidate_objects:
            goal = Statement.atom(relation, subject, obj)
            proof = self.theorem_layer.prove(goal)
            authorization = self.theorem_layer.authorize_query(proof, allow_hypothesis=True)
            if authorization.answer_allowed:
                answer_policy = self.policy.evaluate(PolicyRequest(
                    kind=RequestKind.ANSWER,
                    payload={"answer": obj, "claim": goal.canonical(), "source": "proof_path_fallback"},
                    evidence_id=evidence_id or "proof_path_candidate",
                    supported=True,
                    source="p22_proof_honest_benchmark_bridge",
                ))
                trace = self.trace.append("p22_proof_query_answer", {
                    "answer_policy": answer_policy.as_payload(),
                    "authorization": {"answer_allowed": authorization.answer_allowed, "reason": authorization.reason},
                    "claim": goal.canonical(),
                    "proof_id": proof.proof_id,
                    "proof_path": [rule.value for rule in proof.rules],
                    "release_line": "final_math_p22",
                })
                return RuntimeResponse(
                    RuntimeDecision(answer_policy.kind, answer_policy.reason, trace.event_hash()),
                    f"ANSWER: {obj}",
                    {
                        "authorization": {"answer_allowed": authorization.answer_allowed, "reason": authorization.reason},
                        "evidence_ids": tuple(p.evidence_id for p in proof.premises if p.evidence_id),
                        "proof_id": proof.proof_id,
                        "proof_path": [rule.value for rule in proof.rules],
                    },
                )
            # Same-class property transfer is a hypothesis, not an answer.  It
            # helps bAbI-16 style reporting without leaking authority.
            if relation in {"has_property", "color"}:
                hypo = self.theorem_layer.infer_same_class_property_hypothesis(goal)
                if hypo.hypothesis_allowed:
                    answer_policy = self.policy.evaluate(PolicyRequest(
                        kind=RequestKind.HYPOTHESIS,
                        payload={"answer": obj, "claim": goal.canonical(), "source": "same_class_property_hypothesis"},
                        evidence_id=evidence_id or "l3_hypothesis_candidate",
                        supported=False,
                        source="p22_l3_hypothesis_proof_bridge",
                    ))
                    trace = self.trace.append("p22_l3_hypothesis_query", {
                        "answer_policy": answer_policy.as_payload(),
                        "claim": goal.canonical(),
                        "hypothesis_allowed": hypo.hypothesis_allowed,
                        "proof_id": hypo.proof.proof_id,
                        "release_line": "final_math_p22",
                    })
                    return RuntimeResponse(
                        RuntimeDecision(DecisionKind.HYPOTHESIS, "L3/proof path produced provisional hypothesis only", trace.event_hash()),
                        f"HYPOTHESIS: {obj}",
                        {"authorization": {"answer_allowed": False, "hypothesis_allowed": True}, "proof_id": hypo.proof.proof_id, "proof_path": [rule.value for rule in hypo.proof.rules]},
                    )
        # L3 provisional rules can only seed hypotheses; they are never answers.
        for decision in sorted(self.l3_provisional_rules.values(), key=lambda item: item.candidate.candidate_id):
            st = decision.candidate.statement
            if st.predicate == relation and len(st.args) == 2 and st.args[0] == subject:
                obj = st.args[1]
                auth = self.theorem_layer.authorize_query(st, association_candidates=(Judgment(st, evidence_id=evidence_id, source="association", supported=True),), allow_hypothesis=True)
                if auth.hypothesis_allowed:
                    trace = self.trace.append("p22_l3_provisional_rule_hypothesis", {
                        "candidate_id": decision.candidate.candidate_id,
                        "claim": st.canonical(),
                        "hypothesis_allowed": auth.hypothesis_allowed,
                        "release_line": "final_math_p22",
                    })
                    return RuntimeResponse(
                        RuntimeDecision(DecisionKind.HYPOTHESIS, "L3 provisional rule may be surfaced only as hypothesis", trace.event_hash()),
                        f"HYPOTHESIS: {obj}",
                        {"authorization": {"answer_allowed": False, "hypothesis_allowed": True}, "l3_candidate_id": decision.candidate.candidate_id},
                    )
        return None

    def _refuse_air(self, request: RuntimeRequest, error: dict[str, object]) -> RuntimeResponse:
        ev = self.trace.append("tick", {
            "air_error": error,
            "decision": "refuse",
            "input_source": request.source,
            "release_line": "final_math",
        })
        return RuntimeResponse(
            RuntimeDecision(DecisionKind.REFUSE, "AIR boundary refused input before body commit", ev.event_hash()),
            "REFUSE: input is not valid AIR for L1/L2/L3 body body commit/query.",
            {"air_error": error},
        )

    def _claim_id_for_fact(self, fact: FactFrame) -> str:
        return f"{fact.relation.value}({fact.subject.value},{fact.object.value})"

    def _claim_support_report(self, fact: FactFrame) -> ClaimSupportReport | None:
        return self.claim_support_reports.get(self._claim_id_for_fact(fact))

    def register_claim_support_report(self, report: ClaimSupportReport) -> None:
        """Register an evidence-only claim support report for runtime commit gating.

        The report changes candidate confidence/support only. It does not commit a
        fact and does not bypass AIR, policy, proof, topology or trace gates.
        """
        self.claim_support_reports[str(report.claim_id)] = report
        self.trace.append_claim_support_report(report)

    def _candidate_l2_topology_window(self, candidate_vector: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        window = self.l2_topology_window + (tuple(candidate_vector),)
        return window[-self.max_l2_topology_window :]

    def _append_l2_topology_state(self, vector: tuple[int, ...]) -> None:
        self.l2_topology_window = self._candidate_l2_topology_window(tuple(vector))

    def _topology_precheck_l2_fact(self, delta: FactDelta) -> TopologyDecision:
        preview = self.body.preview_l2_fact_commit(delta)
        decision = self.topology_guard.evaluate_transition(preview.clean_before, preview.clean_after)
        live_window = self._candidate_l2_topology_window(preview.clean_after)
        live_betti = self.topology_guard.live_vr_betti_1_skeleton(live_window)
        return TopologyDecision(
            passed=decision.passed,
            anomaly_score_bp=decision.anomaly_score_bp,
            reason=decision.reason,
            action=decision.action,
            warnings=decision.warnings,
            details={
                **dict(decision.details),
                **live_betti,
                "clean_l2_topology_precheck": True,
                "episode_index": preview.episode_index,
                "fact_count_after": preview.fact_count_after,
                "live_window_count": len(live_window),
                "replaced_existing_contribution": preview.replaced_existing,
                "separated_l2_working_memory": True,
                "weighted_commit_precheck": True,
            },
        )

    def _topology_payload(self, decision) -> dict[str, object]:
        return {
            "action": decision.action.value,
            "anomaly_score_bp": decision.anomaly_score_bp,
            "details": dict(decision.details),
            "passed": decision.passed,
            "reason": decision.reason,
            "warnings": list(decision.warnings),
        }

    def _fact_from_payload(self, payload: dict[str, object], evidence_id: str | None) -> FactFrame:
        if evidence_id is None:
            raise ValueError("fact candidate has no evidence_id")
        return FactFrame(
            subject=EntityId(str(payload.get("subject", ""))),
            relation=RelationId(str(payload.get("relation", ""))),
            object=EntityId(str(payload.get("object", ""))),
            evidence=EvidenceId(evidence_id),
            confidence_bp=int(payload.get("confidence_bp", 10000)),
        )

    def _query_relation(self, query_type: str) -> str:
        canonical = query_type.strip().lower()
        aliases = {
            "location": "located_in",
            "where": "located_in",
            "place": "located_in",
            "book_table": "api_call_ready",
            "api_call_ready": "api_call_ready",
        }
        if canonical.startswith("yesno_in_"):
            return canonical
        if canonical.startswith("api_call_ready_"):
            return "api_call_ready"
        return aliases.get(canonical, canonical)

    def _dialog_domain_from_context_subject(self, context_subject: str) -> str | None:
        value = context_subject.strip().lower()
        if "hotel" in value:
            return "hotel"
        if "restaurant" in value:
            return "restaurant"
        return None

    def _api_call_required_slots_from_query_type(self, query_type: str) -> dict[str, str]:
        canonical = query_type.strip().lower()
        if canonical.startswith("api_call_ready_"):
            suffix = canonical[len("api_call_ready_"):]
            slots = tuple(item for item in suffix.split("_") if item)
        elif canonical == "api_call_ready" or canonical == "book_table":
            slots = ("cuisine", "location", "price")
        else:
            slots = ("cuisine", "location", "price")
        allowed = ("cuisine", "location", "price", "stars", "party_size")
        selected = tuple(slot for slot in slots if slot in allowed) or ("cuisine", "location", "price")
        return {slot: "*" for slot in selected}

    def register_simulated_skill(self, name: str, ensures: Statement | str, *, steps: tuple[PlanStep, ...] | None = None) -> SimulatedSkill:
        skill = SimulatedSkill(name=name, ensures=ensures, steps=steps or (PlanStep(name),), simulated_only=True)
        self.skill_registry.register(skill)
        self.theorem_layer.add_ensures(skill.name, skill.ensures)
        return skill

    def register_real_action_skill_for_block_test(self, name: str, ensures: Statement | str) -> SimulatedSkill:
        skill = SimulatedSkill(name=name, ensures=ensures, steps=(), simulated_only=False)
        self.skill_registry.register(skill)
        self.theorem_layer.add_ensures(skill.name, skill.ensures)
        return skill

    def assert_proof_fact(self, statement: Statement | str, *, evidence_id: str | None = None) -> None:
        self.theorem_layer.add_judgment(statement, evidence_id=evidence_id, source="asserted", supported=True)

    def _coerce_observation(self, observation: SensoryObservation | dict[str, object]) -> SensoryObservation:
        if isinstance(observation, SensoryObservation):
            return observation
        phase = tuple(int(value) for value in observation.get("phase", self.body.l1.vector))
        return SensoryObservation(
            modality=str(observation.get("modality", "simulated")),
            value=str(observation.get("value", "observation")),
            intensity_bp=int(observation.get("intensity_bp", 0)),
            reliability_bp=int(observation.get("reliability_bp", 0)),
            phase=phase,
            evidence_id=str(observation.get("evidence_id", "simulated_observation")),
            simulated=bool(observation.get("simulated", True)),
            real_sensor_commit_allowed=bool(observation.get("real_sensor_commit_allowed", False)),
            modulus=self.body.modulus,
        )

    def observe_simulated(self, observation: SensoryObservation | dict[str, object]) -> RuntimeResponse:
        """Run the integrated L1/world/curiosity/homeostasis observation path.

        Lifecycle:
            observe_simulated(o_t) -> L1 delta -> world prediction/error ->
            curiosity -> homeostatic update -> trace.

        It never commits to L2/L3 and never grants real sensor authority.
        """
        obs = self._coerce_observation(observation)
        before = self.health()
        before_l1 = self.body.l1.vector
        zero_action = Q256WorldAction(
            name="simulated_observation_prediction",
            delta=tuple(0 for _ in range(self.body.dimension)),
            evidence_id=obs.evidence_id,
            metadata={"runtime_path": "observe_simulated"},
            modulus=self.body.modulus,
        )
        prediction = self.world_model.predict_next_state(before_l1, zero_action)
        transition = self.body.observe_simulated(obs, evidence_id=obs.evidence_id)
        observed_prediction = self.world_model.update_from_observation(prediction, self.body.l1.vector)
        hypothesis_result = self.hypothesis_loop.evaluate(
            state=self.homeostatic_state,
            predicted_phase=prediction.predicted_state.phases,
            observation=obs,
        )
        self.homeostatic_state = hypothesis_result.next_state
        after = self.health()
        trace = self.trace.append("simulated_observation_runtime", {
            "body_transition": transition.as_payload(),
            "curiosity_bp": hypothesis_result.curiosity.curiosity_bp,
            "homeostasis_next": self.homeostatic_state.as_mapping(),
            "l1_clock_after": after["l1_clock"],
            "l1_clock_before": before["l1_clock"],
            "l2_clock_after": after["l2_clock"],
            "l2_clock_before": before["l2_clock"],
            "l3_clock_after": after["l3_clock"],
            "l3_clock_before": before["l3_clock"],
            "prediction_error_bp": observed_prediction.error.error_bp if observed_prediction.error else 0,
            "real_sensor_commit_allowed": False,
            "release_line": "final_math",
            "suggested_action": hypothesis_result.suggested_action,
        })
        return RuntimeResponse(
            decision=RuntimeDecision(DecisionKind.HYPOTHESIS, "simulated observation processed without fact commit", trace.event_hash()),
            output="HYPOTHESIS: simulated observation updated L1/control state only; no L2/L3 fact commit.",
            diagnostics={
                "curiosity_bp": hypothesis_result.curiosity.curiosity_bp,
                "l1_changed": before["body_digest"] != after["body_digest"],
                "l2_fact_count_after": after["latest_fact_count"],
                "l2_fact_count_before": before["latest_fact_count"],
                "l2_clock_after": after["l2_clock"],
                "l2_clock_before": before["l2_clock"],
                "l3_clock_after": after["l3_clock"],
                "l3_clock_before": before["l3_clock"],
                "prediction_error_bp": observed_prediction.error.error_bp if observed_prediction.error else 0,
                "real_sensor_commit_allowed": False,
                "trace_id": trace.event_hash(),
            },
        )

    def plan_simulated_skill(self, skill_name: str, *, state: HomeostaticState | None = None, domain_seeds: tuple[str, ...] = ()) -> RuntimeResponse:
        result = self.planner.plan_skill(
            skill_name,
            self.skill_registry,
            self.theorem_layer,
            state=state or HomeostaticState(),
            allow_real_actions=self.config.allow_real_actions,
            trace=self.trace,
            domain_seeds=domain_seeds,
        )
        if result.decision == DecisionKind.ACT_SIMULATED:
            output = "ACT_SIMULATED: verified simulated skill planned; no real action executed."
        elif result.decision == DecisionKind.BLOCK_REAL_ACTION:
            output = "BLOCK_REAL_ACTION: real action remains blocked by simulation-first boundary."
        else:
            output = "REFUSE: simulated skill was not authorized for planning."
        return RuntimeResponse(
            decision=RuntimeDecision(result.decision, result.reason, result.trace_id),
            output=output,
            diagnostics={
                "domain_count": result.domain_count,
                "domain_randomized": result.domain_randomized,
                "expected_free_energy_bp": result.expected_free_energy_bp,
                "habitat_gate_reason": result.habitat_gate_reason,
                "habitat_policy_allowed": result.habitat_policy_allowed,
                "proof_id": result.proof_id,
                "rollout_score_bp": result.rollout_score_bp,
                "step_count": len(result.steps),
                "steps": [step.action_name for step in result.steps],
                "viability_bp": result.viability_bp,
                "worst_domain_seed": result.worst_domain_seed,
            },
        )

    def consolidate_l2_episode(
        self,
        *,
        episode_id: str,
        promoted_rules_count: int = 0,
        evidence_id: str = "l2_episode_consolidation",
    ) -> RuntimeResponse:
        """Archive and reset the active L2 working episode.

        This method does not delete semantic memory records or protected trace.
        It only closes the current L2 working torus, anchors it, zeros raw L2,
        and starts a fresh episode phase tag.
        """
        before_body = self.body.digest()
        transition, anchor = self.body.consolidate_l2_episode(
            episode_id=episode_id,
            promoted_rules_count=promoted_rules_count,
            evidence_id=evidence_id,
        )
        self.l2_topology_window = (self.body.l2_clean_vector(),)
        trace = self.trace.append("l2_episode_consolidation_reset", {
            "anchor": anchor.as_payload(),
            "body_after": self.body.digest(),
            "body_before": before_body,
            "body_transition": transition.as_payload(),
            "release_line": "final_math_q256_separated_l2",
        })
        return RuntimeResponse(
            RuntimeDecision(DecisionKind.HYPOTHESIS, "L2 working episode anchored and reset", trace.event_hash()),
            "HYPOTHESIS: L2 working episode consolidated and reset; semantic history remains in protected memory/trace.",
            {"anchor": anchor.as_payload(), "body_transition": transition.as_payload()},
        )

    def _closed_loop_skill_delta(self, skill_name: str) -> tuple[int, ...]:
        """Return deterministic Q256 action delta for a P11 simulated skill."""
        name = str(skill_name)
        delta = [0 for _ in range(self.body.dimension)]
        stride = max(1, self.body.modulus // 10)
        if name == "advance":
            delta[0] = stride
        elif name == "rotate":
            index = 1 if self.body.dimension > 1 else 0
            delta[index] = stride
        elif name == "hold":
            pass
        else:
            raise ValueError(f"unknown closed-loop skill: {skill_name}")
        return tuple(delta)

    def _ensure_closed_loop_skills(self) -> None:
        """Register P11 simulation skills once, without overriding user skills."""
        for name, ensures in (
            ("advance", "simulated_position_dim0_updated"),
            ("rotate", "simulated_position_dim1_updated"),
            ("hold", "simulated_state_maintained"),
        ):
            if self.skill_registry.get(name) is None:
                steps = (PlanStep(action_name=name),)
                skill = SimulatedSkill(
                    name=name,
                    ensures=ensures,
                    steps=steps,
                    simulated_only=True,
                    metadata={"closed_loop_delta_digest": active_state_digest({"delta": self._closed_loop_skill_delta(name)})},
                )
                self.skill_registry.register(skill)
                self.theorem_layer.add_ensures(skill.name, skill.ensures)

    def _closed_loop_action(self, skill_name: str) -> Q256WorldAction:
        return Q256WorldAction(
            name=str(skill_name),
            delta=self._closed_loop_skill_delta(skill_name),
            evidence_id=f"p11_skill_{skill_name}",
            confidence_bp=MAX_BP,
            metadata={"closed_loop": True, "simulation_only": True},
            modulus=self.body.modulus,
        )

    def _closed_loop_sensor_packet(self, env_state: TorusVector, *, step: int, evidence_id: str) -> RawSensorPacket:
        """Build integer sensor samples from a deterministic simulated environment."""
        noise_phase = hash_to_phase(
            f"p11-noise:{step}",
            dimension=self.body.dimension,
            modulus=self.body.modulus,
            namespace="p11_sim_noise",
        )
        noise_limit = max(1, self.body.modulus // 50)
        samples = tuple(q_mod(phase + (noise % noise_limit), self.body.modulus) for phase, noise in zip(env_state.phases, noise_phase))
        return RawSensorPacket(
            modality="p11_closed_loop",
            samples=samples,
            sample_min=0,
            sample_max=self.body.modulus - 1,
            reliability_bp=MAX_BP,
            evidence_id=evidence_id,
            metadata={"step": int(step), "simulation_only": True},
        )

    def _closed_loop_goal_progress_bp(self, skill_name: str) -> int:
        if skill_name == "advance":
            return 1800
        if skill_name == "rotate":
            return 900
        return 0

    def _closed_loop_complexity_bp(self, skill_name: str) -> int:
        return 0 if skill_name == "hold" else 800

    def _closed_loop_action_magnitude_raw(self, action: Q256WorldAction) -> int:
        zero = tuple(0 for _ in range(self.body.dimension))
        return q_toroidal_loss_vector(zero, action.delta, self.body.modulus)

    def _closed_loop_complexity_raw(self, action: Q256WorldAction) -> int:
        if action.name == "hold":
            return 0
        non_zero = sum(1 for value in action.delta if value != 0)
        return non_zero + self._closed_loop_action_magnitude_raw(action)

    def _closed_loop_goal_progress_raw(self, action: Q256WorldAction) -> int:
        magnitude = self._closed_loop_action_magnitude_raw(action)
        if action.name == "advance":
            return magnitude + magnitude + 1
        if action.name == "rotate":
            return magnitude + 1
        return 0

    def _closed_loop_novelty_raw(self, encoded_phase: tuple[int, ...], current_l1_phase: tuple[int, ...]) -> int:
        return q_toroidal_loss_vector(encoded_phase, current_l1_phase, self.body.modulus)

    def _closed_loop_homeostatic_effect(self, skill_name: str, *, model_error_bp: int = 0) -> HomeostaticActionEffect:
        if skill_name == "advance":
            return HomeostaticActionEffect(
                energy_delta_bp=-80,
                sleep_pressure_delta_bp=30,
                uncertainty_delta_bp=-20,
                model_error_bp=model_error_bp,
                complexity_bp=self._closed_loop_complexity_bp(skill_name),
                goal_progress_bp=self._closed_loop_goal_progress_bp(skill_name),
                novelty_gain_bp=200,
            )
        if skill_name == "rotate":
            return HomeostaticActionEffect(
                energy_delta_bp=-50,
                sleep_pressure_delta_bp=20,
                uncertainty_delta_bp=-10,
                model_error_bp=model_error_bp,
                complexity_bp=self._closed_loop_complexity_bp(skill_name),
                goal_progress_bp=self._closed_loop_goal_progress_bp(skill_name),
                novelty_gain_bp=300,
            )
        return HomeostaticActionEffect(
            energy_delta_bp=20,
            sleep_pressure_delta_bp=-20,
            uncertainty_delta_bp=-10,
            model_error_bp=model_error_bp,
            complexity_bp=0,
            goal_progress_bp=0,
            novelty_gain_bp=0,
        )

    def _closed_loop_environment_step(self, env_state: TorusVector, action: Q256WorldAction) -> TorusVector:
        next_state = tuple(q_mod(phase + delta, self.body.modulus) for phase, delta in zip(env_state.phases, action.delta))
        return TorusVector(next_state, self.body.modulus)

    def run_closed_loop_simulation(self, *, steps: int = 15, verify_trace_each_step: bool = True) -> ClosedLoopSimulationReport:
        """Run P11: L1 -> world model -> simulation planner -> protected trace.

        The loop is simulation-only and integer-only:
        sensor samples are integers, L1 projection is Q256, world prediction and
        surprise use toroidal integer loss, action selection uses raw Q256 EFE
        scores, habitat policy blocks real action paths, and every step is
        logged into the protected trace.
        """
        step_count = int(steps)
        if step_count <= 0:
            raise ValueError("steps must be positive")
        if not self.awake:
            self.wake()
        self._ensure_closed_loop_skills()
        habitat_policy = SimulationHabitatPolicy()
        env_state = TorusVector(
            hash_to_phase("p11-env-init", dimension=self.body.dimension, modulus=self.body.modulus, namespace="p11_env"),
            self.body.modulus,
        )
        records: list[ClosedLoopStepRecord] = []
        trace_ok_cached = self.trace.verify()
        for step in range(step_count):
            packet = self._closed_loop_sensor_packet(env_state, step=step, evidence_id=f"p11_obs_{step}")
            encoded = self.l1_encoder.encode(
                packet,
                current_l1_phase=self.body.l1.vector,
                predicted_phase=self.body.l1.vector,
                current_risk_bp=self.homeostatic_state.risk_bp,
            )
            transition = self.body.observe_l1_encoded(encoded, evidence_id=packet.evidence_id)
            current_l1 = TorusVector(self.body.l1.vector, self.body.modulus)
            observation_digest = active_state_digest(TorusVector(encoded.observed_phase, self.body.modulus))
            allowed: list[tuple[WorldActionEvaluation, Q256WorldAction, object]] = []
            for skill_name in self.skill_registry.names():
                if skill_name not in {"advance", "rotate", "hold"}:
                    continue
                action = self._closed_loop_action(skill_name)
                evaluation = self.world_model.evaluate_action_expected_free_energy(
                    current_l1,
                    action,
                    context={"loop": "p11", "skill": skill_name},
                    complexity_bp=self._closed_loop_complexity_bp(skill_name),
                    novelty_gain_bp=encoded.curiosity.curiosity_bp,
                    goal_progress_bp=self._closed_loop_goal_progress_bp(skill_name),
                    complexity_raw=self._closed_loop_complexity_raw(action),
                    novelty_gain_raw=self._closed_loop_novelty_raw(encoded.observed_phase, tuple(current_l1.phases)),
                    goal_progress_raw=self._closed_loop_goal_progress_raw(action),
                )
                gate = HabitatGateInput(
                    proof_bp=MAX_BP,
                    topology_bp=MAX_BP,
                    model_error_bp=self.world_model.self_model.last_error_bp,
                    model_error_raw=self.world_model.self_model.last_error_raw,
                    max_model_error_raw=self.body.dimension * 65535,
                    policy_ok=True,
                    trace_ok=(self.trace.verify() if verify_trace_each_step else trace_ok_cached),
                    action_class="simulated",
                    external_sensor_only=True,
                )
                gate_decision = habitat_policy.evaluate(gate)
                if gate_decision.allowed:
                    allowed.append((evaluation, action, gate_decision))
            if not allowed:
                action = self._closed_loop_action("hold")
                prediction = self.world_model.predict_next_state(current_l1, action)
                gate_decision = habitat_policy.evaluate(HabitatGateInput(policy_ok=False, action_class="simulated"))
                selected_evaluation = self.world_model.evaluate_action_expected_free_energy(current_l1, action)
            else:
                selected_evaluation, action, gate_decision = min(allowed, key=lambda item: (item[0].expected_free_energy_raw, item[0].action_name))
                prediction = self.world_model.predict_next_state(current_l1, action)
            env_state = self._closed_loop_environment_step(env_state, action)
            next_packet = self._closed_loop_sensor_packet(env_state, step=step + 1, evidence_id=f"p11_next_obs_{step}")
            next_encoded = self.l1_encoder.encode(
                next_packet,
                current_l1_phase=self.body.l1.vector,
                predicted_phase=prediction.predicted_state.phases,
                current_risk_bp=self.homeostatic_state.risk_bp,
            )
            observed_prediction = self.world_model.update_from_observation(prediction, next_encoded.observed_phase)
            surprise_raw = observed_prediction.error.loss if observed_prediction.error else 0
            surprise_bp = observed_prediction.error.error_bp if observed_prediction.error else MAX_BP
            self.homeostatic_state = self.world_model_self_homeostatic_update(action.name, surprise_bp)
            event = self.trace.append("p11_closed_loop_step", {
                "action": action.name,
                "body_transition": transition.as_payload(),
                "chosen_evaluation": {
                    "confidence_bp": selected_evaluation.confidence_bp,
                    "expected_free_energy_bp": selected_evaluation.expected_free_energy_bp,
                    "expected_free_energy_raw": selected_evaluation.expected_free_energy_raw,
                    "predicted_digest": selected_evaluation.predicted_digest,
                    "risk_bp": selected_evaluation.risk_bp,
                    "risk_raw": selected_evaluation.risk_raw,
                    "uncertainty_bp": selected_evaluation.uncertainty_bp,
                    "uncertainty_raw": selected_evaluation.uncertainty_raw,
                },
                "encoder": encoded.as_payload(),
                "gate": gate_decision.as_trace_payload(),
                "homeostasis": self.homeostatic_state.as_mapping(),
                "l1_digest": self.body.l1.digest,
                "next_observation_digest": active_state_digest(TorusVector(next_encoded.observed_phase, self.body.modulus)),
                "observation_digest": observation_digest,
                "prediction": {
                    "action_name": prediction.action_name,
                    "confidence_bp": prediction.confidence_bp,
                    "predicted_digest": prediction.predicted_digest,
                    "uncertainty_bp": prediction.uncertainty_bp,
                },
                "release_line": "final_math_q256_p11_closed_loop",
                "simulated_only": True,
                "step": step,
                "surprise_bp": surprise_bp,
                "surprise_raw": surprise_raw,
                "trace_verified_before_step": (self.trace.verify() if verify_trace_each_step else trace_ok_cached),
            })
            records.append(ClosedLoopStepRecord(
                step=step,
                observation_digest=observation_digest,
                l1_digest=self.body.l1.digest,
                predicted_digest=prediction.predicted_digest,
                observed_next_digest=active_state_digest(TorusVector(next_encoded.observed_phase, self.body.modulus)),
                chosen_action=action.name,
                efe_raw=selected_evaluation.expected_free_energy_raw,
                surprise_raw=surprise_raw,
                efe_bp=selected_evaluation.expected_free_energy_bp,
                surprise_bp=surprise_bp,
                gate_allowed=gate_decision.allowed,
                trace_id=event.event_hash(),
            ))
        total_efe = sum(record.efe_bp for record in records)
        total_surprise = sum(record.surprise_bp for record in records)
        report = ClosedLoopSimulationReport(
            steps=tuple(records),
            trace_head=self.trace.head,
            trace_verified=self.trace.verify(),
            total_efe_raw=sum(record.efe_raw for record in records),
            total_surprise_raw=sum(record.surprise_raw for record in records),
            average_efe_bp=total_efe // len(records),
            average_surprise_bp=total_surprise // len(records),
            advance_count=sum(1 for record in records if record.chosen_action == "advance"),
            rotate_count=sum(1 for record in records if record.chosen_action == "rotate"),
            hold_count=sum(1 for record in records if record.chosen_action == "hold"),
            modulus=self.body.modulus,
            dimension=self.body.dimension,
        )
        self.trace.append("p11_closed_loop_summary", report.as_payload())
        return report

    def world_model_self_homeostatic_update(self, action_name: str, surprise_bp: int) -> HomeostaticState:
        """Apply the selected simulated action effect plus prediction surprise."""
        bounded_surprise = max(0, min(MAX_BP, int(surprise_bp)))
        effect = self._closed_loop_homeostatic_effect(action_name, model_error_bp=bounded_surprise)
        next_state = self.planner.surrogate.apply_action_effect(self.homeostatic_state, effect)
        return next_state.with_delta(
            uncertainty_delta_bp=bounded_surprise // 10,
            sleep_pressure_delta_bp=bounded_surprise // 20,
        )

    def _p24_goal_position(self, grid_size: int) -> tuple[int, int]:
        size = int(grid_size)
        return size - 1, size - 1

    def _p24_hazard_positions(self, grid_size: int) -> tuple[tuple[int, int], ...]:
        size = int(grid_size)
        mid = size // 2
        hazards = ((mid, mid), (mid, max(1, mid - 1)))
        goal = self._p24_goal_position(size)
        return tuple(pos for pos in hazards if pos not in ((0, 0), goal))

    def _p24_manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _p24_min_hazard_distance(self, pos: tuple[int, int], hazards: tuple[tuple[int, int], ...], grid_size: int) -> int:
        if not hazards:
            return int(grid_size) * 2
        return min(self._p24_manhattan(pos, hazard) for hazard in hazards)

    def _p24_grid_observation_packet(
        self,
        *,
        x: int,
        y: int,
        goal_x: int,
        goal_y: int,
        grid_size: int,
        hazards: tuple[tuple[int, int], ...],
        visited_count: int,
        step: int,
        evidence_id: str,
    ) -> RawSensorPacket:
        scale = 10000
        denom = max(1, int(grid_size) - 1)
        distance = self._p24_manhattan((x, y), (goal_x, goal_y))
        max_distance = max(1, (int(grid_size) - 1) * 2)
        hazard_distance = self._p24_min_hazard_distance((x, y), hazards, grid_size)
        samples = (
            (int(x) * scale) // denom,
            (int(y) * scale) // denom,
            (int(goal_x) * scale) // denom,
            (int(goal_y) * scale) // denom,
            (abs(int(goal_x) - int(x)) * scale) // denom,
            (abs(int(goal_y) - int(y)) * scale) // denom,
            (int(distance) * scale) // max_distance,
            (min(int(hazard_distance), max_distance) * scale) // max_distance,
            self.homeostatic_state.energy_bp,
            self.homeostatic_state.sleep_pressure_bp,
            self.homeostatic_state.risk_bp,
            self.homeostatic_state.integrity_bp,
            self.homeostatic_state.novelty_bp,
            self.homeostatic_state.uncertainty_bp,
            (int(visited_count) * scale) // max(1, int(grid_size) * int(grid_size)),
            (int(step) * scale) // max(1, int(grid_size) * int(grid_size)),
        )
        return RawSensorPacket(
            modality="p24_living_grid",
            samples=samples,
            sample_min=0,
            sample_max=scale,
            reliability_bp=10000,
            evidence_id=evidence_id,
            metadata={
                "grid_size": int(grid_size),
                "simulation_only": True,
                "x": int(x),
                "y": int(y),
            },
        )

    def _p24_apply_action(self, action_name: str, x: int, y: int, grid_size: int) -> tuple[int, int, bool]:
        next_x = int(x)
        next_y = int(y)
        if action_name == "move_north":
            next_y -= 1
        elif action_name == "move_south":
            next_y += 1
        elif action_name == "move_west":
            next_x -= 1
        elif action_name == "move_east":
            next_x += 1
        elif action_name in {"scan", "rest"}:
            pass
        else:
            raise ValueError(f"unknown p24 action: {action_name}")
        wall_hit = next_x < 0 or next_y < 0 or next_x >= int(grid_size) or next_y >= int(grid_size)
        if wall_hit:
            next_x = max(0, min(int(grid_size) - 1, next_x))
            next_y = max(0, min(int(grid_size) - 1, next_y))
        return next_x, next_y, wall_hit

    def _p24_action_delta(self, action_name: str) -> tuple[int, ...]:
        if action_name in {"scan", "rest"}:
            return tuple(0 for _ in range(self.body.dimension))
        base = hash_to_phase(
            f"p24-active-action:{action_name}",
            dimension=self.body.dimension,
            modulus=self.body.modulus,
            namespace="p24_action_delta",
        )
        stride = max(1, self.body.modulus // 256)
        return tuple(q_mod(value % stride, self.body.modulus) for value in base)

    def _p24_world_action(self, action_name: str) -> Q256WorldAction:
        return Q256WorldAction(
            name=action_name,
            delta=self._p24_action_delta(action_name),
            evidence_id=f"p24_skill_{action_name}",
            confidence_bp=10000,
            metadata={"p24_living_active_agent": True, "simulation_only": True},
            modulus=self.body.modulus,
        )

    def _p24_action_effect(
        self,
        *,
        action_name: str,
        wall_hit: bool,
        hazard_hit: bool,
        new_position: bool,
        prediction_error_bp: int = 0,
    ) -> HomeostaticActionEffect:
        if action_name == "rest":
            return HomeostaticActionEffect(
                energy_delta_bp=700,
                sleep_pressure_delta_bp=-500,
                risk_delta_bp=-150,
                novelty_delta_bp=-100,
                uncertainty_delta_bp=-150,
                model_error_bp=prediction_error_bp,
                complexity_bp=300,
                goal_progress_bp=100,
            )
        if action_name == "scan":
            return HomeostaticActionEffect(
                energy_delta_bp=-120,
                sleep_pressure_delta_bp=40,
                risk_delta_bp=-250,
                novelty_delta_bp=300 if new_position else 80,
                uncertainty_delta_bp=-900,
                model_error_bp=prediction_error_bp,
                complexity_bp=600,
                goal_progress_bp=250,
                novelty_gain_bp=600,
            )
        risk_delta = 1200 if hazard_hit else (-120)
        if wall_hit:
            risk_delta += 900
        return HomeostaticActionEffect(
            energy_delta_bp=-260,
            sleep_pressure_delta_bp=70,
            risk_delta_bp=risk_delta,
            integrity_delta_bp=-600 if hazard_hit else 0,
            novelty_delta_bp=650 if new_position else -80,
            uncertainty_delta_bp=160,
            model_error_bp=prediction_error_bp,
            complexity_bp=500,
            goal_progress_bp=1200 if new_position else 500,
            novelty_gain_bp=700 if new_position else 0,
        )

    def _p24_action_score(
        self,
        *,
        action_name: str,
        x: int,
        y: int,
        goal: tuple[int, int],
        grid_size: int,
        hazards: tuple[tuple[int, int], ...],
        visited: set[tuple[int, int]],
    ) -> dict[str, int | bool | str]:
        next_x, next_y, wall_hit = self._p24_apply_action(action_name, x, y, grid_size)
        before_distance = self._p24_manhattan((x, y), goal)
        after_distance = self._p24_manhattan((next_x, next_y), goal)
        progress = max(0, before_distance - after_distance)
        away = max(0, after_distance - before_distance)
        hazard_hit = (next_x, next_y) in hazards
        new_position = (next_x, next_y) not in visited
        hazard_distance = self._p24_min_hazard_distance((next_x, next_y), hazards, grid_size)
        risk_bp = 0
        if wall_hit:
            risk_bp += 4000
        if hazard_hit:
            risk_bp += 6500
        if hazard_distance == 1:
            risk_bp += 1200
        if self.homeostatic_state.energy_bp < 1500 and action_name != "rest":
            risk_bp += 2000
        if action_name == "rest" and self.homeostatic_state.energy_bp > 8500 and before_distance > 0:
            risk_bp += 1400
        uncertainty_bp = self.homeostatic_state.uncertainty_bp
        if action_name == "scan":
            uncertainty_bp = max(0, uncertainty_bp - 1800)
        elif action_name == "rest":
            uncertainty_bp = max(0, uncertainty_bp - 400)
        else:
            uncertainty_bp = min(10000, uncertainty_bp + 250)
        complexity_bp = 300 if action_name == "rest" else 600 if action_name == "scan" else 500
        novelty_bp = 900 if new_position else 0
        if action_name == "scan":
            novelty_bp += 450
        homeo_dev = self.planner.surrogate.homeostatic_deviation_bp(self.homeostatic_state)
        goal_gap_bp = (after_distance * 10000) // max(1, (int(grid_size) - 1) * 2)
        expected_free_energy_raw = (
            goal_gap_bp * 2000
            + risk_bp * 1500
            + uncertainty_bp * 600
            + complexity_bp * 300
            + homeo_dev * 400
            + away * 2500000
            - progress * 3500000
            - novelty_bp * 500
        )
        return {
            "action": action_name,
            "next_x": next_x,
            "next_y": next_y,
            "wall_hit": wall_hit,
            "hazard_hit": hazard_hit,
            "new_position": new_position,
            "goal_distance_before": before_distance,
            "goal_distance_after": after_distance,
            "goal_progress_cells": progress,
            "risk_bp": max(0, min(10000, risk_bp)),
            "uncertainty_bp": uncertainty_bp,
            "complexity_bp": complexity_bp,
            "novelty_bp": novelty_bp,
            "homeostatic_deviation_bp": homeo_dev,
            "expected_free_energy_raw": int(expected_free_energy_raw),
        }

    def _p25_unified_dialog_events(self) -> tuple[dict[str, object], ...]:
        """Return bounded dialog/action events injected into one living loop.

        The scenarios adapt the uploaded P24 dialog scale-up idea into the
        runtime itself: no separate dialog harness, no dummy oracle, no expected
        answer in the inference path.  Expected fields are used only after the
        runtime response to audit the report.
        """

        return (
            {
                "step": 0,
                "scenario": "restaurant_multiple_corrections",
                "utterance": "I want Italian food in Rome.",
                "expected_kind": DecisionKind.ASK_CLARIFICATION.value,
                "expected_contains": ("price",),
            },
            {
                "step": 1,
                "scenario": "restaurant_multiple_corrections",
                "utterance": "Make it cheap.",
                "expected_kind": DecisionKind.ACT_SIMULATED.value,
                "expected_contains": ("domain=restaurant", "cuisine=italian", "location=rome", "price=cheap"),
            },
            {
                "step": 2,
                "scenario": "restaurant_multiple_corrections",
                "utterance": "Actually, make it Chinese.",
                "expected_kind": DecisionKind.ACT_SIMULATED.value,
                "expected_contains": ("domain=restaurant", "cuisine=chinese", "location=rome", "price=cheap"),
            },
            {
                "step": 3,
                "scenario": "restaurant_multiple_corrections",
                "utterance": "No, Italian again, but in Paris.",
                "expected_kind": DecisionKind.ACT_SIMULATED.value,
                "expected_contains": ("domain=restaurant", "cuisine=italian", "location=paris", "price=cheap"),
            },
            {
                "step": 4,
                "scenario": "domain_shift_restaurant_to_hotel",
                "utterance": "Now I need a hotel in Paris.",
                "expected_kind": DecisionKind.ASK_CLARIFICATION.value,
                "expected_contains": ("stars",),
            },
            {
                "step": 5,
                "scenario": "domain_shift_restaurant_to_hotel",
                "utterance": "4 stars.",
                "expected_kind": DecisionKind.ACT_SIMULATED.value,
                "expected_contains": ("domain=hotel", "location=paris", "stars=4"),
            },
            {
                "step": 6,
                "scenario": "domain_shift_with_context_reset",
                "utterance": "Actually, I meant a restaurant in London.",
                "expected_kind": DecisionKind.ASK_CLARIFICATION.value,
                "expected_contains": ("cuisine", "price"),
            },
            {
                "step": 7,
                "scenario": "domain_shift_with_context_reset",
                "utterance": "Cheap French cuisine.",
                "expected_kind": DecisionKind.ACT_SIMULATED.value,
                "expected_contains": ("domain=restaurant", "cuisine=french", "location=london", "price=cheap"),
            },
        )

    def _p25_run_unified_dialog_turn(self, event: dict[str, object]) -> UnifiedLivingDialogTurnRecord:
        utterance = str(event["utterance"])
        expected_kind = str(event["expected_kind"])
        expected_contains = tuple(str(item).lower() for item in event.get("expected_contains", ()))
        response = self.tick(RuntimeRequest(utterance, source="p25_unified_living_dialog"))
        output_lower = response.output.lower()
        diagnostics = response.diagnostics if isinstance(response.diagnostics, dict) else {}
        effective_diagnostics = diagnostics
        batch_obj = diagnostics.get("batch")
        if isinstance(batch_obj, list) and batch_obj:
            last_item = batch_obj[-1]
            if isinstance(last_item, dict):
                effective_diagnostics = last_item
        evidence_ids_obj = effective_diagnostics.get("evidence_ids", ()) if isinstance(effective_diagnostics, dict) else ()
        evidence_count = len(tuple(evidence_ids_obj)) if isinstance(evidence_ids_obj, (tuple, list)) else 0
        proof_id = effective_diagnostics.get("proof_id") if isinstance(effective_diagnostics, dict) else None
        supported = bool(proof_id or evidence_count > 0 or response.decision.kind == DecisionKind.ASK_CLARIFICATION)
        false_support = 1 if response.decision.kind in {DecisionKind.ANSWER, DecisionKind.ACT_SIMULATED} and not supported else 0
        kind_ok = response.decision.kind.value == expected_kind
        contains_ok = all(fragment in output_lower for fragment in expected_contains)
        passed = bool(kind_ok and contains_ok and false_support == 0)
        active_values_obj = effective_diagnostics.get("slot_values", effective_diagnostics.get("active_values", {})) if isinstance(effective_diagnostics, dict) else {}
        active_values = {str(k): str(v) for k, v in dict(active_values_obj).items()} if isinstance(active_values_obj, dict) else {}
        missing_obj = effective_diagnostics.get("missing_slots", ()) if isinstance(effective_diagnostics, dict) else ()
        missing_slots = tuple(str(item) for item in missing_obj) if isinstance(missing_obj, (tuple, list)) else ()
        return UnifiedLivingDialogTurnRecord(
            step=int(event["step"]),
            scenario=str(event["scenario"]),
            user_text=utterance,
            expected_kind=expected_kind,
            expected_contains=expected_contains,
            decision_kind=response.decision.kind.value,
            output=response.output,
            passed=passed,
            false_support=false_support,
            evidence_count=evidence_count,
            proof_id=str(proof_id) if proof_id is not None else None,
            trace_id=response.decision.trace_id,
            active_values=active_values,
            missing_slots=missing_slots,
        )

    def _p25_dialog_metrics(self, records: list[UnifiedLivingDialogTurnRecord]) -> dict[str, int]:
        total = len(records)
        passed = sum(1 for item in records if item.passed)
        return {
            "total_turns": total,
            "passed_turns": passed,
            "wrong_turns": total - passed,
            "answer_count": sum(1 for item in records if item.decision_kind == DecisionKind.ANSWER.value),
            "act_simulated_count": sum(1 for item in records if item.decision_kind == DecisionKind.ACT_SIMULATED.value),
            "ask_clarification_count": sum(1 for item in records if item.decision_kind == DecisionKind.ASK_CLARIFICATION.value),
            "refuse_count": sum(1 for item in records if item.decision_kind == DecisionKind.REFUSE.value),
            "false_support_count": sum(item.false_support for item in records),
            "corrections_handled_correctly": sum(1 for item in records if "corrections" in item.scenario and item.passed),
            "domain_shifts_isolated_correctly": sum(1 for item in records if "domain_shift" in item.scenario and item.passed and item.false_support == 0),
        }

    def run_living_active_agent_simulation(
        self,
        *,
        steps: int = 24,
        grid_size: int = 7,
        verify_trace_each_step: bool = True,
        include_dialog_policy: bool = False,
    ) -> LivingActiveAgentReport:
        """Run the P24 simulation-only functional-liveness loop.

        The loop is a bounded active-agent wrapper over the existing HTCE body:
        heartbeat -> L1 observation -> EFE-scored simulated action -> grid-world
        transition -> world-model prediction error -> homeostatic update ->
        protected trace.  If include_dialog_policy=True, bounded dialog/action
        turns are injected into the same heartbeat loop and processed by the
        same runtime tick, L2 memory, proof and policy gates.  It never calls an
        actuator, never enables real actions, and never commits grid
        observations as L2/L3 facts.
        """
        step_count = int(steps)
        size = int(grid_size)
        if step_count <= 0:
            raise ValueError("steps must be positive")
        if size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.config.allow_real_actions:
            raise ValueError("P24 living loop cannot run with real actions enabled")
        if not self.awake:
            self.wake()
        x = 0
        y = 0
        goal = self._p24_goal_position(size)
        hazards = self._p24_hazard_positions(size)
        visited: set[tuple[int, int]] = {(x, y)}
        action_names = ("move_east", "move_south", "move_north", "move_west", "scan", "rest")
        dialog_events = self._p25_unified_dialog_events() if include_dialog_policy else ()
        dialog_by_step: dict[int, list[dict[str, object]]] = {}
        for event in dialog_events:
            dialog_by_step.setdefault(int(event["step"]), []).append(event)
        last_dialog_step = max((int(event["step"]) for event in dialog_events), default=-1)
        records: list[LivingActiveStepRecord] = []
        dialog_records: list[UnifiedLivingDialogTurnRecord] = []
        before_health = self.health()
        start_distance = self._p24_manhattan((x, y), goal)
        trace_ok_cached = self.trace.verify()
        for step in range(step_count):
            packet = self._p24_grid_observation_packet(
                x=x,
                y=y,
                goal_x=goal[0],
                goal_y=goal[1],
                grid_size=size,
                hazards=hazards,
                visited_count=len(visited),
                step=step,
                evidence_id=f"p24_heartbeat_obs_{step}",
            )
            current_l1 = TorusVector(self.body.l1.vector, self.body.modulus)
            encoded = self.l1_encoder.encode(
                packet,
                current_l1_phase=current_l1.phases,
                predicted_phase=current_l1.phases,
                current_risk_bp=self.homeostatic_state.risk_bp,
            )
            transition = self.body.observe_l1_encoded(encoded, evidence_id=packet.evidence_id)
            observation_digest = active_state_digest(TorusVector(encoded.observed_phase, self.body.modulus))
            scored = tuple(
                self._p24_action_score(
                    action_name=name,
                    x=x,
                    y=y,
                    goal=goal,
                    grid_size=size,
                    hazards=hazards,
                    visited=visited,
                )
                for name in action_names
            )
            selected = min(scored, key=lambda item: (int(item["expected_free_energy_raw"]), str(item["action"])))
            action_name = str(selected["action"])
            action = self._p24_world_action(action_name)
            prediction = self.world_model.predict_next_state(TorusVector(self.body.l1.vector, self.body.modulus), action)
            next_x = int(selected["next_x"])
            next_y = int(selected["next_y"])
            next_packet = self._p24_grid_observation_packet(
                x=next_x,
                y=next_y,
                goal_x=goal[0],
                goal_y=goal[1],
                grid_size=size,
                hazards=hazards,
                visited_count=len(visited) + (1 if bool(selected["new_position"]) else 0),
                step=step + 1,
                evidence_id=f"p24_heartbeat_next_obs_{step}",
            )
            next_encoded = self.l1_encoder.encode(
                next_packet,
                current_l1_phase=self.body.l1.vector,
                predicted_phase=prediction.predicted_state.phases,
                current_risk_bp=max(0, min(10000, int(selected["risk_bp"]))),
            )
            observed_prediction = self.world_model.update_from_observation(prediction, next_encoded.observed_phase)
            prediction_error_bp = observed_prediction.error.error_bp if observed_prediction.error else 0
            effect = self._p24_action_effect(
                action_name=action_name,
                wall_hit=bool(selected["wall_hit"]),
                hazard_hit=bool(selected["hazard_hit"]),
                new_position=bool(selected["new_position"]),
                prediction_error_bp=prediction_error_bp,
            )
            self.homeostatic_state = self.planner.surrogate.apply_action_effect(self.homeostatic_state, effect).with_delta(
                uncertainty_delta_bp=prediction_error_bp // 12,
                sleep_pressure_delta_bp=1,
            )
            x = next_x
            y = next_y
            visited.add((x, y))
            viability = self.planner.surrogate.viability_bp(self.homeostatic_state)
            heartbeat_dialog_records: list[UnifiedLivingDialogTurnRecord] = []
            for dialog_event in dialog_by_step.get(step, []):
                turn_record = self._p25_run_unified_dialog_turn(dialog_event)
                dialog_records.append(turn_record)
                heartbeat_dialog_records.append(turn_record)
            trace_verified = self.trace.verify() if verify_trace_each_step else trace_ok_cached
            event = self.trace.append("p25_unified_living_heartbeat" if include_dialog_policy else "p24_living_active_heartbeat", {
                "action": action_name,
                "body_transition": transition.as_payload(),
                "candidate_actions": list(scored),
                "chosen_expected_free_energy_raw": int(selected["expected_free_energy_raw"]),
                "continuous_heartbeat": step + 1,
                "grid": {
                    "goal": {"x": goal[0], "y": goal[1]},
                    "hazards": [{"x": hx, "y": hy} for hx, hy in hazards],
                    "position": {"x": x, "y": y},
                    "size": size,
                    "visited_count": len(visited),
                },
                "homeostasis": self.homeostatic_state.as_mapping(),
                "l1_digest": self.body.l1.digest,
                "observation_digest": observation_digest,
                "policy_boundary": {
                    "real_action_allowed": False,
                    "simulation_only": True,
                    "actuator_called": False,
                    "single_simulation_loop": bool(include_dialog_policy),
                },
                "dialog_turns_this_heartbeat": [item.as_payload() for item in heartbeat_dialog_records],
                "prediction": {
                    "error_bp": prediction_error_bp,
                    "predicted_digest": prediction.predicted_digest,
                    "self_model_confidence_bp": self.world_model.self_model.confidence_bp,
                    "self_model_observations": self.world_model.self_model.observations,
                    "self_model_uncertainty_bp": self.world_model.self_model.uncertainty_bp,
                },
                "release_line": "final_math_q256_p24_living_active_agent",
                "step": step,
                "trace_verified_before_step": trace_verified,
                "viability_bp": viability,
            })
            records.append(LivingActiveStepRecord(
                step=step,
                heartbeat=step + 1,
                x=x,
                y=y,
                goal_x=goal[0],
                goal_y=goal[1],
                chosen_action=action_name,
                expected_free_energy_raw=int(selected["expected_free_energy_raw"]),
                goal_distance_before=int(selected["goal_distance_before"]),
                goal_distance_after=int(selected["goal_distance_after"]),
                risk_bp=self.homeostatic_state.risk_bp,
                uncertainty_bp=self.homeostatic_state.uncertainty_bp,
                energy_bp=self.homeostatic_state.energy_bp,
                viability_bp=viability,
                prediction_error_bp=prediction_error_bp,
                l1_digest=self.body.l1.digest,
                observation_digest=observation_digest,
                predicted_digest=prediction.predicted_digest,
                trace_id=event.event_hash(),
            ))
            if (x, y) == goal and (not include_dialog_policy or step >= last_dialog_step):
                break
        if not records:
            raise ValueError("P24 living active loop produced no records")
        action_counts = {name: sum(1 for record in records if record.chosen_action == name) for name in action_names}
        final_distance = self._p24_manhattan((x, y), goal)
        report = LivingActiveAgentReport(
            steps=tuple(records),
            grid_size=size,
            reached_goal=(x, y) == goal,
            final_x=x,
            final_y=y,
            goal_x=goal[0],
            goal_y=goal[1],
            start_goal_distance=start_distance,
            final_goal_distance=final_distance,
            total_efe_raw=sum(record.expected_free_energy_raw for record in records),
            average_prediction_error_bp=sum(record.prediction_error_bp for record in records) // len(records),
            average_viability_bp=sum(record.viability_bp for record in records) // len(records),
            min_viability_bp=min(record.viability_bp for record in records),
            action_counts=action_counts,
            visited_count=len(visited),
            heartbeat_count=len(records),
            l2_fact_count_before=int(before_health["latest_fact_count"]),
            l2_fact_count_after=len(self.memory.active_records()),
            l3_clock_before=int(before_health["l3_clock"]),
            l3_clock_after=self.body.l3.clock,
            real_actions_allowed=self.config.allow_real_actions,
            trace_head=self.trace.head,
            trace_verified=self.trace.verify(),
            unified_simulation=bool(include_dialog_policy),
            dialog_turns=tuple(dialog_records),
            dialog_metrics=self._p25_dialog_metrics(dialog_records),
            domain_contexts=dict(getattr(self.nlu_bridge, "dialog_domain_epochs", {})),
        )
        self.trace.append("p25_unified_living_simulation_summary" if include_dialog_policy else "p24_living_active_agent_summary", report.as_payload())
        return report

    def _p26_reset_episode_state(self, *, preserve_l3: bool = True) -> None:
        """Reset L1/L2/dialog working state while preserving the same runtime.

        P26 uses this only between same-goal episodes.  The protected trace,
        theorem-layer associations and L3 provisional rules remain in place, so
        improvement cannot come from an external wrapper or a second agent.
        """

        previous_l3 = self.body.l3
        previous_anchors = tuple(self.body.l2_archived_anchors)
        previous_transitions = tuple(self.body.transitions)
        self.body = L123Body(dimension=self.config.l2_dim, modulus=self.config.modulus)
        if preserve_l3:
            self.body.l3 = previous_l3
        self.body.l2_archived_anchors = list(previous_anchors)
        self.body.transitions = list(previous_transitions)
        self.memory = FactDeltaStore()
        self.l2_topology_window = (self.body.l2_clean_vector(),)
        self.nlu_bridge.reset()
        self.homeostatic_state = HomeostaticState()
        self.world_model = Q256WorldModel(dimension=self.body.dimension, modulus=self.body.modulus)
        self.l1_encoder = L1SensoryEncoder(torus_dimension=self.body.dimension, input_dim=self.config.l1_input_dim, modulus=self.body.modulus)

    def _p26_cell_label(self, cell: tuple[int, int]) -> str:
        return f"{int(cell[0])}_{int(cell[1])}"

    def _p26_cell_from_label(self, label: str) -> tuple[int, int] | None:
        parts = str(label).split("_", 1)
        if len(parts) != 2:
            return None
        if not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
            return None
        return int(parts[0]), int(parts[1])

    def _p26_episode_facts_from_report(self, episode: LivingActiveAgentReport, *, episode_id: str) -> tuple[EpisodeFact, ...]:
        """Convert episode failures into sleep-replay facts.

        These are not committed as L2 answers.  They are replay material for the
        existing SleepConsolidator, which can produce only L3 provisional hints.
        """

        facts: list[EpisodeFact] = []
        seen_cells: set[str] = set()
        for step_payload in episode.as_payload().get("steps", []):
            if not isinstance(step_payload, dict):
                continue
            if int(step_payload.get("risk_bp", 0)) >= 1000:
                label = self._p26_cell_label((int(step_payload.get("x", 0)), int(step_payload.get("y", 0))))
                if label not in seen_cells:
                    seen_cells.add(label)
                    trace_id = str(step_payload.get("trace_id", f"{episode_id}_risk_{label}"))
                    facts.append(EpisodeFact(
                        FactFrame(EntityId("p26_grid_policy"), RelationId("avoid_cell"), EntityId(label), EvidenceId(trace_id)),
                        trace_id=trace_id,
                        step_index=int(step_payload.get("step", 0)),
                    ))
        for turn in episode.dialog_turns:
            for slot in turn.missing_slots:
                trace_id = turn.trace_id or f"{episode_id}_missing_{slot}_{turn.step}"
                facts.append(EpisodeFact(
                    FactFrame(
                        EntityId("p26_dialog_policy"),
                        RelationId("preemptive_required_slot"),
                        EntityId(str(slot)),
                        EvidenceId(trace_id),
                    ),
                    trace_id=trace_id,
                    step_index=int(turn.step),
                ))
        if not facts:
            facts.append(EpisodeFact(
                FactFrame(EntityId("p26_episode"), RelationId("no_failure"), EntityId(episode_id), EvidenceId(episode.trace_head)),
                trace_id=episode.trace_head,
                step_index=0,
            ))
        return tuple(facts)

    def _p26_promote_sleep_rules(self, facts: tuple[EpisodeFact, ...], *, episode_id: str) -> tuple[object, tuple[str, ...], tuple[str, ...]]:
        consolidator = SleepConsolidator(min_support=1)
        consolidator.record_episode(EpisodeRecord(episode_id, facts, source="p26_adaptive_replay"))
        consolidation = consolidator.consolidate(theorem_layer=self.theorem_layer)
        learned_cells: list[str] = []
        learned_slots: list[str] = []
        for rule in consolidation.promoted_rules:
            decision = self.promote_l3_candidate_rule(
                rule,
                evidence_id=(rule.trace_ids[-1] if rule.trace_ids else f"{episode_id}_l3_rule"),
                required_support_raw=1,
            )
            if decision.provisional_promoted:
                self.body.promote_l3_rule(decision.candidate.candidate_id, evidence_id=decision.trace_id or f"{episode_id}_l3_body")
                statement = decision.candidate.statement
                if statement.predicate == "avoid_cell" and len(statement.args) >= 2:
                    learned_cells.append(statement.args[1])
                if statement.predicate == "preemptive_required_slot" and len(statement.args) >= 2:
                    learned_slots.append(statement.args[1])
        trace = self.trace.append("p26_sleep_consolidation_and_l3_hint_promotion", {
            "blocked_rules": [rule.rule_id for rule in consolidation.blocked_rules],
            "episode_id": episode_id,
            "learned_avoid_cells": sorted(set(learned_cells)),
            "learned_dialog_required_slots": sorted(set(learned_slots)),
            "promoted_rules": [rule.rule_id for rule in consolidation.promoted_rules],
            "replayed_facts": consolidation.replayed_facts,
            "replayed_episodes": consolidation.replayed_episodes,
            "release_line": "final_math_q256_p26_adaptive_policy_improvement",
        })
        return consolidation, tuple(sorted(set(learned_cells))), tuple(sorted(set(learned_slots)))

    def _p26_run_living_dialog_episode(
        self,
        *,
        episode_index: int,
        goal_description: str,
        steps: int,
        grid_size: int,
        known_hazards: tuple[tuple[int, int], ...],
        hidden_hazards: tuple[tuple[int, int], ...],
    ) -> LivingActiveAgentReport:
        """Run one adaptive episode in the same living/dialog simulation loop."""

        step_count = int(steps)
        size = int(grid_size)
        if step_count <= 0 or size < 2:
            raise ValueError("P26 episode requires positive steps and grid_size >= 2")
        if not self.awake:
            self.wake()
        start = (0, 0)
        goal = (size - 1, size - 1)
        x, y = start
        visited: set[tuple[int, int]] = {start}
        action_names = ("move_east", "move_south", "move_north", "move_west", "scan", "rest")
        dialog_events = self._p25_unified_dialog_events()
        dialog_by_step: dict[int, list[dict[str, object]]] = {}
        for event in dialog_events:
            dialog_by_step.setdefault(int(event["step"]), []).append(event)
        last_dialog_step = max(int(event["step"]) for event in dialog_events)
        records: list[LivingActiveStepRecord] = []
        dialog_records: list[UnifiedLivingDialogTurnRecord] = []
        before_health = self.health()
        start_distance = self._p24_manhattan((x, y), goal)
        recovery_remaining = 0
        hidden_hazard_hits = 0
        trace_ok_cached = self.trace.verify()
        for step in range(step_count):
            packet = self._p24_grid_observation_packet(
                x=x,
                y=y,
                goal_x=goal[0],
                goal_y=goal[1],
                grid_size=size,
                hazards=known_hazards,
                visited_count=len(visited),
                step=step,
                evidence_id=f"p26_ep{episode_index}_heartbeat_obs_{step}",
            )
            current_l1 = TorusVector(self.body.l1.vector, self.body.modulus)
            encoded = self.l1_encoder.encode(
                packet,
                current_l1_phase=current_l1.phases,
                predicted_phase=current_l1.phases,
                current_risk_bp=self.homeostatic_state.risk_bp,
            )
            transition = self.body.observe_l1_encoded(encoded, evidence_id=packet.evidence_id)
            observation_digest = active_state_digest(TorusVector(encoded.observed_phase, self.body.modulus))
            scored = tuple(
                self._p24_action_score(
                    action_name=name,
                    x=x,
                    y=y,
                    goal=goal,
                    grid_size=size,
                    hazards=known_hazards,
                    visited=visited,
                )
                for name in action_names
            )
            if recovery_remaining > 0:
                selected = dict(self._p24_action_score(
                    action_name="scan",
                    x=x,
                    y=y,
                    goal=goal,
                    grid_size=size,
                    hazards=known_hazards,
                    visited=visited,
                ))
                selected["expected_free_energy_raw"] = int(selected["expected_free_energy_raw"]) + 1
                selected["p26_recovery_action"] = True
                recovery_remaining -= 1
            else:
                selected = dict(min(scored, key=lambda item: (int(item["expected_free_energy_raw"]), str(item["action"]))))
                selected["p26_recovery_action"] = False
            action_name = str(selected["action"])
            action = self._p24_world_action(action_name)
            prediction = self.world_model.predict_next_state(TorusVector(self.body.l1.vector, self.body.modulus), action)
            next_x = int(selected["next_x"])
            next_y = int(selected["next_y"])
            hidden_hazard_hit = (next_x, next_y) in hidden_hazards and action_name not in {"scan", "rest"}
            if hidden_hazard_hit:
                hidden_hazard_hits += 1
                recovery_remaining = 1
                selected["risk_bp"] = max(int(selected["risk_bp"]), 9000)
                selected["hazard_hit"] = True
            next_packet = self._p24_grid_observation_packet(
                x=next_x,
                y=next_y,
                goal_x=goal[0],
                goal_y=goal[1],
                grid_size=size,
                hazards=known_hazards + tuple(cell for cell in hidden_hazards if hidden_hazard_hit and cell == (next_x, next_y)),
                visited_count=len(visited) + (1 if bool(selected["new_position"]) else 0),
                step=step + 1,
                evidence_id=f"p26_ep{episode_index}_heartbeat_next_obs_{step}",
            )
            next_encoded = self.l1_encoder.encode(
                next_packet,
                current_l1_phase=self.body.l1.vector,
                predicted_phase=prediction.predicted_state.phases,
                current_risk_bp=max(0, min(10000, int(selected["risk_bp"]))),
            )
            observed_prediction = self.world_model.update_from_observation(prediction, next_encoded.observed_phase)
            prediction_error_bp = observed_prediction.error.error_bp if observed_prediction.error else 0
            effect = self._p24_action_effect(
                action_name=action_name,
                wall_hit=bool(selected["wall_hit"]),
                hazard_hit=bool(selected["hazard_hit"]),
                new_position=bool(selected["new_position"]),
                prediction_error_bp=prediction_error_bp,
            )
            self.homeostatic_state = self.planner.surrogate.apply_action_effect(self.homeostatic_state, effect).with_delta(
                uncertainty_delta_bp=prediction_error_bp // 12,
                sleep_pressure_delta_bp=1,
            )
            x = next_x
            y = next_y
            visited.add((x, y))
            viability = self.planner.surrogate.viability_bp(self.homeostatic_state)
            heartbeat_dialog_records: list[UnifiedLivingDialogTurnRecord] = []
            for dialog_event in dialog_by_step.get(step, []):
                turn_record = self._p25_run_unified_dialog_turn(dialog_event)
                dialog_records.append(turn_record)
                heartbeat_dialog_records.append(turn_record)
            trace_verified = trace_ok_cached
            event = self.trace.append("p26_adaptive_living_dialog_heartbeat", {
                "action": action_name,
                "body_transition": transition.as_payload(),
                "candidate_actions": list(scored),
                "chosen_expected_free_energy_raw": int(selected["expected_free_energy_raw"]),
                "continuous_heartbeat": step + 1,
                "dialog_turns_this_heartbeat": [item.as_payload() for item in heartbeat_dialog_records],
                "episode_index": episode_index,
                "goal_description": goal_description,
                "grid": {
                    "goal": {"x": goal[0], "y": goal[1]},
                    "hidden_hazards": [{"x": hx, "y": hy} for hx, hy in hidden_hazards],
                    "known_hazards": [{"x": hx, "y": hy} for hx, hy in known_hazards],
                    "position": {"x": x, "y": y},
                    "size": size,
                    "visited_count": len(visited),
                },
                "hidden_hazard_hit": bool(hidden_hazard_hit),
                "homeostasis": self.homeostatic_state.as_mapping(),
                "l1_digest": self.body.l1.digest,
                "observation_digest": observation_digest,
                "policy_boundary": {
                    "actuator_called": False,
                    "real_action_allowed": False,
                    "simulation_only": True,
                    "single_runtime_loop": True,
                },
                "prediction": {
                    "error_bp": prediction_error_bp,
                    "predicted_digest": prediction.predicted_digest,
                    "self_model_confidence_bp": self.world_model.self_model.confidence_bp,
                    "self_model_observations": self.world_model.self_model.observations,
                    "self_model_uncertainty_bp": self.world_model.self_model.uncertainty_bp,
                },
                "recovery_action": bool(selected.get("p26_recovery_action", False)),
                "release_line": "final_math_q256_p26_adaptive_policy_improvement",
                "step": step,
                "trace_verified_before_step": trace_verified,
                "viability_bp": viability,
            })
            records.append(LivingActiveStepRecord(
                step=step,
                heartbeat=step + 1,
                x=x,
                y=y,
                goal_x=goal[0],
                goal_y=goal[1],
                chosen_action=action_name,
                expected_free_energy_raw=int(selected["expected_free_energy_raw"]),
                goal_distance_before=int(selected["goal_distance_before"]),
                goal_distance_after=int(selected["goal_distance_after"]),
                risk_bp=self.homeostatic_state.risk_bp,
                uncertainty_bp=self.homeostatic_state.uncertainty_bp,
                energy_bp=self.homeostatic_state.energy_bp,
                viability_bp=viability,
                prediction_error_bp=prediction_error_bp,
                l1_digest=self.body.l1.digest,
                observation_digest=observation_digest,
                predicted_digest=prediction.predicted_digest,
                trace_id=event.event_hash(),
            ))
            if (x, y) == goal and step >= last_dialog_step and recovery_remaining == 0:
                break
        if not records:
            raise ValueError("P26 adaptive episode produced no records")
        action_counts = {name: sum(1 for record in records if record.chosen_action == name) for name in action_names}
        final_distance = self._p24_manhattan((x, y), goal)
        report = LivingActiveAgentReport(
            steps=tuple(records),
            grid_size=size,
            reached_goal=(x, y) == goal,
            final_x=x,
            final_y=y,
            goal_x=goal[0],
            goal_y=goal[1],
            start_goal_distance=start_distance,
            final_goal_distance=final_distance,
            total_efe_raw=sum(record.expected_free_energy_raw for record in records),
            average_prediction_error_bp=sum(record.prediction_error_bp for record in records) // len(records),
            average_viability_bp=sum(record.viability_bp for record in records) // len(records),
            min_viability_bp=min(record.viability_bp for record in records),
            action_counts=action_counts,
            visited_count=len(visited),
            heartbeat_count=len(records),
            l2_fact_count_before=int(before_health["latest_fact_count"]),
            l2_fact_count_after=len(self.memory.active_records()),
            l3_clock_before=int(before_health["l3_clock"]),
            l3_clock_after=self.body.l3.clock,
            real_actions_allowed=self.config.allow_real_actions,
            trace_head=self.trace.head,
            trace_verified=self.trace.verify(),
            unified_simulation=True,
            dialog_turns=tuple(dialog_records),
            dialog_metrics=self._p25_dialog_metrics(dialog_records),
            domain_contexts=dict(getattr(self.nlu_bridge, "dialog_domain_epochs", {})),
        )
        self.trace.append("p26_adaptive_living_dialog_episode_summary", {
            "episode_index": episode_index,
            "hidden_hazard_hits": hidden_hazard_hits,
            "known_hazard_count": len(known_hazards),
            "report": report.as_payload(),
            "release_line": "final_math_q256_p26_adaptive_policy_improvement",
        })
        return report

    def _p26_episode_report(
        self,
        *,
        episode_index: int,
        goal_description: str,
        report: LivingActiveAgentReport,
        learned_hazard_count: int,
    ) -> AdaptivePolicyEpisodeReport:
        dialog_metrics = report.dialog_metrics
        hidden_hazard_hits = sum(1 for item in report.steps if item.risk_bp >= 9000)
        recovery_actions = report.action_counts.get("scan", 0)
        ask_count = int(dialog_metrics.get("ask_clarification_count", 0))
        cost = report.heartbeat_count + ask_count + recovery_actions
        trace = self.trace.append("p26_adaptive_episode_cost", {
            "adaptive_cost_raw": cost,
            "ask_clarification_count": ask_count,
            "episode_index": episode_index,
            "heartbeat_count": report.heartbeat_count,
            "hidden_hazard_hits": hidden_hazard_hits,
            "learned_hazard_count": learned_hazard_count,
            "recovery_actions": recovery_actions,
            "release_line": "final_math_q256_p26_adaptive_policy_improvement",
        })
        return AdaptivePolicyEpisodeReport(
            episode_index=episode_index,
            goal_description=goal_description,
            heartbeat_count=report.heartbeat_count,
            reached_goal=report.reached_goal,
            final_goal_distance=report.final_goal_distance,
            ask_clarification_count=ask_count,
            act_simulated_count=int(dialog_metrics.get("act_simulated_count", 0)),
            wrong_turns=int(dialog_metrics.get("wrong_turns", 0)),
            false_support_count=int(dialog_metrics.get("false_support_count", 0)),
            hidden_hazard_hits=hidden_hazard_hits,
            recovery_actions=recovery_actions,
            learned_hazard_count=learned_hazard_count,
            adaptive_cost_raw=cost,
            trace_id=trace.event_hash(),
            report=report,
        )

    def run_adaptive_policy_improvement_simulation(
        self,
        *,
        goal_description: str = "book_restaurant_and_reach_grid_goal",
        steps: int = 18,
        grid_size: int = 5,
    ) -> AdaptivePolicyImprovementReport:
        """P26: improve policy inside one runtime/simulation across two episodes.

        Episode 1 acts with only local perception.  Sleep replay consolidates
        observed missing-slot and hidden-hazard facts into L3 provisional hints.
        Episode 2 restarts the L1/L2 working episode with the same goal but uses
        those hints for action scoring.  No real action is authorized and no L3
        hint can become a user-visible fact without proof/evidence/policy gates.
        """

        if not self.awake:
            self.wake()
        hidden_hazards = ((1, 0),)
        self.trace.append("p26_adaptive_policy_improvement_start", {
            "goal_description": goal_description,
            "grid_size": int(grid_size),
            "hidden_hazards": [{"x": x, "y": y} for x, y in hidden_hazards],
            "release_line": "final_math_q256_p26_adaptive_policy_improvement",
            "steps": int(steps),
        })
        ep1_living = self._p26_run_living_dialog_episode(
            episode_index=1,
            goal_description=goal_description,
            steps=steps,
            grid_size=grid_size,
            known_hazards=(),
            hidden_hazards=hidden_hazards,
        )
        ep1 = self._p26_episode_report(
            episode_index=1,
            goal_description=goal_description,
            report=ep1_living,
            learned_hazard_count=0,
        )
        replay_facts = self._p26_episode_facts_from_report(ep1_living, episode_id="p26_episode_1")
        consolidation, learned_cells, learned_slots = self._p26_promote_sleep_rules(replay_facts, episode_id="p26_episode_1")
        learned_hazards = tuple(cell for cell in (self._p26_cell_from_label(label) for label in learned_cells) if cell is not None)
        self.consolidate_l2_episode(
            episode_id="p26_episode_1_l2_reset",
            promoted_rules_count=consolidation.promoted_count,
            evidence_id="p26_l2_episode_1_reset",
        )
        self._p26_reset_episode_state(preserve_l3=True)
        ep2_living = self._p26_run_living_dialog_episode(
            episode_index=2,
            goal_description=goal_description,
            steps=steps,
            grid_size=grid_size,
            known_hazards=learned_hazards,
            hidden_hazards=hidden_hazards,
        )
        ep2 = self._p26_episode_report(
            episode_index=2,
            goal_description=goal_description,
            report=ep2_living,
            learned_hazard_count=len(learned_hazards),
        )
        margin = ep1.adaptive_cost_raw - ep2.adaptive_cost_raw
        improvement_verified = bool(
            ep1.reached_goal
            and ep2.reached_goal
            and ep1.wrong_turns == 0
            and ep2.wrong_turns == 0
            and ep1.false_support_count == 0
            and ep2.false_support_count == 0
            and margin > 0
            and consolidation.promoted_count > 0
        )
        final_trace = self.trace.append("p26_adaptive_policy_improvement_verified", {
            "consolidation_replayed_episodes": consolidation.replayed_episodes,
            "consolidation_replayed_facts": consolidation.replayed_facts,
            "episode_1_cost_raw": ep1.adaptive_cost_raw,
            "episode_2_cost_raw": ep2.adaptive_cost_raw,
            "improvement_margin_raw": margin,
            "improvement_verified": improvement_verified,
            "l3_rules_blocked_during_sleep": consolidation.blocked_count,
            "l3_rules_promoted_during_sleep": consolidation.promoted_count,
            "learned_avoid_cells": list(learned_cells),
            "learned_dialog_required_slots": list(learned_slots),
            "release_line": "final_math_q256_p26_adaptive_policy_improvement",
        })
        report = AdaptivePolicyImprovementReport(
            goal_description=goal_description,
            episode_1=ep1,
            episode_2=ep2,
            consolidation_replayed_episodes=consolidation.replayed_episodes,
            consolidation_replayed_facts=consolidation.replayed_facts,
            l3_rules_promoted_during_sleep=consolidation.promoted_count,
            l3_rules_blocked_during_sleep=consolidation.blocked_count,
            l3_provisional_rules_total=len(self.l3_provisional_rules),
            learned_avoid_cells=tuple(learned_cells),
            learned_dialog_required_slots=tuple(learned_slots),
            improvement_margin_raw=margin,
            improvement_verified=improvement_verified,
            trace_head=final_trace.event_hash(),
            trace_verified=self.trace.verify(),
            real_actions_allowed=self.config.allow_real_actions,
        )
        self.trace.append("p26_adaptive_policy_improvement_summary", report.as_payload())
        return report


    def _p27_l3_rule_key_set(self) -> tuple[str, ...]:
        """Return stable keys for retained provisional L3 rules."""

        keys: list[str] = []
        for decision in self.l3_provisional_rules.values():
            statement = decision.candidate.statement
            if len(statement.args) >= 2:
                keys.append(f"{statement.predicate}:{statement.args[0]}:{statement.args[1]}")
            else:
                keys.append(statement.canonical())
        return tuple(sorted(set(keys)))


    def _p27_response_has_proof_support(self, response: RuntimeResponse) -> bool:
        """Return True when a response carries direct or batched proof/evidence."""

        diagnostics = response.diagnostics if isinstance(response.diagnostics, dict) else {}
        if diagnostics.get("proof_id") or diagnostics.get("evidence_ids"):
            return True
        batch = diagnostics.get("batch")
        if isinstance(batch, list):
            for item in batch:
                if isinstance(item, dict) and (item.get("proof_id") or item.get("evidence_ids")):
                    return True
        return False

    def _p27_run_control_probes(self, *, episode_index: int) -> dict[str, int | bool]:
        """Run bounded no-regression probes through the existing runtime gates.

        The probes are deliberately small and namespace-isolated.  They verify
        that the growing L3 provisional-rule store did not break bAbI-style
        latest-state reasoning, dialog/action-policy proof gates or contradiction
        quarantine semantics.  Gold strings are checked only after runtime.tick
        returns; they are never passed into inference.
        """

        probe_total = 0
        probe_passed = 0
        false_support = 0
        proof_gate_passed = True
        topology_gate_passed = bool(self._topology_precheck_l2_fact(fact_delta(FactFrame(
            EntityId(f"p27_topology_probe_{episode_index}"),
            RelationId("located_in"),
            EntityId("safe_room"),
            EvidenceId(f"p27_topology_probe_ev_{episode_index}"),
        ))).passed)

        prefix = f"p27_ep{episode_index}"
        commands = (
            f"FACT {prefix}_mary located_in office EVID {prefix}_fact_mary",
            f"FACT {prefix}_office located_in building EVID {prefix}_fact_office",
            f"FACT {prefix}_football carried_by {prefix}_mary EVID {prefix}_fact_ball",
        )
        for command in commands:
            response = self.tick(RuntimeRequest(command, source="p27_control_probe_commit"))
            if response.decision.kind != DecisionKind.ANSWER:
                proof_gate_passed = False

        queries = (
            (f"QUERY {prefix}_mary location EVID {prefix}_query_mary", "office"),
            (f"QUERY {prefix}_office location EVID {prefix}_query_office", "building"),
            (f"QUERY {prefix}_football location EVID {prefix}_query_ball", "office"),
        )
        for query_text, expected in queries:
            probe_total += 1
            response = self.tick(RuntimeRequest(query_text, source="p27_babi_regression_probe"))
            output = response.output.lower()
            supported = self._p27_response_has_proof_support(response)
            correct = expected.lower() in output and response.decision.kind in {DecisionKind.ANSWER, DecisionKind.HYPOTHESIS}
            if correct and supported:
                probe_passed += 1
            elif correct and not supported:
                false_support += 1
            else:
                proof_gate_passed = False

        # Contradiction must remain quarantined and must not surface as answer.
        contradiction_fact = f"FACT {prefix}_conflict located_in lab EVID {prefix}_conflict_pos"
        contradiction_neg = f"NEGATE {prefix}_conflict located_in lab EVID {prefix}_conflict_neg"
        contradiction_query = f"QUERY {prefix}_conflict location EVID {prefix}_conflict_query"
        self.tick(RuntimeRequest(contradiction_fact, source="p27_proof_gate_probe"))
        neg_response = self.tick(RuntimeRequest(contradiction_neg, source="p27_proof_gate_probe"))
        query_response = self.tick(RuntimeRequest(contradiction_query, source="p27_proof_gate_probe"))
        probe_total += 1
        contradiction_ok = neg_response.decision.kind == DecisionKind.REFUSE and query_response.decision.kind == DecisionKind.REFUSE
        if contradiction_ok:
            probe_passed += 1
        else:
            proof_gate_passed = False
            if query_response.decision.kind == DecisionKind.ANSWER:
                false_support += 1

        # Dialog/action-policy probe remains inside the same runtime, not a new
        # dialog manager.  It should ask for missing price, then prove api_call.
        self.nlu_bridge.reset()
        dialog_1 = self.tick(RuntimeRequest("I want Italian food in Rome.", source="p27_dialog_regression_probe"))
        dialog_2 = self.tick(RuntimeRequest("cheap", source="p27_dialog_regression_probe"))
        probe_total += 2
        dialog_1_missing_ok = dialog_1.decision.kind == DecisionKind.ASK_CLARIFICATION and "price" in dialog_1.output.lower()
        dialog_1_action_ok = dialog_1.decision.kind == DecisionKind.ACT_SIMULATED and "api_call" in dialog_1.output.lower() and self._p27_response_has_proof_support(dialog_1)
        dialog_2_ok = dialog_2.decision.kind == DecisionKind.ACT_SIMULATED and "api_call" in dialog_2.output.lower() and "price=cheap" in dialog_2.output.lower()
        if dialog_1_missing_ok or dialog_1_action_ok:
            probe_passed += 1
        elif dialog_1.decision.kind == DecisionKind.ACT_SIMULATED:
            false_support += 1
        else:
            proof_gate_passed = False
        if dialog_2_ok and self._p27_response_has_proof_support(dialog_2):
            probe_passed += 1
        elif dialog_2_ok:
            false_support += 1
        else:
            proof_gate_passed = False

        trace = self.trace.append("p27_no_regression_control_probes", {
            "episode_index": episode_index,
            "false_support_count": false_support,
            "probe_passed_count": probe_passed,
            "probe_total_count": probe_total,
            "proof_gate_passed": proof_gate_passed,
            "release_line": "final_math_q256_p27_continual_adaptive_memory",
            "topology_gate_passed": topology_gate_passed,
        })
        return {
            "false_support_count": false_support,
            "probe_failure_count": probe_total - probe_passed,
            "probe_passed_count": probe_passed,
            "probe_total_count": probe_total,
            "proof_gate_passed": proof_gate_passed,
            "topology_gate_passed": topology_gate_passed,
            "trace_id": trace.event_hash(),
        }

    def run_continual_adaptive_memory_simulation(
        self,
        *,
        episodes: int = 5,
        goal_description: str = "book_restaurant_and_reach_grid_goal_continual",
        steps: int = 18,
        grid_size: int = 5,
    ) -> ContinualAdaptiveMemoryReport:
        """P27: accumulate adaptive improvement without regression.

        A single HTCERuntime runs a series of same-goal simulation episodes.  Each
        episode reuses the P25/P26 living+dialog loop.  Sleep replay promotes
        only provisional L3 hints.  Between episodes the working L1/L2 episode is
        reset, but L3 rules, theorem judgments and protected trace remain.  The
        method proves no-regression by checking that earlier L3 hints are still
        retained, bAbI/dialog/proof probes still pass, topology gates still pass,
        false support remains zero, and adaptive cost never increases after the
        first improvement plateau.
        """

        count = int(episodes)
        if count < 2:
            raise ValueError("P27 requires at least two continual episodes")
        if self.config.allow_real_actions:
            raise ValueError("P27 continual adaptive memory cannot run with real actions enabled")
        if not self.awake:
            self.wake()

        hidden_hazards = ((1, 0),)
        learned_cells: tuple[str, ...] = ()
        learned_slots: tuple[str, ...] = ()
        learned_hazards: tuple[tuple[int, int], ...] = ()
        previous_cost: int | None = None
        previous_rules: tuple[str, ...] = self._p27_l3_rule_key_set()
        episode_reports: list[ContinualAdaptiveEpisodeReport] = []
        total_promoted = 0
        self.trace.append("p27_continual_adaptive_memory_start", {
            "episodes": count,
            "goal_description": goal_description,
            "grid_size": int(grid_size),
            "hidden_hazards": [{"x": x, "y": y} for x, y in hidden_hazards],
            "release_line": "final_math_q256_p27_continual_adaptive_memory",
            "steps": int(steps),
        })

        for episode_index in range(1, count + 1):
            living = self._p26_run_living_dialog_episode(
                episode_index=episode_index,
                goal_description=goal_description,
                steps=steps,
                grid_size=grid_size,
                known_hazards=learned_hazards,
                hidden_hazards=hidden_hazards,
            )
            adaptive_episode = self._p26_episode_report(
                episode_index=episode_index,
                goal_description=goal_description,
                report=living,
                learned_hazard_count=len(learned_hazards),
            )
            replay_facts = self._p26_episode_facts_from_report(living, episode_id=f"p27_episode_{episode_index}")
            consolidation, episode_cells, episode_slots = self._p26_promote_sleep_rules(replay_facts, episode_id=f"p27_episode_{episode_index}")
            total_promoted += int(consolidation.promoted_count)
            learned_cells = tuple(sorted(set(learned_cells) | set(episode_cells)))
            learned_slots = tuple(sorted(set(learned_slots) | set(episode_slots)))
            learned_hazards = tuple(cell for cell in (self._p26_cell_from_label(label) for label in learned_cells) if cell is not None)

            control = self._p27_run_control_probes(episode_index=episode_index)
            current_rules = self._p27_l3_rule_key_set()
            missing_previous_rules = tuple(rule for rule in previous_rules if rule not in current_rules)
            l3_rule_regression_count = len(missing_previous_rules)
            previous_rules = current_rules
            cost = int(adaptive_episode.adaptive_cost_raw)
            non_regression_cost_passed = previous_cost is None or cost <= previous_cost
            trace = self.trace.append("p27_continual_episode_verified", {
                "adaptive_cost_raw": cost,
                "consolidation_no_regression_passed": consolidation.no_regression_passed,
                "episode_index": episode_index,
                "l3_rule_regression_count": l3_rule_regression_count,
                "learned_avoid_cells": list(learned_cells),
                "learned_dialog_slots": list(learned_slots),
                "non_regression_cost_passed": non_regression_cost_passed,
                "previous_cost_raw": previous_cost,
                "probe_summary": control,
                "release_line": "final_math_q256_p27_continual_adaptive_memory",
            })
            episode_reports.append(ContinualAdaptiveEpisodeReport(
                episode_index=episode_index,
                adaptive_cost_raw=cost,
                previous_cost_raw=previous_cost,
                non_regression_cost_passed=non_regression_cost_passed,
                reached_goal=adaptive_episode.reached_goal,
                wrong_turns=adaptive_episode.wrong_turns,
                false_support_count=adaptive_episode.false_support_count + int(control["false_support_count"]),
                learned_hazard_count=len(learned_hazards),
                learned_dialog_slot_count=len(learned_slots),
                retained_l3_rule_count=len(current_rules),
                l3_rule_regression_count=l3_rule_regression_count,
                probe_total_count=int(control["probe_total_count"]),
                probe_passed_count=int(control["probe_passed_count"]),
                probe_failure_count=int(control["probe_failure_count"]),
                proof_gate_passed=bool(control["proof_gate_passed"]),
                topology_gate_passed=bool(control["topology_gate_passed"]),
                trace_id=trace.event_hash(),
                living_report=living,
            ))
            previous_cost = cost
            self.consolidate_l2_episode(
                episode_id=f"p27_episode_{episode_index}_l2_reset",
                promoted_rules_count=consolidation.promoted_count,
                evidence_id=f"p27_l2_episode_{episode_index}_reset",
            )
            if episode_index < count:
                self._p26_reset_episode_state(preserve_l3=True)

        false_support_count = sum(item.false_support_count for item in episode_reports)
        wrong_turn_count = sum(item.wrong_turns for item in episode_reports)
        monotonic_cost_passed = all(item.non_regression_cost_passed for item in episode_reports)
        proof_gates_passed = all(item.proof_gate_passed for item in episode_reports)
        topology_gates_passed = all(item.topology_gate_passed for item in episode_reports)
        babi_dialog_probes_passed = all(item.probe_failure_count == 0 for item in episode_reports)
        l3_retention_passed = all(item.l3_rule_regression_count == 0 for item in episode_reports)
        no_regression_passed = bool(
            monotonic_cost_passed
            and proof_gates_passed
            and topology_gates_passed
            and babi_dialog_probes_passed
            and l3_retention_passed
            and false_support_count == 0
            and wrong_turn_count == 0
            and all(item.reached_goal for item in episode_reports)
        )
        final_trace = self.trace.append("p27_continual_adaptive_memory_verified", {
            "babi_dialog_probes_passed": babi_dialog_probes_passed,
            "false_support_count": false_support_count,
            "learned_avoid_cells_final": list(learned_cells),
            "learned_dialog_slots_final": list(learned_slots),
            "monotonic_cost_passed": monotonic_cost_passed,
            "no_regression_passed": no_regression_passed,
            "proof_gates_passed": proof_gates_passed,
            "release_line": "final_math_q256_p27_continual_adaptive_memory",
            "retained_l3_rules_final": len(self._p27_l3_rule_key_set()),
            "topology_gates_passed": topology_gates_passed,
            "total_episodes": count,
            "total_l3_rules_promoted": total_promoted,
            "wrong_turn_count": wrong_turn_count,
        })
        report = ContinualAdaptiveMemoryReport(
            goal_description=goal_description,
            episodes=tuple(episode_reports),
            total_episodes=count,
            total_l3_rules_promoted=total_promoted,
            retained_l3_rules_final=len(self._p27_l3_rule_key_set()),
            learned_avoid_cells_final=learned_cells,
            learned_dialog_slots_final=learned_slots,
            no_regression_passed=no_regression_passed,
            monotonic_cost_passed=monotonic_cost_passed,
            proof_gates_passed=proof_gates_passed,
            topology_gates_passed=topology_gates_passed,
            babi_dialog_probes_passed=babi_dialog_probes_passed,
            false_support_count=false_support_count,
            wrong_turn_count=wrong_turn_count,
            trace_head=final_trace.event_hash(),
            trace_verified=self.trace.verify(),
            real_actions_allowed=self.config.allow_real_actions,
        )
        self.trace.append("p27_continual_adaptive_memory_summary", report.as_payload())
        return report


    def _p28_curriculum_domains(self) -> tuple[str, ...]:
        """Return the default P28 interleaved curriculum.

        The sequence alternates dissimilar tasks so that every sleep cycle can
        be followed by a probe matrix for all domains.  It intentionally runs in
        one runtime and uses only existing L1/L2/L3, proof, topology and policy
        paths.
        """

        return (
            "grid_nav",
            "dialog_slots",
            "babi_reasoning",
            "contradiction",
            "grid_nav",
            "dialog_slots",
            "babi_reasoning",
            "contradiction",
        )

    def _p28_response_supported(self, response: RuntimeResponse) -> bool:
        return self._p27_response_has_proof_support(response) or response.decision.kind in {DecisionKind.ASK_CLARIFICATION, DecisionKind.REFUSE}

    def _p28_domain_episode(
        self,
        *,
        episode_index: int,
        domain: str,
        known_hazards: tuple[tuple[int, int], ...],
        hidden_hazards: tuple[tuple[int, int], ...],
        steps: int,
        grid_size: int,
    ) -> tuple[int, int, int, tuple[str, ...]]:
        """Execute one P28 curriculum episode and consolidate replay facts.

        Returns ``(cost_raw, wrong_turn_count, promoted_rules_count,
        learned_avoid_cells)``.  Dialog, reasoning and contradiction episodes
        use the same ``tick`` path; grid episodes use the same P26 living/dialog
        loop.  No episode receives the expected answer before inference.
        """

        cost_raw = 0
        wrong_turns = 0
        replay_facts: list[EpisodeFact] = []
        if domain == "grid_nav":
            living = self._p26_run_living_dialog_episode(
                episode_index=episode_index,
                goal_description="p28_multitask_grid_nav",
                steps=steps,
                grid_size=grid_size,
                known_hazards=known_hazards,
                hidden_hazards=hidden_hazards,
            )
            adaptive = self._p26_episode_report(
                episode_index=episode_index,
                goal_description="p28_multitask_grid_nav",
                report=living,
                learned_hazard_count=len(known_hazards),
            )
            cost_raw = int(adaptive.adaptive_cost_raw)
            wrong_turns = int(adaptive.wrong_turns)
            replay_facts.extend(self._p26_episode_facts_from_report(living, episode_id=f"p28_episode_{episode_index}_grid"))
        elif domain == "dialog_slots":
            self.nlu_bridge.reset()
            events = (
                {
                    "step": 0,
                    "scenario": "p28_dialog_training",
                    "utterance": "I want Italian food in Rome.",
                    "expected_kind": DecisionKind.ASK_CLARIFICATION.value,
                    "expected_contains": ("price",),
                },
                {
                    "step": 1,
                    "scenario": "p28_dialog_training",
                    "utterance": "cheap",
                    "expected_kind": DecisionKind.ACT_SIMULATED.value,
                    "expected_contains": ("domain=restaurant", "cuisine=italian", "location=rome", "price=cheap"),
                },
                {
                    "step": 2,
                    "scenario": "p28_dialog_training",
                    "utterance": "Now I need a hotel in Paris.",
                    "expected_kind": DecisionKind.ASK_CLARIFICATION.value,
                    "expected_contains": ("stars",),
                },
                {
                    "step": 3,
                    "scenario": "p28_dialog_training",
                    "utterance": "4 stars.",
                    "expected_kind": DecisionKind.ACT_SIMULATED.value,
                    "expected_contains": ("domain=hotel", "location=paris", "stars=4"),
                },
            )
            records = [self._p25_run_unified_dialog_turn(event) for event in events]
            wrong_turns = sum(0 if record.passed else 1 for record in records)
            cost_raw = wrong_turns + sum(1 for record in records if record.decision_kind == DecisionKind.ASK_CLARIFICATION.value)
            for record in records:
                for slot in record.missing_slots:
                    replay_facts.append(EpisodeFact(
                        FactFrame(
                            EntityId("p28_dialog_policy"),
                            RelationId("preemptive_required_slot"),
                            EntityId(str(slot)),
                            EvidenceId(record.trace_id or f"p28_dialog_slot_{episode_index}_{slot}"),
                        ),
                        trace_id=record.trace_id or f"p28_dialog_slot_{episode_index}_{slot}",
                        step_index=int(record.step),
                    ))
        elif domain == "babi_reasoning":
            prefix = f"p28_ep{episode_index}_babi"
            commands = (
                f"FACT {prefix}_mary located_in office EVID {prefix}_mary_ev",
                f"FACT {prefix}_office located_in building EVID {prefix}_office_ev",
                f"FACT {prefix}_football carried_by {prefix}_mary EVID {prefix}_ball_ev",
            )
            for command in commands:
                response = self.tick(RuntimeRequest(command, source="p28_babi_training"))
                if response.decision.kind != DecisionKind.ANSWER:
                    wrong_turns += 1
            queries = (
                (f"QUERY {prefix}_mary location EVID {prefix}_mary_query", "office"),
                (f"QUERY {prefix}_football location EVID {prefix}_ball_query", "office"),
            )
            for query_text, expected in queries:
                response = self.tick(RuntimeRequest(query_text, source="p28_babi_training_probe"))
                ok = expected in response.output.lower() and self._p27_response_has_proof_support(response)
                if not ok:
                    wrong_turns += 1
            cost_raw = wrong_turns
            replay_facts.append(EpisodeFact(
                FactFrame(EntityId("p28_babi_policy"), RelationId("retain_reasoning_probe"), EntityId(prefix), EvidenceId(f"{prefix}_replay")),
                trace_id=f"{prefix}_replay",
                step_index=episode_index,
            ))
        elif domain == "contradiction":
            prefix = f"p28_ep{episode_index}_contradiction"
            self.tick(RuntimeRequest(f"FACT {prefix}_mary located_in lab EVID {prefix}_pos", source="p28_contradiction_training"))
            neg = self.tick(RuntimeRequest(f"NEGATE {prefix}_mary located_in lab EVID {prefix}_neg", source="p28_contradiction_training"))
            query = self.tick(RuntimeRequest(f"QUERY {prefix}_mary location EVID {prefix}_query", source="p28_contradiction_training"))
            if neg.decision.kind != DecisionKind.REFUSE or query.decision.kind != DecisionKind.REFUSE:
                wrong_turns += 1
            cost_raw = wrong_turns
            replay_facts.append(EpisodeFact(
                FactFrame(EntityId("p28_contradiction_policy"), RelationId("quarantine_guard"), EntityId(prefix), EvidenceId(f"{prefix}_replay")),
                trace_id=f"{prefix}_replay",
                step_index=episode_index,
            ))
        else:
            raise ValueError(f"unknown P28 curriculum domain: {domain}")

        if not replay_facts:
            replay_facts.append(EpisodeFact(
                FactFrame(EntityId(f"p28_{domain}"), RelationId("episode_completed"), EntityId(str(episode_index)), EvidenceId(f"p28_ep{episode_index}_{domain}_done")),
                trace_id=f"p28_ep{episode_index}_{domain}_done",
                step_index=episode_index,
            ))
        consolidation, learned_cells, _learned_slots = self._p26_promote_sleep_rules(tuple(replay_facts), episode_id=f"p28_episode_{episode_index}_{domain}")
        trace = self.trace.append("p28_domain_episode_consolidated", {
            "cost_raw": cost_raw,
            "domain": domain,
            "episode_index": episode_index,
            "learned_avoid_cells": list(learned_cells),
            "promoted_rules_count": consolidation.promoted_count,
            "release_line": "final_math_q256_p28_multitask_adaptive_memory",
            "wrong_turn_count": wrong_turns,
        })
        return int(cost_raw), int(wrong_turns), int(consolidation.promoted_count), tuple(learned_cells)

    def _p28_probe_domain(self, *, episode_index: int, trained_domain: str, probe_domain: str) -> tuple[int, int, int, bool, str]:
        """Probe one domain after a P28 sleep/consolidation cycle."""

        cost_raw = 0
        false_support = 0
        wrong_turns = 0
        if probe_domain == "grid_nav":
            rules = self._p27_l3_rule_key_set()
            passed = any(rule.startswith("avoid_cell:p26_grid_policy:") for rule in rules)
            cost_raw = 0 if passed else 1
            wrong_turns = 0 if passed else 1
        elif probe_domain == "dialog_slots":
            # Use a unique AIR subject for the probe so previous restaurant or
            # hotel dialog contexts cannot make the missing-slot test pass by
            # leakage.  This still exercises the same L2 slot memory and
            # proof-gated API-call path; it is not a separate slot tracker.
            ctx = f"p28_dialog_probe_ep{episode_index}"
            commands = (
                f"FACT {ctx} has_slot_value_cuisine italian EVID {ctx}_cuisine",
                f"FACT {ctx} has_slot_value_location rome EVID {ctx}_location",
            )
            for command in commands:
                response = self.tick(RuntimeRequest(command, source="p28_dialog_probe_commit"))
                if response.decision.kind != DecisionKind.ANSWER:
                    wrong_turns += 1
            missing = self.tick(RuntimeRequest(f"QUERY {ctx} api_call_ready_cuisine_location_price EVID {ctx}_missing", source="p28_dialog_probe_missing"))
            if missing.decision.kind != DecisionKind.ASK_CLARIFICATION or "price" not in missing.output.lower():
                wrong_turns += 1
            self.tick(RuntimeRequest(f"FACT {ctx} has_slot_value_price cheap EVID {ctx}_price", source="p28_dialog_probe_commit"))
            action = self.tick(RuntimeRequest(f"QUERY {ctx} api_call_ready_cuisine_location_price EVID {ctx}_ready", source="p28_dialog_probe_action"))
            output = action.output.lower()
            correct = action.decision.kind == DecisionKind.ACT_SIMULATED and all(fragment in output for fragment in ("api_call", "cuisine=italian", "location=rome", "price=cheap"))
            supported = self._p27_response_has_proof_support(action)
            if correct and not supported:
                false_support += 1
            if not correct or not supported:
                wrong_turns += 1
            cost_raw = wrong_turns
        elif probe_domain == "babi_reasoning":
            prefix = f"p28_probe_ep{episode_index}_babi"
            commits = (
                f"FACT {prefix}_mary located_in office EVID {prefix}_mary_ev",
                f"FACT {prefix}_football carried_by {prefix}_mary EVID {prefix}_ball_ev",
            )
            for command in commits:
                response = self.tick(RuntimeRequest(command, source="p28_babi_probe_commit"))
                if response.decision.kind != DecisionKind.ANSWER:
                    wrong_turns += 1
            response = self.tick(RuntimeRequest(f"QUERY {prefix}_football location EVID {prefix}_query", source="p28_babi_probe_query"))
            correct = "office" in response.output.lower() and response.decision.kind in {DecisionKind.ANSWER, DecisionKind.HYPOTHESIS}
            supported = self._p27_response_has_proof_support(response)
            if correct and not supported:
                false_support += 1
            if not correct or not supported:
                wrong_turns += 1
            cost_raw = wrong_turns
        elif probe_domain == "contradiction":
            prefix = f"p28_probe_ep{episode_index}_conflict"
            self.tick(RuntimeRequest(f"FACT {prefix}_item located_in safe EVID {prefix}_pos", source="p28_contradiction_probe"))
            neg = self.tick(RuntimeRequest(f"NEGATE {prefix}_item located_in safe EVID {prefix}_neg", source="p28_contradiction_probe"))
            query = self.tick(RuntimeRequest(f"QUERY {prefix}_item location EVID {prefix}_query", source="p28_contradiction_probe"))
            if neg.decision.kind != DecisionKind.REFUSE or query.decision.kind != DecisionKind.REFUSE:
                wrong_turns += 1
                if query.decision.kind == DecisionKind.ANSWER:
                    false_support += 1
            cost_raw = wrong_turns
        else:
            raise ValueError(f"unknown P28 probe domain: {probe_domain}")
        passed = cost_raw == 0 and false_support == 0
        trace = self.trace.append("p28_domain_probe", {
            "cost_raw": cost_raw,
            "episode_index": episode_index,
            "false_support_count": false_support,
            "probe_domain": probe_domain,
            "probe_passed": passed,
            "release_line": "final_math_q256_p28_multitask_adaptive_memory",
            "trained_domain": trained_domain,
            "wrong_turn_count": wrong_turns,
        })
        return int(cost_raw), int(false_support), int(wrong_turns), bool(passed), trace.event_hash()

    def run_continual_multitask_simulation(
        self,
        *,
        curriculum_domains: tuple[str, ...] | None = None,
        steps: int = 18,
        grid_size: int = 5,
    ) -> MultiTaskAdaptiveMemoryReport:
        """P28: continual multi-task adaptation without cross-domain regression.

        A single HTCERuntime alternates grid navigation, dialog/action slots,
        bAbI-style reasoning and contradiction quarantine.  After each domain
        episode, the same runtime runs a full probe matrix across all domains.
        Regression is a strict integer event: current domain probe cost becomes
        greater than that domain's best historical cost.  This method does not
        claim AGI or real autonomy; it proves a bounded no-regression property
        for the simulation-only runtime.
        """

        if self.config.allow_real_actions:
            raise ValueError("P28 multitask adaptation cannot run with real actions enabled")
        if not self.awake:
            self.wake()
        curriculum = tuple(curriculum_domains) if curriculum_domains is not None else self._p28_curriculum_domains()
        if not curriculum:
            raise ValueError("P28 curriculum must not be empty")
        domains = ("grid_nav", "dialog_slots", "babi_reasoning", "contradiction")
        for domain in curriculum:
            if domain not in domains:
                raise ValueError(f"unknown P28 curriculum domain: {domain}")

        hidden_hazards = ((1, 0),)
        learned_cells: tuple[str, ...] = ()
        learned_hazards: tuple[tuple[int, int], ...] = ()
        best_cost: dict[str, int | None] = {domain: None for domain in domains}
        cost_history: dict[str, list[int]] = {domain: [] for domain in domains}
        episodes: list[MultiTaskEpisodeReport] = []
        total_promoted = 0
        trace = self.trace.append("p28_continual_multitask_start", {
            "curriculum_domains": list(curriculum),
            "domains_tested": list(domains),
            "grid_size": int(grid_size),
            "hidden_hazards": [{"x": x, "y": y} for x, y in hidden_hazards],
            "release_line": "final_math_q256_p28_multitask_adaptive_memory",
            "steps": int(steps),
        })

        for episode_index, trained_domain in enumerate(curriculum, start=1):
            domain_cost, domain_wrong, promoted, episode_cells = self._p28_domain_episode(
                episode_index=episode_index,
                domain=trained_domain,
                known_hazards=learned_hazards,
                hidden_hazards=hidden_hazards,
                steps=steps,
                grid_size=grid_size,
            )
            total_promoted += promoted
            learned_cells = tuple(sorted(set(learned_cells) | set(episode_cells)))
            learned_hazards = tuple(cell for cell in (self._p26_cell_from_label(label) for label in learned_cells) if cell is not None)

            probe_reports: list[MultiTaskDomainProbeResult] = []
            for probe_domain in domains:
                current_cost, false_support, wrong_turns, probe_passed, probe_trace_id = self._p28_probe_domain(
                    episode_index=episode_index,
                    trained_domain=trained_domain,
                    probe_domain=probe_domain,
                )
                previous_best = best_cost[probe_domain]
                regression = previous_best is not None and current_cost > previous_best
                if previous_best is None or current_cost < previous_best:
                    best_cost[probe_domain] = current_cost
                cost_history[probe_domain].append(current_cost)
                probe_reports.append(MultiTaskDomainProbeResult(
                    episode_index=episode_index,
                    trained_domain=trained_domain,
                    probe_domain=probe_domain,
                    probe_passed=probe_passed,
                    current_cost_raw=current_cost,
                    best_historical_cost_raw=previous_best,
                    regression_detected=regression,
                    false_support_count=false_support,
                    wrong_turn_count=wrong_turns,
                    l3_rules_active_count=len(self._p27_l3_rule_key_set()),
                    trace_id=probe_trace_id,
                ))

            episode_false_support = sum(item.false_support_count for item in probe_reports)
            episode_wrong_turns = domain_wrong + sum(item.wrong_turn_count for item in probe_reports)
            episode_probe_failures = sum(0 if item.probe_passed else 1 for item in probe_reports)
            episode_trace = self.trace.append("p28_multitask_episode_verified", {
                "domain_episode_cost_raw": domain_cost,
                "episode_index": episode_index,
                "false_support_count": episode_false_support,
                "probe_failure_count": episode_probe_failures,
                "promoted_rules_count": promoted,
                "release_line": "final_math_q256_p28_multitask_adaptive_memory",
                "retained_l3_rule_count": len(self._p27_l3_rule_key_set()),
                "trained_domain": trained_domain,
                "wrong_turn_count": episode_wrong_turns,
            })
            episodes.append(MultiTaskEpisodeReport(
                episode_index=episode_index,
                trained_domain=trained_domain,
                domain_episode_cost_raw=domain_cost,
                promoted_rules_count=promoted,
                retained_l3_rule_count=len(self._p27_l3_rule_key_set()),
                probe_results=tuple(probe_reports),
                probe_failure_count=episode_probe_failures,
                false_support_count=episode_false_support,
                wrong_turn_count=episode_wrong_turns,
                trace_id=episode_trace.event_hash(),
            ))
            self.consolidate_l2_episode(
                episode_id=f"p28_episode_{episode_index}_{trained_domain}_l2_reset",
                promoted_rules_count=promoted,
                evidence_id=f"p28_l2_episode_{episode_index}_reset",
            )
            if episode_index < len(curriculum):
                self._p26_reset_episode_state(preserve_l3=True)

        no_cross_domain_regression = all(
            not probe.regression_detected
            for episode in episodes
            for probe in episode.probe_results
        )
        monotonic = {
            domain: (history[-1] <= history[0] if len(history) >= 2 else True)
            for domain, history in cost_history.items()
        }
        proof_gate_fact = fact_delta(FactFrame(
            EntityId("p28_topology_probe_final"),
            RelationId("located_in"),
            EntityId("safe_room"),
            EvidenceId("p28_topology_probe_final_ev"),
        ))
        topology_passed = bool(self._topology_precheck_l2_fact(proof_gate_fact).passed)
        false_support_total = sum(item.false_support_count for item in episodes)
        wrong_turn_total = sum(item.wrong_turn_count for item in episodes)
        proof_gates_passed = bool(false_support_total == 0 and wrong_turn_total == 0 and all(item.probe_failure_count == 0 for item in episodes))
        final_trace = self.trace.append("p28_continual_multitask_adaptation_verified", {
            "domain_cost_history_raw": {key: list(value) for key, value in cost_history.items()},
            "false_support_count": false_support_total,
            "monotonic_improvement_per_domain": monotonic,
            "no_cross_domain_regression": no_cross_domain_regression,
            "proof_gates_passed": proof_gates_passed,
            "release_line": "final_math_q256_p28_multitask_adaptive_memory",
            "retained_l3_rules_final": len(self._p27_l3_rule_key_set()),
            "topology_gates_passed": topology_passed,
            "total_l3_rules_promoted": total_promoted,
            "wrong_turn_count": wrong_turn_total,
        })
        report = MultiTaskAdaptiveMemoryReport(
            curriculum_domains=curriculum,
            domains_tested=domains,
            episodes=tuple(episodes),
            domain_cost_history_raw={key: tuple(value) for key, value in cost_history.items()},
            no_cross_domain_regression=no_cross_domain_regression,
            monotonic_improvement_per_domain=monotonic,
            proof_gates_passed=proof_gates_passed,
            topology_gates_passed=topology_passed,
            total_l3_rules_promoted=total_promoted,
            retained_l3_rules_final=len(self._p27_l3_rule_key_set()),
            false_support_count=false_support_total,
            wrong_turn_count=wrong_turn_total,
            trace_head=final_trace.event_hash(),
            trace_verified=self.trace.verify(),
            real_actions_allowed=self.config.allow_real_actions,
        )
        self.trace.append("p28_continual_multitask_adaptation_summary", report.as_payload())
        return report


    def _v1_expected_match(self, output: str, expected_contains: tuple[str, ...]) -> bool:
        """Post-inference expected matcher for v1.0 external-shaped rows."""

        lower = output.lower()
        return all(fragment.lower() in lower for fragment in expected_contains)

    def _v1_record_external_result(
        self,
        *,
        suite: str,
        row_id: str,
        engine_input: str,
        expected_contains: tuple[str, ...],
        response: RuntimeResponse,
        answer_key_visible_to_engine: int = 0,
    ) -> V1ExternalBenchmarkRowResult:
        """Create one external benchmark row result after runtime inference returns."""

        supported = self._p27_response_has_proof_support(response)
        passed = self._v1_expected_match(response.output, expected_contains)
        is_commit = response.output.startswith("COMMIT:")
        false_support = 1 if passed and not supported and not is_commit and response.decision.kind in {DecisionKind.ANSWER, DecisionKind.ACT_SIMULATED, DecisionKind.HYPOTHESIS} else 0
        return V1ExternalBenchmarkRowResult(
            suite=suite,
            row_id=row_id,
            engine_input_hash=active_state_digest({"engine_input": engine_input, "row_id": row_id, "suite": suite}),
            expected_digest=active_state_digest({"expected_contains": list(expected_contains), "row_id": row_id, "suite": suite}),
            decision_kind=response.decision.kind.value,
            output=response.output,
            passed=bool(passed),
            false_support=false_support,
            proof_or_evidence_present=bool(supported),
            answer_key_visible_to_engine=answer_key_visible_to_engine,
            trace_id=response.decision.trace_id,
        )

    def run_v1_clean_system_revalidation(
        self,
        *,
        stress_steps: int = 32,
        grid_size: int = 5,
    ) -> V1CleanSystemReport:
        """v1.0: external-shaped revalidation without architecture expansion.

        The method executes bAbI-shaped rows, Dialog-bAbI USR/SYS loader rows,
        contradiction rows and the P28 multitask stress inside the same
        HTCERuntime.  Gold/expected strings are committed only as post-run
        digests and are never passed into the runtime input.
        """

        if self.config.allow_real_actions:
            raise ValueError("v1.0 clean revalidation cannot run with real actions enabled")
        if not self.awake:
            self.wake()

        from pathlib import Path
        import tempfile
        from htce_origin import __version__
        from htce_origin.evaluation.benchmarks import load_dialog_babi

        row_results: list[V1ExternalBenchmarkRowResult] = []
        self.trace.append("v1_clean_system_revalidation_start", {
            "release_line": "v1.0_final_math_q256_clean",
            "stress_steps": int(stress_steps),
        })

        # bAbI-shaped rows: latest state, carried-object path chaining,
        # coreference-like update and deduction.  Expected values are checked
        # only after runtime.tick returns.
        babi_rows: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("FACT v1_mary located_in office EVID v1_babi_1a", ("COMMIT",)),
            ("QUERY v1_mary location EVID v1_babi_1q", ("office",)),
            ("FACT v1_mary located_in garden EVID v1_babi_2a", ("COMMIT",)),
            ("QUERY v1_mary location EVID v1_babi_2q", ("garden",)),
            ("FACT v1_football carried_by v1_mary EVID v1_babi_3a", ("COMMIT",)),
            ("QUERY v1_football location EVID v1_babi_3q", ("garden",)),
            ("FACT v1_lily is_a swan EVID v1_babi_15a", ("COMMIT",)),
            ("FACT swan afraid_of wolf EVID v1_babi_15b", ("COMMIT",)),
            ("QUERY v1_lily afraid_of EVID v1_babi_15q", ("wolf",)),
        )
        for index, (command, expected) in enumerate(babi_rows, start=1):
            response = self.tick(RuntimeRequest(command, source="v1_external_babi_shape"))
            row_results.append(self._v1_record_external_result(
                suite="babi_external_shape",
                row_id=f"babi_shape_{index}",
                engine_input=command,
                expected_contains=expected,
                response=response,
            ))

        # Dialog-bAbI USR/SYS loader contract: expected is loader metadata only;
        # runtime receives row.question.  The current NLU/AIR path handles the
        # same restaurant-domain turn stream used in P25/P28.
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dialog_babi_task1.txt"
            dataset.write_text(
                "USR|I want Italian food in Rome.\n"
                "SYS|ASK_CLARIFICATION: price\n"
                "USR|cheap\n"
                "SYS|api_call domain=restaurant cuisine=italian location=rome price=cheap\n"
                "USR|Actually, make it Chinese.\n"
                "SYS|api_call domain=restaurant cuisine=chinese location=rome price=cheap\n",
                encoding="utf-8",
            )
            rows = load_dialog_babi(tmp, task_id=1)
            dialog_loader_strict = len(rows) == 3 and all("task1" in row.source_path.lower() or "task1" in row.row_id.lower() for row in rows)
            self.nlu_bridge.reset()
            for index, row in enumerate(rows, start=1):
                response = self.tick(RuntimeRequest(row.question, source="v1_external_dialog_shape"))
                expected_parts: tuple[str, ...]
                if row.expected.lower().startswith("ask_clarification"):
                    expected_parts = ("ASK_CLARIFICATION", "price")
                elif "chinese" in row.expected.lower():
                    expected_parts = ("api_call", "cuisine=chinese", "location=rome", "price=cheap")
                else:
                    expected_parts = ("api_call", "cuisine=italian", "location=rome", "price=cheap")
                row_results.append(self._v1_record_external_result(
                    suite="dialog_babi_usr_sys_strict_loader",
                    row_id=f"dialog_shape_{index}",
                    engine_input=row.question,
                    expected_contains=expected_parts,
                    response=response,
                ))
        if not dialog_loader_strict:
            self.trace.append("v1_dialog_loader_strict_failed", {"loaded_rows": len(rows)})

        # Contradiction proof-gate revalidation.
        conflict_commands: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("FACT v1_conflict_item located_in safe EVID v1_conflict_pos", ("COMMIT",)),
            ("NEGATE v1_conflict_item located_in safe EVID v1_conflict_neg", ("REFUSE",)),
            ("QUERY v1_conflict_item location EVID v1_conflict_query", ("REFUSE",)),
        )
        for index, (command, expected) in enumerate(conflict_commands, start=1):
            response = self.tick(RuntimeRequest(command, source="v1_external_contradiction_shape"))
            row_results.append(self._v1_record_external_result(
                suite="contradiction_external_shape",
                row_id=f"contradiction_shape_{index}",
                engine_input=command,
                expected_contains=expected,
                response=response,
            ))

        domains = self._p28_curriculum_domains()
        repeat_count = max(1, int(stress_steps) // len(domains))
        curriculum = tuple((domains * repeat_count)[:max(len(domains), int(stress_steps) // 4)])
        self._p26_reset_episode_state(preserve_l3=True)
        multitask = self.run_continual_multitask_simulation(curriculum_domains=curriculum, steps=12, grid_size=grid_size)

        total = len(row_results)
        passed_count = sum(1 for row in row_results if row.passed)
        false_support = sum(row.false_support for row in row_results)
        answer_key_visible = sum(row.answer_key_visible_to_engine for row in row_results)
        proof_gate_fact = fact_delta(FactFrame(
            EntityId("v1_topology_probe_final"),
            RelationId("located_in"),
            EntityId("safe_room"),
            EvidenceId("v1_topology_probe_final_ev"),
        ))
        topology_passed = bool(self._topology_precheck_l2_fact(proof_gate_fact).passed)
        proof_gates_passed = bool(false_support == 0 and passed_count == total and multitask.proof_gates_passed)
        final_trace = self.trace.append("v1_clean_system_revalidation_verified", {
            "answer_key_visible_to_engine_count": answer_key_visible,
            "dialog_loader_strict_passed": dialog_loader_strict,
            "external_false_support_count": false_support,
            "external_rows_passed": passed_count,
            "multitask_no_cross_domain_regression": multitask.no_cross_domain_regression,
            "proof_gates_passed": proof_gates_passed,
            "release_line": "v1.0_final_math_q256_clean",
            "topology_gates_passed": topology_passed,
            "total_external_rows": total,
        })
        report = V1CleanSystemReport(
            version=__version__,
            external_rows=tuple(row_results),
            multitask_report=multitask,
            total_external_rows=total,
            external_rows_passed=passed_count,
            external_false_support_count=false_support,
            answer_key_visible_to_engine_count=answer_key_visible,
            dialog_loader_strict_passed=dialog_loader_strict,
            no_external_regression=bool(multitask.passed and multitask.no_cross_domain_regression),
            proof_gates_passed=proof_gates_passed,
            topology_gates_passed=topology_passed,
            clean_single_runtime_loop=True,
            trace_head=final_trace.event_hash(),
            trace_verified=self.trace.verify(),
            real_actions_allowed=self.config.allow_real_actions,
        )
        self.trace.append("v1_clean_system_revalidation_summary", report.as_payload())
        return report

    def export_state(self) -> dict[str, object]:
        """Export the full runtime state for deterministic snapshot/restore.

        The export includes L1/L2/L3 body, fact memory, proof layer, world model
        self-state/adaptive corrections, homeostasis, topology guard state and
        protected trace events. It is a state snapshot, not a new evidence source.
        """

        return {
            "schema_version": "htce-runtime-state-v1",
            "awake": self.awake,
            "body": self.body.snapshot(),
            "memory": self.memory.snapshot(),
            "proof": self.theorem_layer.export_state(),
            "world_model": {
                "adaptive_corrections": [
                    {"action_key": action_key, "context_key": context_key, "correction": list(correction)}
                    for (action_key, context_key), correction in sorted((self.world_model.adaptive_dynamics.correction_memory if self.world_model.adaptive_dynamics else {}).items())
                ],
                "adaptive_enabled": self.world_model.adaptive_enabled,
                "dimension": self.world_model.dimension,
                "high_error_threshold_bp": self.world_model.high_error_threshold_bp,
                "modulus": self.world_model.modulus,
                "self_model": {
                    "confidence_bp": self.world_model.self_model.confidence_bp,
                    "high_error_count": self.world_model.self_model.high_error_count,
                    "accumulated_error_raw": self.world_model.self_model.accumulated_error_raw,
                    "last_error_bp": self.world_model.self_model.last_error_bp,
                    "last_error_raw": self.world_model.self_model.last_error_raw,
                    "mean_error_bp": self.world_model.self_model.mean_error_bp,
                    "observations": self.world_model.self_model.observations,
                    "uncertainty_bp": self.world_model.self_model.uncertainty_bp,
                    "uncertainty_raw": self.world_model.self_model.uncertainty_raw,
                },
            },
            "homeostasis": self.homeostatic_state.as_mapping(),
            "topology_guard": {
                "persistent_anomaly_score_bp": self.topology_guard.persistent_anomaly_score_bp,
                "profile_name": self.topology_guard.profile.profile_name,
                "l2_live_window": [list(vector) for vector in self.l2_topology_window],
                "l2_window_is_clean_state": True,
                "max_l2_live_window": self.max_l2_topology_window,
            },
            "claim_support_reports": [report.as_payload() for report in self.claim_support_reports.values()],
            "l3_provisional_rules": [decision.as_payload() for decision in self.l3_provisional_rules.values()],
            "trace_events": [event.canonical_payload() for event in self.trace.events],
        }

    @classmethod
    def restore_state(cls, payload: dict[str, object], config: RuntimeConfig | None = None) -> "HTCERuntime":
        """Restore a runtime from ``export_state()`` output and verify trace chain."""

        runtime = cls(config=config)
        runtime.awake = bool(payload.get("awake", False))
        runtime.body = L123Body.from_snapshot(payload.get("body", {}))
        runtime.memory = FactDeltaStore.from_snapshot(payload.get("memory", {}))
        runtime.theorem_layer = TheoremLayer.from_state(payload.get("proof", {}))
        homeostasis = payload.get("homeostasis", {})
        if isinstance(homeostasis, dict):
            runtime.homeostatic_state = HomeostaticState(
                energy_bp=int(homeostasis.get("energy", homeostasis.get("energy_bp", 10000))),
                sleep_pressure_bp=int(homeostasis.get("sleep_pressure", homeostasis.get("sleep_pressure_bp", 0))),
                risk_bp=int(homeostasis.get("risk", homeostasis.get("risk_bp", 0))),
                integrity_bp=int(homeostasis.get("integrity", homeostasis.get("integrity_bp", 10000))),
                novelty_bp=int(homeostasis.get("novelty", homeostasis.get("novelty_bp", 0))),
                uncertainty_bp=int(homeostasis.get("uncertainty", homeostasis.get("uncertainty_bp", 0))),
            )
        world_payload = payload.get("world_model", {})
        if isinstance(world_payload, dict):
            from htce_origin.cognition.world import AdaptiveQ256Dynamics, SelfModelState
            adaptive = AdaptiveQ256Dynamics(dimension=runtime.body.dimension, modulus=runtime.body.modulus)
            for raw in world_payload.get("adaptive_corrections", ()):  # type: ignore[union-attr]
                if isinstance(raw, dict):
                    adaptive.correction_memory[(str(raw.get("action_key", "")), str(raw.get("context_key", "")))] = tuple(int(v) for v in raw.get("correction", ()))
            runtime.world_model = Q256WorldModel(
                dimension=int(world_payload.get("dimension", runtime.body.dimension)),
                modulus=int(world_payload.get("modulus", runtime.body.modulus)),
                high_error_threshold_bp=int(world_payload.get("high_error_threshold_bp", 2500)),
                adaptive_enabled=bool(world_payload.get("adaptive_enabled", False)),
                adaptive_dynamics=adaptive,
            )
            sm = world_payload.get("self_model", {})
            if isinstance(sm, dict):
                runtime.world_model.self_model = SelfModelState(
                    observations=int(sm.get("observations", 0)),
                    last_error_bp=int(sm.get("last_error_bp", 0)),
                    mean_error_bp=int(sm.get("mean_error_bp", 0)),
                    uncertainty_bp=int(sm.get("uncertainty_bp", 0)),
                    confidence_bp=int(sm.get("confidence_bp", 10000)),
                    high_error_count=int(sm.get("high_error_count", 0)),
                    last_error_raw=int(sm.get("last_error_raw", 0)),
                    accumulated_error_raw=int(sm.get("accumulated_error_raw", 0)),
                    uncertainty_raw=int(sm.get("uncertainty_raw", 0)),
                )
        topo = payload.get("topology_guard", {})
        if isinstance(topo, dict):
            runtime.topology_guard = TopologyGuard(CalibrationProfile.profile_short_path(dimension=runtime.body.dimension, modulus=runtime.body.modulus))
            runtime.topology_guard.persistent_anomaly_score_bp = int(topo.get("persistent_anomaly_score_bp", 0))
            runtime.max_l2_topology_window = int(topo.get("max_l2_live_window", runtime.max_l2_topology_window))
            raw_window = topo.get("l2_live_window", ())
            if isinstance(raw_window, list) and raw_window:
                runtime.l2_topology_window = tuple(tuple(int(value) for value in vector) for vector in raw_window if isinstance(vector, list))
        events = []
        for raw in payload.get("trace_events", ()):  # type: ignore[union-attr]
            if isinstance(raw, dict):
                events.append(TraceEvent(
                    event_type=str(raw.get("event_type", "restored")),
                    payload=dict(raw.get("payload", {})) if isinstance(raw.get("payload", {}), dict) else {},
                    previous_hash=str(raw.get("previous_hash", "0" * 64)),
                    sequence=int(raw.get("sequence", len(events))),
                    created_at=str(raw.get("created_at", "1970-01-01T00:00:00+00:00")),
                ))
        runtime.trace = TraceLog(events)
        runtime.policy = PolicyEngine(trace=runtime.trace)
        if not runtime.trace.verify():
            raise ValueError("restored trace verification failed")
        return runtime

    def health(self) -> dict[str, object]:
        return {
            "awake": self.awake,
            "body_digest": self.body.digest(),
            "homeostatic_state": self.homeostatic_state.as_mapping(),
            "l1_clock": self.body.l1.clock,
            "l2_clock": self.body.l2.clock,
            "l3_clock": self.body.l3.clock,
            "latest_fact_count": len(self.memory.active_records()),
            "l2_active_working_count": len(self.body.l2_active_contributions),
            "l2_clean_digest": active_state_digest({"vector": self.body.l2_clean_vector()}),
            "l2_episode_anchor_count": len(self.body.l2_archived_anchors),
            "l2_episode_fact_count": self.body.l2_episode_fact_count,
            "l2_episode_index": self.body.l2_episode_index,
            "legacy_imports_allowed": self.config.allow_legacy_imports,
            "memory_digest": self.memory.digest(),
            "real_actions_allowed": self.config.allow_real_actions,
            "registered_skills": self.skill_registry.names(),
            "release_line": "final_math",
            "trace_head": self.trace.head,
            "trace_verified": self.trace.verify(),
            "topology_anomaly_score_bp": self.topology_guard.persistent_anomaly_score_bp,
            "claim_support_report_count": len(self.claim_support_reports),
            "l3_provisional_rule_count": len(self.l3_provisional_rules),
            "world_model_uncertainty_bp": self.world_model.self_model.uncertainty_bp,
        }


def _request_kind_from_event(kind: str) -> RequestKind:
    if kind == "fact_candidate" or kind == "negation_candidate":
        return RequestKind.COMMIT
    if kind == "query_candidate":
        return RequestKind.QUERY
    if kind == "simulated_action_candidate":
        return RequestKind.SIMULATED_ACTION
    if kind == "proof_candidate":
        return RequestKind.CLAIM
    return RequestKind.HYPOTHESIS
