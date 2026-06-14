from pathlib import Path
import ast

from htce_origin.governance.evidence import HashChain
from htce_origin.control.homeostasis import HomeostaticState
from htce_origin.control.planner import DomainRandomizedSimulatorWorld, PlanStep, ProofGuidedPlanner, SimulatedSkill, SkillRegistry
from htce_origin.governance.policy import DecisionKind
from htce_origin.governance.proof import Statement, TheoremLayer
from htce_origin.body.runtime import HTCERuntime, RuntimeRequest

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def _verified_skill_context():
    registry = SkillRegistry()
    theorem = TheoremLayer()
    postcondition = Statement.atom("location", "robot", "table")
    skill = registry.register(SimulatedSkill(
        name="move_demo",
        ensures=postcondition,
        steps=(PlanStep("simulate_path"), PlanStep("simulate_arrival")),
    ))
    theorem.add_ensures(skill.name, postcondition)
    theorem.add_judgment(postcondition, evidence_id="proof_event_1", source="asserted", supported=True)
    return registry, theorem, skill


def test_simulation_planner_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "control" / "planner.py") == 0
    assert _float_constant_count(ROOT / "htce_origin" / "body" / "runtime.py") == 0


def test_verified_simulated_skill_can_run():
    registry, theorem, _skill = _verified_skill_context()
    result = ProofGuidedPlanner().plan_skill("move_demo", registry, theorem)

    assert result.decision == DecisionKind.ACT_SIMULATED
    assert result.allow_simulated_action is True
    assert tuple(step.action_name for step in result.steps) == ("simulate_path", "simulate_arrival")
    assert result.proof_id
    assert result.rollout_score_bp > 0


def test_unverified_skill_blocked():
    registry = SkillRegistry()
    theorem = TheoremLayer()
    postcondition = Statement.atom("location", "robot", "table")
    registry.register(SimulatedSkill("move_demo", postcondition, steps=(PlanStep("simulate_path"),)))
    theorem.add_ensures("move_demo", postcondition)

    result = ProofGuidedPlanner().plan_skill("move_demo", registry, theorem)

    assert result.decision == DecisionKind.REFUSE
    assert result.steps == ()
    assert not result.allow_simulated_action
    assert "not proven" in result.reason or "blocked" in result.reason


def test_real_action_always_blocked_unless_signed_manifest_mode_exists():
    registry = SkillRegistry()
    theorem = TheoremLayer()
    postcondition = Statement.atom("location", "robot", "dock")
    registry.register(SimulatedSkill("real_move", postcondition, simulated_only=False))
    theorem.add_ensures("real_move", postcondition)
    theorem.add_judgment(postcondition, source="asserted", supported=True)

    result = ProofGuidedPlanner().plan_skill("real_move", registry, theorem, allow_real_actions=True)

    assert result.decision == DecisionKind.BLOCK_REAL_ACTION
    assert result.steps == ()
    assert not result.allow_simulated_action
    assert "real action blocked" in result.reason
    assert "signed manifest" in result.reason


def test_rollout_scoring_penalizes_risk_and_longer_rollouts():
    planner = ProofGuidedPlanner()
    safe_short = planner.score_rollout((PlanStep("a"),), HomeostaticState())
    risky_long = planner.score_rollout(
        (PlanStep("a"), PlanStep("b"), PlanStep("c")),
        HomeostaticState(risk_bp=8000, uncertainty_bp=5000),
    )

    assert safe_short.utility_bp > risky_long.utility_bp
    assert risky_long.step_count == 3


def test_plan_produces_trace():
    registry, theorem, _skill = _verified_skill_context()
    trace = HashChain()
    result = ProofGuidedPlanner().plan_skill("move_demo", registry, theorem, trace=trace)

    assert result.trace_id
    assert trace.count == 1
    assert trace.verify()
    assert trace.snapshot().head == result.trace_id


def test_runtime_verified_simulated_skill_plan_produces_trace():
    runtime = HTCERuntime()
    postcondition = Statement.atom("location", "robot", "table")
    runtime.register_simulated_skill("move_demo", postcondition, steps=(PlanStep("simulate_arrival"),))
    runtime.assert_proof_fact(postcondition, evidence_id="proof_event_1")

    response = runtime.plan_simulated_skill("move_demo")

    assert response.decision.kind == DecisionKind.ACT_SIMULATED
    assert response.decision.trace_id
    assert response.diagnostics["step_count"] == 1
    assert response.diagnostics["rollout_score_bp"] > 0
    assert runtime.health()["trace_verified"] is True


def test_runtime_unverified_skill_is_refused_with_trace():
    runtime = HTCERuntime()
    runtime.register_simulated_skill("move_demo", Statement.atom("location", "robot", "table"))

    response = runtime.plan_simulated_skill("move_demo")

    assert response.decision.kind == DecisionKind.REFUSE
    assert response.decision.trace_id
    assert "not proven" in response.decision.reason or "blocked" in response.decision.reason
    assert runtime.trace.verify()


def test_runtime_real_action_skill_is_blocked_with_trace():
    runtime = HTCERuntime()
    postcondition = Statement.atom("location", "robot", "dock")
    runtime.register_real_action_skill_for_block_test("real_move", postcondition)
    runtime.assert_proof_fact(postcondition)

    response = runtime.plan_simulated_skill("real_move")

    assert response.decision.kind == DecisionKind.BLOCK_REAL_ACTION
    assert response.decision.trace_id
    assert "real action" in response.decision.reason
    assert runtime.trace.verify()



def test_domain_randomized_sample_domain_is_deterministic_and_seed_sensitive():
    simulator = DomainRandomizedSimulatorWorld()
    first = simulator.sample_domain("domain-a")
    repeat = simulator.sample_domain("domain-a")
    other = simulator.sample_domain("domain-b")

    assert first == repeat
    assert first.domain_hash != other.domain_hash
    assert 0 <= first.observation_noise_bp <= 1500
    assert 0 <= first.terrain_resistance_bp <= 1500


def test_domain_randomized_rollout_returns_worst_case_score():
    simulator = DomainRandomizedSimulatorWorld()
    steps = (PlanStep("simulate_path"), PlanStep("simulate_arrival"))
    seeds = ("nominal", "noisy", "resistant")

    robust = simulator.robust_rollout(steps, seeds, initial_state=HomeostaticState())
    rollout_scores = tuple(row.score_bp for row in robust.rollouts)

    assert robust.domain_count == len(seeds)
    assert robust.utility_bp == min(rollout_scores)
    assert robust.worst_domain_seed in seeds
    assert robust.step_count == 2


def test_planner_can_score_domain_randomized_plan_without_authorizing_real_action():
    planner = ProofGuidedPlanner()
    steps = (PlanStep("simulate_path"), PlanStep("simulate_arrival"))
    robust = planner.score_domain_randomized_rollout(
        steps,
        HomeostaticState(),
        seeds=("nominal", "noisy", "resistant"),
    )

    assert robust.domain_count == 3
    assert robust.utility_bp >= 0
    assert all(row.step_count == 2 for row in robust.rollouts)


def test_verified_skill_can_use_domain_randomized_robust_score_and_trace():
    registry, theorem, _skill = _verified_skill_context()
    trace = HashChain()
    result = ProofGuidedPlanner().plan_skill(
        "move_demo",
        registry,
        theorem,
        trace=trace,
        domain_seeds=("nominal", "noisy", "resistant"),
    )

    assert result.decision == DecisionKind.ACT_SIMULATED
    assert result.domain_randomized is True
    assert result.domain_count == 3
    assert result.worst_domain_seed in ("nominal", "noisy", "resistant")
    assert result.trace_id
    assert trace.verify()


def test_runtime_domain_randomized_skill_plan_reports_diagnostics():
    runtime = HTCERuntime()
    postcondition = Statement.atom("location", "robot", "table")
    runtime.register_simulated_skill("move_demo", postcondition, steps=(PlanStep("simulate_arrival"),))
    runtime.assert_proof_fact(postcondition, evidence_id="proof_event_1")

    response = runtime.plan_simulated_skill("move_demo", domain_seeds=("nominal", "noisy", "resistant"))

    assert response.decision.kind == DecisionKind.ACT_SIMULATED
    assert response.diagnostics["domain_randomized"] is True
    assert response.diagnostics["domain_count"] == 3
    assert response.diagnostics["worst_domain_seed"] in ("nominal", "noisy", "resistant")
    assert runtime.trace.verify()


def test_simulation_habitat_policy_allows_verified_simulation_only_action():
    from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy

    policy = SimulationHabitatPolicy()
    gate = HabitatGateInput(
        proof_bp=9000,
        topology_bp=9000,
        model_error_bp=1000,
        policy_ok=True,
        trace_ok=True,
        action_class="simulated",
        external_sensor_only=True,
    )
    decision = policy.evaluate(gate)

    assert decision.allowed is True
    assert decision.decision == DecisionKind.ACT_SIMULATED
    assert policy.allowed_real_action(gate) is False


def test_simulation_habitat_policy_blocks_low_confidence_gates():
    from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy

    policy = SimulationHabitatPolicy()

    assert not policy.evaluate(HabitatGateInput(proof_bp=6000)).allowed
    assert "proof" in policy.evaluate(HabitatGateInput(proof_bp=6000)).reason
    assert not policy.evaluate(HabitatGateInput(topology_bp=6000)).allowed
    assert "topology" in policy.evaluate(HabitatGateInput(topology_bp=6000)).reason
    assert not policy.evaluate(HabitatGateInput(model_error_bp=4000)).allowed
    assert "model error" in policy.evaluate(HabitatGateInput(model_error_bp=4000)).reason
    assert not policy.evaluate(HabitatGateInput(policy_ok=False)).allowed
    assert not policy.evaluate(HabitatGateInput(trace_ok=False)).allowed


def test_simulation_habitat_policy_blocks_real_action_even_with_signed_manifest_predicates():
    from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy

    policy = SimulationHabitatPolicy()
    gate = HabitatGateInput(
        action_class="real",
        signed_manifest=True,
        operator_enable=True,
        safety_case=True,
        dry_run_pass=True,
    )
    decision = policy.evaluate(gate)

    assert decision.allowed is False
    assert decision.allowed_real_action is False
    assert decision.decision == DecisionKind.BLOCK_REAL_ACTION
    assert "AllowedRealAction=0" in decision.reason
    assert policy.allowed_real_action(gate) is False


def test_planner_blocks_verified_skill_when_habitat_gate_fails_model_error():
    from htce_origin.control.planner import HabitatGateInput

    registry, theorem, _skill = _verified_skill_context()
    trace = HashChain()
    result = ProofGuidedPlanner().plan_skill(
        "move_demo",
        registry,
        theorem,
        trace=trace,
        habitat_gate_input=HabitatGateInput(model_error_bp=4000),
    )

    assert result.decision == DecisionKind.REFUSE
    assert not result.allow_simulated_action
    assert result.habitat_policy_allowed is False
    assert "model error" in result.habitat_gate_reason
    assert result.trace_id
    assert trace.verify()


def test_planner_blocks_verified_skill_when_external_sensor_boundary_violated():
    from htce_origin.control.planner import HabitatGateInput

    registry, theorem, _skill = _verified_skill_context()
    result = ProofGuidedPlanner().plan_skill(
        "move_demo",
        registry,
        theorem,
        habitat_gate_input=HabitatGateInput(external_sensor_only=False),
    )

    assert result.decision == DecisionKind.REFUSE
    assert not result.allow_simulated_action
    assert "external sensor-only" in result.reason


def test_runtime_reports_habitat_policy_diagnostics_for_simulated_plan():
    runtime = HTCERuntime()
    postcondition = Statement.atom("location", "robot", "table")
    runtime.register_simulated_skill("move_demo", postcondition, steps=(PlanStep("simulate_arrival"),))
    runtime.assert_proof_fact(postcondition, evidence_id="proof_event_1")

    response = runtime.plan_simulated_skill("move_demo")

    assert response.decision.kind == DecisionKind.ACT_SIMULATED
    assert response.diagnostics["habitat_policy_allowed"] is True
    assert "simulated action only" in str(response.diagnostics["habitat_gate_reason"])


def test_runtime_commit_passes_active_fact_to_immune_gate_and_quarantines_weaker_same_key():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    before = runtime.health()

    response = runtime._commit_fact(  # intentional white-box runtime integration check
        {"subject": "Mary", "relation": "located_in", "object": "garden", "confidence_bp": 9000},
        "event43",
        RuntimeRequest("FACT Mary located_in garden EVID event43"),
    )
    after = runtime.health()
    latest = runtime.memory.query("Mary", "located_in")

    assert response.decision.kind == DecisionKind.REFUSE
    assert any(gate["code"] == "IMMUNE_CONFLICT_WITHOUT_EVIDENCE" for gate in response.diagnostics["policy_decision"]["gates"])
    assert latest.answer == "office"
    assert after["latest_fact_count"] == before["latest_fact_count"]
    assert after["l2_clock"] == before["l2_clock"]


def test_runtime_commit_passes_active_fact_to_immune_gate_and_supersedes_stronger_same_key():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))

    response = runtime._commit_fact(  # intentional white-box runtime integration check
        {"subject": "Mary", "relation": "located_in", "object": "garden", "confidence_bp": 10000},
        "event43",
        RuntimeRequest("FACT Mary located_in garden EVID event43"),
    )
    latest = runtime.memory.query("Mary", "located_in")
    history = runtime.memory.history("Mary", "located_in")

    assert response.decision.kind == DecisionKind.ANSWER
    assert any(gate["code"] == "IMMUNE_SUPERSEDE" for gate in response.diagnostics["policy_decision"]["gates"])
    assert latest.answer == "garden"
    assert any(record.status.value == "superseded" for record in history)


def test_runtime_low_confidence_fact_is_refused_before_memory_commit():
    runtime = HTCERuntime()
    before = runtime.health()

    response = runtime._commit_fact(
        {"subject": "Mary", "relation": "located_in", "object": "office", "confidence_bp": 6500},
        "event_low",
        RuntimeRequest("FACT Mary located_in office EVID event_low"),
    )
    after = runtime.health()

    assert response.decision.kind == DecisionKind.REFUSE
    assert any(gate["code"] == "IMMUNE_LOW_CONFIDENCE" for gate in response.diagnostics["policy_decision"]["gates"])
    assert after["latest_fact_count"] == before["latest_fact_count"]
    assert after["l2_clock"] == before["l2_clock"]


def test_runtime_unsafe_action_relation_is_blocked_by_immune_gate():
    runtime = HTCERuntime()

    response = runtime._commit_fact(
        {"subject": "robot", "relation": "real_world_actuator", "object": "move_arm"},
        "event_real",
        RuntimeRequest("FACT robot real_world_actuator move_arm EVID event_real"),
    )

    assert response.decision.kind == DecisionKind.BLOCK_REAL_ACTION
    assert any(gate["code"] == "IMMUNE_UNSAFE_REAL_ACTION" for gate in response.diagnostics["policy_decision"]["gates"])
    assert runtime.health()["latest_fact_count"] == 0


def test_runtime_query_uses_latest_state_proof_bridge_and_records_proof_path():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))

    response = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask42"))
    trace_payload = runtime.trace.events[-1].payload

    assert response.output == "ANSWER: office"
    assert response.diagnostics["authorization"]["answer_allowed"] is True
    assert response.diagnostics["proof_id"]
    assert "LATEST_STATE" in response.diagnostics["proof_path"]
    assert "QUERY_STRATEGY" in response.diagnostics["proof_path"]
    assert trace_payload["proof_id"] == response.diagnostics["proof_id"]
    assert trace_payload["proof_path"] == response.diagnostics["proof_path"]


def test_runtime_association_only_candidate_cannot_authorize_answer():
    runtime = HTCERuntime()
    runtime.theorem_layer.add_association(Statement.atom("located_in", "mary", "office"))

    response = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask42"))

    assert response.decision.kind in {DecisionKind.ASK_CLARIFICATION, DecisionKind.REFUSE}
    assert response.diagnostics["query"]["status"] == "unknown"
    assert runtime.trace.verify()


def test_runtime_observe_simulated_closes_l1_world_curiosity_homeostasis_loop_without_l2_l3_commit():
    from htce_origin.control.homeostasis import SensoryObservation

    runtime = HTCERuntime()
    before = runtime.health()
    observation = SensoryObservation(
        modality="vision_sim",
        value="moving_edge_cluster",
        intensity_bp=8200,
        reliability_bp=9000,
        phase=tuple((index + 1) * 31 for index in range(runtime.body.dimension)),
        evidence_id="sim_obs_p1",
    )

    response = runtime.observe_simulated(observation)
    after = runtime.health()
    trace_payload = runtime.trace.events[-1].payload

    assert response.decision.kind == DecisionKind.HYPOTHESIS
    assert response.diagnostics["l1_changed"] is True
    assert response.diagnostics["l2_fact_count_after"] == response.diagnostics["l2_fact_count_before"]
    assert response.diagnostics["l2_clock_after"] == response.diagnostics["l2_clock_before"]
    assert response.diagnostics["l3_clock_after"] == response.diagnostics["l3_clock_before"]
    assert response.diagnostics["prediction_error_bp"] == trace_payload["prediction_error_bp"]
    assert response.diagnostics["curiosity_bp"] == trace_payload["curiosity_bp"]
    assert response.diagnostics["real_sensor_commit_allowed"] is False
    assert after["l1_clock"] == before["l1_clock"] + 1
    assert after["latest_fact_count"] == before["latest_fact_count"]
    assert runtime.trace.verify()


def test_robust_multi_plan_selection_compares_candidates_and_uses_worst_case_score():
    from htce_origin.control.planner import CandidatePlan

    planner = ProofGuidedPlanner()
    seeds = ("nominal", "noisy", "resistant", "terrain-hard")
    short_low_goal = CandidatePlan(
        "short_low_goal",
        (PlanStep("wait"),),
        goal_progress_bp=5000,
    )
    robust_two_step = CandidatePlan(
        "robust_two_step",
        (PlanStep("stabilize"), PlanStep("advance")),
        goal_progress_bp=10000,
        stability_bonus_bp=1000,
    )
    fragile_short = CandidatePlan(
        "fragile_short",
        (PlanStep("quick"),),
        goal_progress_bp=10000,
        domain_fragility_bp=10000,
    )

    selection = planner.choose_robust_plan(
        (short_low_goal, robust_two_step, fragile_short),
        HomeostaticState(),
        seeds=seeds,
    )
    robust_scores = {score.plan.name: score.robust_score.utility_bp for score in selection.candidate_scores}

    assert selection.compared_plan_count == 3
    assert selection.selected_plan.name == max(robust_scores, key=robust_scores.get)
    assert selection.selected_score.robust_score.utility_bp == max(robust_scores.values())
    assert selection.selected_plan.name != selection.shortest_plan_name
    assert len(selection.selected_plan.steps) == 2


def test_runtime_topology_gate_blocks_l2_commit_before_memory_mutation():
    from htce_origin.topology.guard import CalibrationProfile, TopologyGuard

    runtime = HTCERuntime()
    runtime.topology_guard = TopologyGuard(CalibrationProfile.profile_phase_shock(dimension=runtime.body.dimension, modulus=runtime.body.modulus))
    before = runtime.health()
    response = runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    after = runtime.health()

    assert response.decision.kind == DecisionKind.REFUSE
    assert "topology" in response.output.lower()
    assert after["latest_fact_count"] == before["latest_fact_count"]
    assert after["l2_clock"] == before["l2_clock"]
    assert response.diagnostics["topology_decision"]["passed"] is False
    assert runtime.trace.verify()


def test_runtime_evidence_report_controls_fact_confidence_and_support():
    from htce_origin.governance.evidence import EvidenceAnchor, EvidenceRelation, EvidenceWeigher, SourceManifest, sha256_hex as evidence_sha256_hex

    weak = SourceManifest(
        source_id="weak-blog",
        uri="https://example.org/blog",
        title="Weak blog",
        source_type="blog",
        base_quality_bp=4200,
        weak_source=1,
    )
    strong = SourceManifest(
        source_id="primary",
        uri="https://example.org/primary",
        title="Primary source",
        source_type="primary_paper",
        base_quality_bp=7200,
        primary_source=1,
        independent_replication_count=1,
    )
    weigher = EvidenceWeigher(support_threshold_bp=6000, contradiction_threshold_bp=4000)

    weak_runtime = HTCERuntime()
    weak_anchor = EvidenceAnchor("weak", "located_in(mary,office)", weak, EvidenceRelation.SUPPORT, evidence_sha256_hex("weak"))
    weak_runtime.register_claim_support_report(weigher.score_claim("located_in(mary,office)", (weak_anchor,)))
    weak_response = weak_runtime.tick(RuntimeRequest("FACT Mary located_in office EVID weak1"))
    assert weak_response.decision.kind == DecisionKind.REFUSE
    assert weak_runtime.health()["latest_fact_count"] == 0

    strong_runtime = HTCERuntime()
    strong_anchor = EvidenceAnchor("strong", "located_in(mary,office)", strong, EvidenceRelation.SUPPORT, evidence_sha256_hex("strong"))
    strong_runtime.register_claim_support_report(weigher.score_claim("located_in(mary,office)", (strong_anchor,)))
    strong_response = strong_runtime.tick(RuntimeRequest("FACT Mary located_in office EVID strong1"))
    assert strong_response.decision.kind == DecisionKind.ANSWER
    assert strong_response.diagnostics["record"]["object"] == "office"
    assert strong_response.diagnostics["claim_support_report"]["net_support_bp"] >= 6000


def test_runtime_full_export_restore_round_trip_preserves_body_memory_world_homeostasis_trace():
    runtime = HTCERuntime()
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    runtime.observe_simulated({
        "modality": "text-sim",
        "value": "Mary visible near office",
        "intensity_bp": 7000,
        "reliability_bp": 8000,
        "phase": tuple(17 for _ in range(runtime.body.dimension)),
        "evidence_id": "sim_obs_1",
    })
    before_health = runtime.health()
    exported = runtime.export_state()
    restored = HTCERuntime.restore_state(exported)
    after_health = restored.health()
    query = restored.tick(RuntimeRequest("QUERY Mary location EVID ask42"))

    assert before_health["body_digest"] == after_health["body_digest"]
    assert before_health["memory_digest"] == after_health["memory_digest"]
    assert before_health["trace_head"] == after_health["trace_head"]
    assert before_health["trace_verified"] is True
    assert after_health["trace_verified"] is True
    assert query.output == "ANSWER: office"
    assert query.diagnostics["proof_path"] == ["LATEST_STATE", "QUERY_STRATEGY"]


def test_runtime_located_query_uses_associative_toroidal_read_path():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))

    response = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask42"))

    assert response.output == "ANSWER: office"
    assert "associative_toroidal_read" in response.diagnostics["query"]["reason"]
    assert response.diagnostics["query"]["evidence_id"] == "event42"
    assert response.diagnostics["authorization"]["answer_allowed"] is True


def test_runtime_topology_precheck_uses_weighted_commit_and_live_betti_diagnostics():
    runtime = HTCERuntime()
    response = runtime._commit_fact(
        {"subject": "Mary", "relation": "located_in", "object": "office", "confidence_bp": 9000},
        "event42",
        RuntimeRequest("FACT Mary located_in office EVID event42"),
    )
    topology = response.diagnostics["topology_decision"]

    assert topology["details"]["weighted_commit_precheck"] is True
    assert "live_beta0" in topology["details"]
    assert "live_beta1" in topology["details"]
    assert topology["details"]["live_window_count"] >= 2
    assert runtime.export_state()["topology_guard"]["l2_live_window"]
