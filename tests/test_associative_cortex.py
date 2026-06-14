from htce_origin.cognition.cortex import AssociativeCortex, CandidateStatus, EdgeKind
from htce_origin.governance.proof import Statement, TheoremLayer


def test_similar_memory_retrieved():
    cortex = AssociativeCortex()
    target = cortex.remember("located_in(Mary,office)", evidence_id="event42", tags=("location",))
    cortex.remember("located_in(John,garden)", evidence_id="event43", tags=("location",))

    candidates = cortex.retrieve("located_in(Mary,?)", top_k=2)

    assert candidates
    assert candidates[0].node_id == target.node_id
    assert candidates[0].label == "located_in(Mary,office)"
    assert candidates[0].status == CandidateStatus.HYPOTHESIS
    assert candidates[0].answer_authorized is False


def test_temporal_and_causal_graph_links_are_preserved():
    cortex = AssociativeCortex()
    a = cortex.remember("located_in(Mary,hall)", evidence_id="event1")
    b = cortex.remember("located_in(Mary,office)", evidence_id="event2")
    c = cortex.remember("has_property(office,workplace)", evidence_id="event3")

    temporal = cortex.link_temporal(a, b, evidence_id="trace_t")
    causal = cortex.link_causal(c, b, weight_bp=7000, evidence_id="trace_c")
    candidates = cortex.retrieve("located_in(Mary,office)", top_k=1)

    assert temporal.kind == EdgeKind.TEMPORAL
    assert causal.kind == EdgeKind.CAUSAL
    assert any("temporal:in" in item for item in candidates[0].graph_context)
    assert any("causal:in" in item for item in candidates[0].graph_context)


def test_candidate_without_proof_is_hypothesis_not_fact():
    cortex = AssociativeCortex()
    cortex.remember("located_in(Mary,office)", evidence_id="event42")

    candidate = cortex.retrieve("located_in(Mary,office)", top_k=1)[0]

    assert candidate.status == CandidateStatus.HYPOTHESIS
    assert candidate.answer_authorized is False
    assert "hypothesis" in candidate.reason


def test_association_alone_cannot_answer_even_with_matching_query():
    cortex = AssociativeCortex()
    cortex.remember_association("located_in(Mary,office)")
    proof = TheoremLayer()
    proof.add_association("located_in(Mary,office)")

    candidate = cortex.retrieve("located_in(Mary,office)", proof_layer=proof, top_k=1)[0]

    assert candidate.status == CandidateStatus.HYPOTHESIS
    assert candidate.answer_authorized is False
    assert candidate.proof_valid is False
    assert "association" in candidate.reason


def test_proven_candidate_can_be_marked_authorized_by_proof_layer():
    cortex = AssociativeCortex()
    cortex.remember("located_in(Mary,office)", evidence_id="event42")
    proof = TheoremLayer()
    proof.add_judgment("located_in(Mary,office)", evidence_id="event42")

    candidate = cortex.retrieve("located_in(Mary,office)", proof_layer=proof, top_k=1)[0]

    assert candidate.status == CandidateStatus.PROVEN
    assert candidate.proof_valid is True
    assert candidate.answer_authorized is True
    assert candidate.proof_id


def test_contradicted_candidate_is_blocked_and_downranked():
    cortex = AssociativeCortex()
    cortex.remember("located_in(Mary,office)", evidence_id="event42")
    proof = TheoremLayer()
    proof.add_judgment("located_in(Mary,office)", evidence_id="event42")
    proof.add_judgment("NOT located_in(Mary,office)", evidence_id="correction43")

    candidate = cortex.retrieve("located_in(Mary,office)", proof_layer=proof, top_k=1)[0]

    assert candidate.status == CandidateStatus.BLOCKED
    assert candidate.blocked is True
    assert candidate.answer_authorized is False
    assert candidate.score_bp == 0
    assert "contradiction" in candidate.reason


def test_contradicted_candidate_does_not_outrank_safe_candidate():
    cortex = AssociativeCortex()
    contradicted = cortex.remember("located_in(Mary,office)", evidence_id="event42")
    safe = cortex.remember("located_in(Mary,garden)", evidence_id="event44")
    proof = TheoremLayer()
    proof.add_judgment("located_in(Mary,office)", evidence_id="event42")
    proof.add_judgment("NOT located_in(Mary,office)", evidence_id="correction43")
    proof.add_judgment("located_in(Mary,garden)", evidence_id="event44")

    candidates = cortex.retrieve("located_in(Mary,office)", proof_layer=proof, top_k=2)

    assert candidates[0].node_id == safe.node_id
    assert candidates[0].status == CandidateStatus.PROVEN
    assert any(c.node_id == contradicted.node_id and c.blocked for c in candidates)


def test_weighted_graph_scoring_exposes_phase_temporal_causal_components():
    cortex = AssociativeCortex()
    cue = cortex.remember("same_goal(Mary,robot)", evidence_id="cue1")
    target = cortex.remember("helps(robot,Mary)", evidence_id="support1")
    distractor = cortex.remember("helps(robot,box)", evidence_id="support2")

    cortex.link_temporal(cue, target, weight_bp=9000, evidence_id="trace_t", temporal_delta=1)
    cortex.link_causal(cue, target, weight_bp=8500, evidence_id="trace_c")

    candidates = cortex.retrieve("helps(robot,Mary)", top_k=3)

    target_candidate = next(c for c in candidates if c.node_id == target.node_id)
    distractor_candidate = next(c for c in candidates if c.node_id == distractor.node_id)

    assert target_candidate.status == CandidateStatus.HYPOTHESIS
    assert target_candidate.answer_authorized is False
    assert target_candidate.phase_similarity_bp == 10000
    assert target_candidate.temporal_relevance_bp > 0
    assert target_candidate.causal_support_bp > 0
    assert target_candidate.score_bp > distractor_candidate.score_bp


def test_graph_contradiction_penalty_downranks_without_turning_into_fact():
    cortex = AssociativeCortex()
    safe = cortex.remember("located_in(Mary,garden)", evidence_id="event44")
    risky = cortex.remember("located_in(Mary,office)", evidence_id="event42")
    contradiction = cortex.remember("NOT located_in(Mary,office)", evidence_id="correction43")

    cortex.link_contradiction(contradiction, risky, weight_bp=10000, evidence_id="conflict1")
    candidates = cortex.retrieve("located_in(Mary,office)", top_k=2)

    risky_candidate = next(c for c in candidates if c.node_id == risky.node_id)
    assert risky_candidate.contradiction_penalty_bp == 10000
    assert risky_candidate.status == CandidateStatus.HYPOTHESIS
    assert risky_candidate.answer_authorized is False
    assert "contradiction penalty" in risky_candidate.reason
    assert any(c.node_id == safe.node_id for c in candidates)


def test_analogy_link_supports_hypothesis_grade_transfer_only():
    cortex = AssociativeCortex()
    known = cortex.remember("uses_tool(engineer,wrench)", evidence_id="event7")
    analogy = cortex.remember_association("uses_tool(robot,gripper)")

    edge = cortex.link_analogy(known, analogy, weight_bp=8000, evidence_id="analogy1")
    candidates = cortex.retrieve("uses_tool(robot,gripper)", top_k=2)
    candidate = next(c for c in candidates if c.node_id == analogy.node_id)

    assert edge.kind == EdgeKind.ANALOGY
    assert candidate.causal_support_bp > 0
    assert candidate.phase_similarity_bp == 10000
    assert candidate.status == CandidateStatus.HYPOTHESIS
    assert candidate.answer_authorized is False
    assert "association" in candidate.reason


def test_association_scoring_weights_are_integer_and_reject_bad_values():
    from htce_origin.cognition.cortex import AssociationScoringWeights, CortexError

    weights = AssociationScoringWeights(phase_weight=40, temporal_weight=20, causal_weight=25, contradiction_weight=15)
    assert weights.denominator == 100

    try:
        AssociationScoringWeights(phase_weight=-1)
    except CortexError:
        pass
    else:
        raise AssertionError("negative association score weight must be rejected")


def test_cortex_runtime_float_count_is_zero():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "htce_origin" / "cognition" / "cortex.py"
    tree = ast.parse(path.read_text())
    count = sum(isinstance(node.value, float) for node in ast.walk(tree) if isinstance(node, ast.Constant))
    assert count == 0


def test_candidate_theory_scores_but_remains_hypothesis_only():
    from htce_origin.cognition.cortex import CandidateTheory

    cortex = AssociativeCortex()
    cortex.remember("located_in(Mary,office)", evidence_id="e1")
    cortex.remember_association("likes(Mary,quiet_room)")
    candidates = cortex.retrieve("located_in(Mary,office)", top_k=2)

    theory = cortex.build_candidate_theory(candidates, proof_score_bp=2500, evidence_score_bp=5000, falsifiability_bp=3000)
    report = cortex.evaluate_candidate_theories((theory,))

    assert isinstance(theory, CandidateTheory)
    assert theory.status == CandidateStatus.HYPOTHESIS
    assert theory.answer_authorized is False
    assert theory.score_bp == (4 * 2500 + 4 * 5000 + theory.simplicity_bp + 3000) // 10
    assert report.best_theory_id == theory.theory_id
    assert report.answer_authorized is False


def test_counterfactual_query_returns_hypothesis_candidates_without_commit():
    from htce_origin.cognition.cortex import CounterfactualQuery

    cortex = AssociativeCortex()
    node = cortex.remember_association("located_in(robot,charging_station)")
    query = CounterfactualQuery(
        given=Statement.atom("battery_low", "robot"),
        ask=Statement.atom("located_in", "robot", "charging_station"),
    )

    candidates = cortex.counterfactual_hypotheses(query, top_k=1)

    assert candidates
    assert candidates[0].node_id == node.node_id
    assert candidates[0].status == CandidateStatus.HYPOTHESIS
    assert candidates[0].answer_authorized is False
    assert query.query_id.startswith("cf_")


def test_analogy_transfer_result_is_hypothesis_grade_only():
    from htce_origin.cognition.cortex import AnalogyTransferResult

    cortex = AssociativeCortex()
    source = cortex.remember("uses_tool(engineer,wrench)", evidence_id="e1")

    result = cortex.transfer_by_analogy(source, "uses_tool(robot,gripper)", weight_bp=8000, evidence_id="analogy1")

    assert isinstance(result, AnalogyTransferResult)
    assert result.source_node_id == source.node_id
    assert result.candidate.status == CandidateStatus.HYPOTHESIS
    assert result.answer_authorized is False
    assert result.transfer_score_bp > 0
    assert "hypothesis" in result.reason


def test_cortex_sequence_counter_wraps_modulo_without_float_or_unbounded_growth():
    from htce_origin.cognition.cortex import MAX_SEQUENCE_COUNTER

    cortex = AssociativeCortex()
    cortex._sequence_counter = MAX_SEQUENCE_COUNTER - 1
    wrapped = cortex.remember("located_in(robot,dock)", evidence_id="wrap1")
    next_node = cortex.remember("located_in(robot,lab)", evidence_id="wrap2")

    assert wrapped.timestamp == 0
    assert next_node.timestamp == 1
    assert cortex._sequence_counter == 1
