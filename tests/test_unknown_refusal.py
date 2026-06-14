from htce_origin.governance.evidence import EvidenceRecord, HashChain, sha256_hex
from htce_origin.governance.policy import DecisionKind, PolicyEngine, PolicyRequest, RequestKind


def test_unknown_unsupported_answer_refuses_with_trace_id_and_reason():
    engine = PolicyEngine(trace=HashChain())
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.ANSWER,
            payload={"claim": "location(john)"},
            evidence_id=None,
            supported=False,
        )
    )
    assert decision.kind == DecisionKind.REFUSE
    assert decision.trace_id
    assert "missing evidence" in decision.reason
    assert engine.trace.verify()


def test_supported_answer_has_trace_id():
    record = EvidenceRecord("event42", "operator", sha256_hex("Mary located_in office"))
    engine = PolicyEngine(evidence_records={"event42": record})
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.ANSWER,
            payload={"answer": "office", "claim": "location(mary,office)"},
            evidence_id="event42",
            supported=True,
        )
    )
    assert decision.kind == DecisionKind.ANSWER
    assert decision.trace_id
    assert not decision.is_hypothesis
    assert engine.trace.verify()


def test_missing_evidence_blocks_commit_candidate():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={"subject": "Mary", "relation": "located_in", "object": "office"},
            evidence_id=None,
            supported=True,
        )
    )
    assert decision.kind == DecisionKind.REFUSE
    assert any(gate.code == "EVIDENCE_MISSING" for gate in decision.gates)
    assert decision.reason


def test_unsupported_claim_refused():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.CLAIM,
            payload={"claim": "robot_can_lift_100kg"},
            evidence_id="event99",
            supported=False,
        )
    )
    assert decision.kind == DecisionKind.REFUSE
    assert any(gate.code in {"CLAIM_UNSUPPORTED", "EVIDENCE_RECORD_MISSING"} for gate in decision.gates)
    assert "unsupported" in decision.reason or "no supported record" in decision.reason


def test_hypothesis_marked_as_hypothesis_not_fact():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.HYPOTHESIS,
            payload={"hypothesis": "Mary may be in the office"},
            is_hypothesis=True,
        )
    )
    assert decision.kind == DecisionKind.HYPOTHESIS
    assert decision.is_hypothesis is True
    assert decision.trace_id
    assert "hypothesis" in decision.reason


def test_real_action_blocked():
    record = EvidenceRecord("event_act", "operator", sha256_hex("move real actuator"))
    engine = PolicyEngine(evidence_records={"event_act": record})
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.REAL_ACTION,
            payload={"action": "move_arm", "target": "table"},
            evidence_id="event_act",
            supported=True,
            wants_real_action=True,
        )
    )
    assert decision.kind == DecisionKind.BLOCK_REAL_ACTION
    assert any(gate.code == "POLICY_REAL_ACTION_BLOCKED" for gate in decision.gates)
    assert decision.trace_id
    assert engine.trace.verify()


def test_simulated_action_can_be_candidate_but_not_real_action():
    record = EvidenceRecord("event_sim", "operator", sha256_hex("simulate move"))
    engine = PolicyEngine(evidence_records={"event_sim": record})
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.SIMULATED_ACTION,
            payload={"action": "move_demo"},
            evidence_id="event_sim",
            supported=True,
            wants_real_action=False,
        )
    )
    assert decision.kind == DecisionKind.ACT_SIMULATED
    assert decision.trace_id
    assert not decision.is_hypothesis



def test_cognitive_immune_low_confidence_fact_is_quarantined():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "subject": "Mary",
                "relation": "located_in",
                "object": "office",
                "confidence_bp": 6500,
            },
            evidence_id="event_low",
            supported=True,
        )
    )

    assert decision.kind == DecisionKind.REFUSE
    assert any(gate.code == "IMMUNE_LOW_CONFIDENCE" for gate in decision.gates)
    assert "low confidence" in decision.reason
    assert engine.trace.verify()


def test_cognitive_immune_supersession_allows_stronger_supported_fact():
    from htce_origin.governance.policy import FactCandidate

    engine = PolicyEngine()
    active = FactCandidate("Mary", "located_in", "office", evidence_id="event_old", confidence_bp=8000, revision=1)
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "subject": "Mary",
                "relation": "located_in",
                "object": "garden",
                "confidence_bp": 9000,
                "revision": 2,
            },
            evidence_id="event_new",
            supported=True,
        ),
        active_fact=active,
    )

    assert decision.kind == DecisionKind.ANSWER
    assert any(gate.code == "IMMUNE_SUPERSEDE" for gate in decision.gates)
    assert decision.trace_id
    assert engine.trace.verify()


def test_cognitive_immune_conflict_without_stronger_evidence_is_quarantined():
    from htce_origin.governance.policy import FactCandidate

    engine = PolicyEngine()
    active = FactCandidate("Mary", "located_in", "office", evidence_id="event_old", confidence_bp=9000, revision=1)
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "subject": "Mary",
                "relation": "located_in",
                "object": "garden",
                "confidence_bp": 8500,
            },
            evidence_id=None,
            supported=True,
        ),
        active_fact=active,
    )

    assert decision.kind == DecisionKind.REFUSE
    assert any(gate.code in {"EVIDENCE_MISSING", "IMMUNE_CONFLICT_WITHOUT_EVIDENCE"} for gate in decision.gates)
    assert engine.trace.verify()


def test_cognitive_immune_forbidden_relation_is_quarantined():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "subject": "agent",
                "relation": "bypass_claim_boundary",
                "object": "true",
                "confidence_bp": 10000,
            },
            evidence_id="event_forbidden",
            supported=True,
        )
    )

    assert decision.kind == DecisionKind.REFUSE
    assert any(gate.code == "IMMUNE_FORBIDDEN_RELATION" for gate in decision.gates)
    assert engine.trace.verify()


def test_cognitive_immune_blocks_unsafe_real_action_detection():
    engine = PolicyEngine()
    decision = engine.evaluate(
        PolicyRequest(
            kind=RequestKind.COMMIT,
            payload={
                "subject": "robot",
                "relation": "real_world_actuator",
                "object": "move_arm",
                "confidence_bp": 10000,
            },
            evidence_id="event_real",
            supported=True,
        )
    )

    assert decision.kind == DecisionKind.BLOCK_REAL_ACTION
    assert any(gate.code == "IMMUNE_UNSAFE_REAL_ACTION" for gate in decision.gates)
    assert engine.trace.verify()
