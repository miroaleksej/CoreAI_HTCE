from htce_origin.governance.proof import Judgment, RuleKind, Statement, TheoremLayer


def test_asserted_fact_proves_itself():
    layer = TheoremLayer()
    layer.add_judgment("A", evidence_id="eventA")
    proof = layer.prove("A")
    assert proof.valid
    assert not proof.quarantined
    assert proof.rules == (RuleKind.ASSERTED,)
    assert proof.conclusion.statement == Statement.atom("A")


def test_a_and_implication_proves_b():
    layer = TheoremLayer()
    layer.add_judgment("A", evidence_id="eventA")
    layer.add_implication("A", "B")
    proof = layer.prove("B")
    assert proof.valid
    assert proof.rules == (RuleKind.IMPLIES,)
    assert proof.premises[0].statement == Statement.atom("A")


def test_a_and_not_a_quarantines_a():
    layer = TheoremLayer()
    layer.add_judgment("A", evidence_id="eventA")
    layer.add_judgment("NOT A", evidence_id="eventNotA")
    proof = layer.prove("A")
    assert not proof.valid
    assert proof.quarantined
    assert RuleKind.CONTRADICTION in proof.rules
    assert proof.reason == "contradiction detected"


def test_association_alone_cannot_answer():
    layer = TheoremLayer()
    layer.add_association("located_in(Mary,office)")
    proof = layer.prove("located_in(Mary,office)")
    assert not proof.valid
    assert not proof.quarantined
    assert proof.rules == (RuleKind.ASSOCIATION,)
    assert "association alone" in proof.reason


def test_transitive_location_proof():
    layer = TheoremLayer()
    layer.add_judgment(Statement.atom("located_in", "Mary", "office"), evidence_id="event1")
    layer.add_judgment(Statement.atom("inside", "office", "building"), evidence_id="event2")
    proof = layer.prove(Statement.atom("located_in", "Mary", "building"))
    assert proof.valid
    assert proof.rules == (RuleKind.TRANSITIVE_LOCATION,)
    assert len(proof.premises) == 2


def test_class_inheritance_proof():
    layer = TheoremLayer()
    layer.add_judgment(Statement.atom("is_a", "sparrow", "bird"), evidence_id="event1")
    layer.add_judgment(Statement.atom("has_property", "bird", "has_wings"), evidence_id="event2")
    proof = layer.prove(Statement.atom("has_property", "sparrow", "has_wings"))
    assert proof.valid
    assert proof.rules == (RuleKind.CLASS_INHERITANCE,)


def test_skill_requires_ensures_proof():
    layer = TheoremLayer()
    layer.add_ensures("move_demo", Statement.atom("located_in", "robot", "table"))
    blocked = layer.verify_skill("move_demo")
    assert not blocked.valid
    assert RuleKind.ENSURES in blocked.rules
    assert "not proven" in blocked.reason

    layer.add_judgment(Statement.atom("located_in", "robot", "table"), evidence_id="event_move")
    accepted = layer.verify_skill("move_demo")
    assert accepted.valid
    assert accepted.conclusion.statement == Statement.atom("skill_verified", "move_demo")
    assert RuleKind.SKILL in accepted.rules
    assert RuleKind.ENSURES in accepted.rules


def test_skill_without_ensures_is_not_verified():
    layer = TheoremLayer()
    result = layer.verify_skill("unbounded_skill")
    assert not result.valid
    assert "no ENSURES" in result.reason


def test_proof_object_id_is_deterministic():
    layer = TheoremLayer()
    layer.add_judgment("A", evidence_id="eventA")
    first = layer.prove("A")
    second = layer.prove("A")
    assert first.proof_id == second.proof_id


def test_unsupported_judgment_does_not_prove():
    layer = TheoremLayer()
    layer.add_judgment(Judgment("A", evidence_id="eventA", supported=False))
    proof = layer.prove("A")
    assert not proof.valid
    assert proof.reason == "no proof found"


def test_where_query_proves_from_latest_state_index():
    layer = TheoremLayer()
    proof = layer.prove_where("Mary", {("mary", "located_in"): "office"})
    assert proof.valid
    assert proof.conclusion.statement == Statement.atom("located_in", "mary", "office")
    assert RuleKind.LATEST_STATE in proof.rules
    assert RuleKind.QUERY_STRATEGY in proof.rules


def test_where_query_without_latest_state_does_not_prove():
    layer = TheoremLayer()
    proof = layer.prove_where("Mary", {})
    assert not proof.valid
    assert RuleKind.LATEST_STATE in proof.rules
    assert "no latest-state" in proof.reason


def test_class_rule_deduction_for_afraid_of():
    layer = TheoremLayer()
    layer.add_judgment(Statement.atom("is_a", "tom", "cat"), evidence_id="e1")
    layer.add_judgment(Statement.atom("afraid_of", "cat", "water"), evidence_id="e2")
    proof = layer.prove(Statement.atom("afraid_of", "tom", "water"))
    assert proof.valid
    assert proof.rules == (RuleKind.CLASS_RULE_DEDUCTION,)
    assert len(proof.premises) == 2


def test_same_class_property_induction_is_hypothesis_not_answer():
    layer = TheoremLayer()
    layer.add_judgment(Statement.atom("is_a", "mittens", "cat"), evidence_id="e0")
    layer.add_judgment(Statement.atom("is_a", "cat_a", "cat"), evidence_id="e1")
    layer.add_judgment(Statement.atom("is_a", "cat_b", "cat"), evidence_id="e2")
    layer.add_judgment(Statement.atom("color_of", "cat_a", "black"), evidence_id="e3")
    layer.add_judgment(Statement.atom("color_of", "cat_b", "black"), evidence_id="e4")
    result = layer.infer_same_class_property_hypothesis(Statement.atom("color_of", "mittens", "black"))
    assert not result.proof.valid
    assert not result.answer_allowed
    assert result.hypothesis_allowed
    assert result.association_report is not None
    assert RuleKind.SAME_CLASS_PROPERTY_INDUCTION in result.proof.rules


def test_association_report_never_equals_proof():
    layer = TheoremLayer()
    layer.add_association(Statement.atom("located_in", "mary", "office"))
    result = layer.authorize_query(Statement.atom("located_in", "mary", "office"))
    assert not result.proof.valid
    assert not result.answer_allowed
    assert result.hypothesis_allowed
    assert result.association_report is not None
    assert result.association_report.has_candidates
    assert result.proof.proof_id != result.association_report.report_id


def test_valid_proof_authorizes_answer_not_hypothesis():
    layer = TheoremLayer()
    layer.add_judgment(Statement.atom("located_in", "mary", "office"), evidence_id="e1")
    result = layer.authorize_query(Statement.atom("located_in", "mary", "office"))
    assert result.proof.valid
    assert result.answer_allowed
    assert not result.hypothesis_allowed


def test_proof_path_scoring_prefers_valid_evidenced_short_proof():
    layer = TheoremLayer()
    layer.add_judgment("A", evidence_id="eA")
    proof = layer.prove("A")
    score = layer.score_proof_path(proof, falsifiability_bp=2000)

    assert score.answer_allowed is True
    assert score.proof_score_bp == 10000
    assert score.evidence_score_bp == 10000
    assert score.simplicity_bp > 0
    assert score.theory_score_bp == (4 * score.proof_score_bp + 4 * score.evidence_score_bp + score.simplicity_bp + 2000) // 10


def test_proof_path_scoring_does_not_authorize_association_only():
    layer = TheoremLayer()
    layer.add_association("located_in(Mary,office)")
    proof = layer.prove("located_in(Mary,office)")
    score = layer.score_proof_path(proof, evidence_score_bp=0, falsifiability_bp=5000)

    assert proof.valid is False
    assert score.answer_allowed is False
    assert score.proof_score_bp == 0
    assert score.theory_score_bp < 5000
