from htce_origin.body.runtime import HTCERuntime, RuntimeConfig, RuntimeRequest
from htce_origin.cognition.l3_promotion import (
    L3RuleCandidate,
    L3RulePromotionStatus,
)
from htce_origin.governance.proof import Statement


def test_p16_l3_candidate_promotes_only_provisional_rule_without_authority():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    before = runtime.health()
    candidate = L3RuleCandidate(
        statement=Statement.atom("has_property", "cat", "fur"),
        support_count_raw=2,
        trace_ids=("trace_a", "trace_b"),
        source_rule_id="sleep_rule_cat_fur",
        l3_state_digest=runtime.body.l3.digest,
    )

    decision = runtime.promote_l3_candidate_rule(candidate, evidence_id="l3_rule_ev", required_support_raw=2)
    after = runtime.health()

    assert decision.status == L3RulePromotionStatus.PROVISIONAL
    assert decision.provisional_promoted is True
    assert decision.support_report.passed is True
    assert decision.conflict_report.passed is True
    assert decision.policy_allowed is True
    assert decision.may_answer is False
    assert decision.may_commit_l2_fact is False
    assert decision.may_execute_real_action is False
    assert after["latest_fact_count"] == before["latest_fact_count"]
    assert after["l2_clock"] == before["l2_clock"]
    assert after["l3_clock"] == before["l3_clock"]
    assert after["l3_provisional_rule_count"] == before["l3_provisional_rule_count"] + 1
    assert runtime.theorem_layer.prove(candidate.statement).valid is False
    assert runtime.trace.verify() is True


def test_p16_l3_candidate_with_insufficient_support_is_blocked():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    candidate = L3RuleCandidate(
        statement=Statement.atom("has_property", "dog", "fur"),
        support_count_raw=1,
        trace_ids=("trace_only",),
        source_rule_id="weak_sleep_rule",
        l3_state_digest=runtime.body.l3.digest,
    )

    decision = runtime.promote_l3_candidate_rule(candidate, evidence_id="l3_weak_ev", required_support_raw=2)

    assert decision.status == L3RulePromotionStatus.BLOCKED
    assert decision.provisional_promoted is False
    assert decision.support_report.evidence_supported is False
    assert decision.may_answer is False
    assert decision.may_commit_l2_fact is False
    assert decision.may_execute_real_action is False
    assert candidate.candidate_id not in runtime.l3_provisional_rules


def test_p16_l3_candidate_contradicted_by_latest_state_is_blocked():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT mary located_in garden EVID ev_mary_garden", source="p16_test"))
    candidate = L3RuleCandidate(
        statement=Statement.atom("located_in", "mary", "office"),
        support_count_raw=2,
        trace_ids=("trace_a", "trace_b"),
        source_rule_id="conflicting_l3_rule",
        l3_state_digest=runtime.body.l3.digest,
    )

    decision = runtime.promote_l3_candidate_rule(candidate, evidence_id="l3_conflict_ev", required_support_raw=2)

    assert decision.status == L3RulePromotionStatus.BLOCKED
    assert decision.conflict_report.contradiction_found is True
    assert decision.provisional_promoted is False
    assert candidate.candidate_id not in runtime.l3_provisional_rules
    assert runtime.trace.verify() is True


def test_p16_promotion_export_contains_authority_boundary():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    candidate = L3RuleCandidate(
        statement="safe_rule(alpha,beta)",
        support_count_raw=2,
        trace_ids=("trace_1", "trace_2"),
        source_rule_id="export_rule",
        l3_state_digest=runtime.body.l3.digest,
    )
    decision = runtime.promote_l3_candidate_rule(candidate, evidence_id="l3_export_ev", required_support_raw=2)
    exported = runtime.export_state()

    exported_rules = exported.get("l3_provisional_rules", [])
    assert decision.provisional_promoted is True
    assert exported_rules
    authority = exported_rules[0]["authority_boundary"]
    assert authority == {
        "may_answer": False,
        "may_commit_l2_fact": False,
        "may_execute_real_action": False,
    }
