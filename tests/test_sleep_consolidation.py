import ast
from pathlib import Path

import pytest

from htce_origin.kernel.core import EntityId, EvidenceId, FactFrame, RelationId
from htce_origin.cognition.learning import (
    EpisodeFact,
    EpisodeRecord,
    LearningError,
    NoRegressionProbe,
    RuleStatus,
    SleepConsolidator,
)
from htce_origin.governance.proof import Statement, TheoremLayer

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def _fact(subject="Mary", relation="located_in", obj="office", evid="event42"):
    return FactFrame(EntityId(subject), RelationId(relation), EntityId(obj), EvidenceId(evid))


def _episode_fact(subject="Mary", relation="located_in", obj="office", evid="event42", trace="trace42", step=0, negated=False):
    return EpisodeFact(_fact(subject, relation, obj, evid), trace_id=trace, step_index=step, negated=negated)


def test_learning_runtime_float_count_is_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "cognition" / "learning.py") == 0


def test_episode_replay_preserves_record_order():
    consolidator = SleepConsolidator()
    first = EpisodeRecord("ep1", (_episode_fact(trace="t1"),))
    second = EpisodeRecord("ep2", (_episode_fact(subject="John", obj="garden", trace="t2"),))

    consolidator.record_episode(first)
    consolidator.record_episode(second)

    assert consolidator.replay() == (first, second)


def test_repeated_episode_promotes_rule():
    consolidator = SleepConsolidator(min_support=2)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1", step=1),)))
    consolidator.record_episode(EpisodeRecord("ep2", (_episode_fact(evid="event43", trace="t2", step=2),)))

    report = consolidator.consolidate()

    assert report.replayed_episodes == 2
    assert report.promoted_count == 1
    promoted = report.promoted_rules[0]
    assert promoted.status == RuleStatus.PROMOTED
    assert promoted.support_count == 2
    assert promoted.statement == Statement.atom("located_in", "mary", "office")
    assert promoted.trace_ids == ("t1", "t2")


def test_l2_to_l3_compression_can_feed_theorem_layer_without_answering_by_itself():
    consolidator = SleepConsolidator(min_support=2)
    theorem = TheoremLayer()
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1"),)))
    consolidator.record_episode(EpisodeRecord("ep2", (_episode_fact(evid="event43", trace="t2"),)))

    report = consolidator.consolidate(theorem_layer=theorem)
    proof = theorem.prove(Statement.atom("located_in", "mary", "office"))

    assert report.promoted_count == 1
    assert proof.valid is True
    assert proof.premises[0].source == "derived"


def test_contradictory_rule_blocked_by_quarantine():
    consolidator = SleepConsolidator(min_support=2)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1"),)))
    consolidator.record_episode(EpisodeRecord("ep2", (_episode_fact(evid="correction43", trace="t2", negated=True),)))
    consolidator.record_episode(EpisodeRecord("ep3", (_episode_fact(evid="event44", trace="t3"),)))

    report = consolidator.consolidate()

    assert report.promoted_count == 0
    assert report.blocked_count == 1
    blocked = report.blocked_rules[0]
    assert blocked.status == RuleStatus.BLOCKED
    assert "contradiction quarantine" in blocked.reason
    assert blocked.trace_ids == ("t1", "t3", "t2")


def test_negative_only_material_is_retained_but_not_promoted():
    consolidator = SleepConsolidator(min_support=1)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1", negated=True),)))

    report = consolidator.consolidate()

    assert report.promoted_count == 0
    assert report.blocked_count == 1
    assert "negative-only" in report.blocked_rules[0].reason


def test_old_facts_remain_traceable_after_sleep():
    consolidator = SleepConsolidator(min_support=2)
    old_fact = _episode_fact(trace="old_trace", step=1)
    new_fact = _episode_fact(evid="event43", trace="new_trace", step=2)
    consolidator.record_episode(EpisodeRecord("ep1", (old_fact,)))
    consolidator.record_episode(EpisodeRecord("ep2", (new_fact,)))

    report = consolidator.consolidate()

    assert old_fact.signature in report.trace_index
    assert report.trace_index[old_fact.signature] == ("old_trace", "new_trace")
    assert old_fact.signature in report.old_fact_signatures


def test_sleep_does_not_break_previous_qa_lookup():
    consolidator = SleepConsolidator(min_support=2)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1", step=1),)))
    consolidator.record_episode(EpisodeRecord("ep2", (_episode_fact(evid="event43", trace="t2", step=2),)))

    before = consolidator.lookup_latest_fact("Mary", "located_in")
    assert before is not None
    assert before.fact.object.value == "office"

    report = consolidator.consolidate(
        probes=(NoRegressionProbe("where_is_mary", "office"),),
        answerer=lambda query: consolidator.lookup_latest_fact("Mary", "located_in").fact.object.value,
    )
    after = consolidator.lookup_latest_fact("Mary", "located_in")

    assert report.no_regression_passed is True
    assert report.regression_failures == ()
    assert after is not None
    assert after.fact.object.value == "office"


def test_no_regression_probe_reports_failure():
    consolidator = SleepConsolidator(min_support=1)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1"),)))

    report = consolidator.consolidate(
        probes=(NoRegressionProbe("where_is_mary", "office"),),
        answerer=lambda query: "garden",
    )

    assert report.no_regression_passed is False
    assert "where_is_mary" in report.regression_failures[0]


def test_no_regression_probe_requires_answerer():
    consolidator = SleepConsolidator(min_support=1)
    consolidator.record_episode(EpisodeRecord("ep1", (_episode_fact(trace="t1"),)))

    with pytest.raises(LearningError):
        consolidator.consolidate(probes=(NoRegressionProbe("q", "a"),))


def test_toroidal_sleep_consolidator_reduces_reconstruction_loss_without_runtime_authority():
    from htce_origin.cognition.learning import ToroidalSleepConsolidator

    start_a = (1000, 2000, 3000, 4000)
    start_b = (5000, 6000, 7000, 8000)
    residual = (128, 256, 384, 512)
    end_a = tuple((a + d) % 65536 for a, d in zip(start_a, residual))
    end_b = tuple((a + d) % 65536 for a, d in zip(start_b, residual))

    consolidator = ToroidalSleepConsolidator(dim_l2=4, dim_l3=4, learning_rate_bp=10000, sparsity_lambda_bp=0)
    before = consolidator.reconstruction_loss((start_a, start_b), (end_a, end_b))
    report = consolidator.consolidate_offline((start_a, start_b), (end_a, end_b), epochs=1)
    after = consolidator.reconstruction_loss((start_a, start_b), (end_a, end_b))

    assert report.rule_authorized is False
    assert report.sample_count == 2
    assert report.l3_state_q16 == residual
    assert report.accepted_updates >= 1
    assert after <= before
    assert report.final_reconstruction_loss == after
    assert consolidator.query_rule_q16(start_a) == end_a


def test_toroidal_sleep_consolidator_rejects_bad_batches():
    from htce_origin.cognition.learning import LearningError, ToroidalSleepConsolidator

    consolidator = ToroidalSleepConsolidator(dim_l2=2, dim_l3=1)
    with pytest.raises(LearningError):
        consolidator.consolidate_offline(((1, 2),), (), epochs=1)
    with pytest.raises(LearningError):
        consolidator.consolidate_offline(((1, 2),), ((3, 4),), epochs=0)
