from htce_origin.kernel.core import EvidenceId, FactFrame, EntityId, RelationId, fact_delta
from htce_origin.body.memory import FactDeltaStore, FactStatus


def _fact(subject, relation, obj, evidence):
    return FactFrame(EntityId(subject), RelationId(relation), EntityId(obj), EvidenceId(evidence))


def test_fact_delta_memory_commits_latest_state():
    store = FactDeltaStore()
    fact = _fact("Mary", "located_in", "office", "event42")
    record = store.commit(fact_delta(fact), trace_id="trace1")
    query = store.query("Mary", "located_in")

    assert record.status == FactStatus.ACTIVE
    assert query.answered
    assert query.answer == "office"
    assert query.record_id == record.record_id


def test_new_location_supersedes_old_location_not_conflict():
    store = FactDeltaStore()
    office = store.commit(fact_delta(_fact("Mary", "located_in", "office", "event42")), trace_id="trace1")
    garden = store.commit(fact_delta(_fact("Mary", "located_in", "garden", "event43")), trace_id="trace2")
    history = store.history("Mary", "located_in")
    query = store.query("Mary", "located_in")

    assert query.answer == "garden"
    assert garden.revision == office.revision + 1
    assert any(item.status == FactStatus.SUPERSEDED and item.object_value == "office" for item in history)
    assert not store.conflicts.has_conflict(garden.key)


def test_old_facts_remain_traceable_after_supersession():
    store = FactDeltaStore()
    store.commit(fact_delta(_fact("Mary", "located_in", "office", "event42")), trace_id="trace1")
    store.commit(fact_delta(_fact("Mary", "located_in", "garden", "event43")), trace_id="trace2")
    history = store.history("Mary", "located_in")

    assert len(history) == 2
    assert {item.object_value for item in history} == {"office", "garden"}
    assert all(item.trace_id for item in history)


def test_direct_negation_quarantines_contradiction():
    store = FactDeltaStore()
    store.commit(fact_delta(_fact("Mary", "located_in", "office", "event42")), trace_id="trace1")
    neg = store.commit_negation(_fact("Mary", "located_in", "office", "correction43"), trace_id="trace2")
    query = store.query("Mary", "located_in")

    assert neg.status == FactStatus.QUARANTINED
    assert query.status == FactStatus.QUARANTINED
    assert not query.answered
    assert "quarantined" in query.reason


def test_unknown_query_refuses_without_fabricating_answer():
    store = FactDeltaStore()
    query = store.query("John", "located_in")

    assert not query.answered
    assert query.answer is None
    assert query.status == "unknown"


def test_weighted_l2_fact_commit_applies_confidence_weight_to_phase_delta():
    from htce_origin.body.layers import L123Body
    from htce_origin.kernel.q16 import q_mod

    body = L123Body(dimension=4)
    fact = FactFrame(
        EntityId("Mary"),
        RelationId("located_in"),
        EntityId("office"),
        EvidenceId("event_weighted"),
        confidence_bp=9000,
    )
    delta = fact_delta(fact, dimension=4, modulus=body.modulus)
    expected = tuple(q_mod(value * delta.weight, body.modulus) for value in delta.delta)

    body.commit_l2_fact(delta)

    assert body.l2_clean_vector() == expected
    assert body.l2.vector != expected
    assert body.l2.vector != delta.delta


def test_associative_toroidal_read_uses_weighted_l2_state_and_supported_record():
    from htce_origin.body.layers import L123Body

    body = L123Body(dimension=4)
    store = FactDeltaStore()
    fact = FactFrame(
        EntityId("Mary"),
        RelationId("located_in"),
        EntityId("office"),
        EvidenceId("event_toroidal"),
        confidence_bp=9000,
    )
    delta = fact_delta(fact, dimension=4, modulus=body.modulus)
    record = store.commit(delta, trace_id="trace_toroidal")
    body.commit_l2_fact(delta)

    query = store.associative_toroidal_read(
        "Mary",
        "located_in",
        current_l2_state=body.l2_clean_vector(),
        candidate_objects=("office", "garden"),
        trace_id="trace_query",
        modulus=body.modulus,
    )

    assert query.answered
    assert query.answer == "office"
    assert query.record_id == record.record_id
    assert query.evidence_id == "event_toroidal"
    assert "associative_toroidal_read_min_loss" in query.reason



def test_l2_episode_tag_unbinding_subtracts_weighted_tag_accumulator_for_multiple_facts():
    from htce_origin.body.layers import L123Body
    from htce_origin.kernel.q16 import q_vector_add

    body = L123Body(dimension=4)
    first = fact_delta(FactFrame(EntityId("Mary"), RelationId("located_in"), EntityId("office"), EvidenceId("ev1"), confidence_bp=9000), dimension=4, modulus=body.modulus)
    second = fact_delta(FactFrame(EntityId("John"), RelationId("located_in"), EntityId("garden"), EvidenceId("ev2"), confidence_bp=8000), dimension=4, modulus=body.modulus)
    expected = q_vector_add(body.weighted_fact_delta(first), body.weighted_fact_delta(second), body.modulus)

    body.commit_l2_fact(first)
    body.commit_l2_fact(second)

    assert body.l2_clean_vector() == expected
    assert body.l2_episode_fact_count == 2
    assert len(body.l2_active_contributions) == 2


def test_l2_same_key_supersession_reverses_previous_active_residual():
    from htce_origin.body.layers import L123Body

    body = L123Body(dimension=4)
    old = fact_delta(FactFrame(EntityId("Mary"), RelationId("located_in"), EntityId("office"), EvidenceId("ev_old"), confidence_bp=9000), dimension=4, modulus=body.modulus)
    new = fact_delta(FactFrame(EntityId("Mary"), RelationId("located_in"), EntityId("garden"), EvidenceId("ev_new"), confidence_bp=7000), dimension=4, modulus=body.modulus)

    body.commit_l2_fact(old)
    body.commit_l2_fact(new)

    assert body.l2_clean_vector() == body.weighted_fact_delta(new)
    assert body.l2_episode_fact_count == 1
    assert len(body.l2_active_contributions) == 1


def test_l2_consolidation_reset_anchors_and_zeros_working_torus():
    from htce_origin.body.layers import L123Body

    body = L123Body(dimension=4)
    item = fact_delta(FactFrame(EntityId("Mary"), RelationId("located_in"), EntityId("office"), EvidenceId("ev1"), confidence_bp=9000), dimension=4, modulus=body.modulus)
    body.commit_l2_fact(item)
    assert any(body.l2.vector)
    assert any(body.l2_clean_vector())

    transition, anchor = body.consolidate_l2_episode(episode_id="ep0", promoted_rules_count=1, evidence_id="sleep0")

    assert transition.target_layer.value == "L2"
    assert anchor.fact_count == 1
    assert anchor.promoted_rules_count == 1
    assert body.l2.vector == tuple(0 for _ in range(body.dimension))
    assert body.l2_clean_vector() == tuple(0 for _ in range(body.dimension))
    assert body.l2_episode_fact_count == 0
    assert len(body.l2_active_contributions) == 0
    assert len(body.l2_archived_anchors) == 1
