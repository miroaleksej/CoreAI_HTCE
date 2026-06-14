from htce_origin.governance.policy import DecisionKind
from htce_origin.body.runtime import HTCERuntime, RuntimeRequest


def test_mary_at_office_query_returns_office():
    runtime = HTCERuntime()
    commit = runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    answer = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask42"))

    assert commit.decision.trace_id
    assert answer.decision.kind == DecisionKind.ANSWER
    assert answer.output == "ANSWER: office"
    assert answer.diagnostics["query"]["answer"] == "office"
    assert answer.decision.trace_id


def test_mary_moved_to_garden_latest_state_is_garden():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    runtime.tick(RuntimeRequest("FACT Mary located_in garden EVID event43"))
    answer = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask43"))
    history = runtime.memory.history("Mary", "located_in")

    assert answer.output == "ANSWER: garden"
    assert answer.diagnostics["query"]["answer"] == "garden"
    assert len(history) == 2
    assert any(item.status.value == "superseded" and item.object_value == "office" for item in history)
    assert runtime.memory.query("Mary", "located_in").answer == "garden"


def test_unknown_fact_triggers_clarification_or_refusal():
    runtime = HTCERuntime()
    response = runtime.tick(RuntimeRequest("QUERY John location EVID ask99"))

    assert response.decision.kind in {DecisionKind.ASK_CLARIFICATION, DecisionKind.REFUSE}
    assert response.decision.trace_id
    assert response.diagnostics["query"]["answer"] is None


def test_contradiction_quarantines_latest_state():
    runtime = HTCERuntime()
    runtime.tick(RuntimeRequest("FACT Mary located_in office EVID event42"))
    neg = runtime.tick(RuntimeRequest("NEGATE Mary located_in office EVID correction43"))
    answer = runtime.tick(RuntimeRequest("QUERY Mary location EVID ask44"))

    assert neg.decision.kind == DecisionKind.REFUSE
    assert "quarantined" in neg.output.lower()
    assert answer.decision.kind == DecisionKind.REFUSE
    assert answer.diagnostics["query"]["status"] == "quarantined"
    assert runtime.trace.verify()


def test_supported_plain_natural_language_enters_checked_nlu_air_bridge():
    runtime = HTCERuntime()
    response = runtime.tick(RuntimeRequest("Mary is in office"))

    assert response.decision.kind == DecisionKind.ANSWER
    assert response.output == "COMMIT: mary located_in office"
    assert runtime.health()["latest_fact_count"] == 1
