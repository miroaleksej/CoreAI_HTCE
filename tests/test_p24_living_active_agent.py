from htce_origin import HTCERuntime


def test_p24_living_active_agent_reduces_goal_distance_and_keeps_boundary():
    runtime = HTCERuntime()
    runtime.wake()
    before = runtime.health()
    report = runtime.run_living_active_agent_simulation(steps=16, grid_size=5)
    payload = report.as_payload()
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["heartbeat_count"] >= 1
    assert payload["final_goal_distance"] < payload["start_goal_distance"]
    assert payload["reached_goal"] is True
    assert payload["l2_fact_count_before"] == before["latest_fact_count"]
    assert payload["l2_fact_count_after"] == before["latest_fact_count"]
    assert payload["l3_clock_after"] == payload["l3_clock_before"]


def test_p24_living_active_agent_has_continuous_l1_and_self_model_updates():
    runtime = HTCERuntime()
    runtime.wake()
    l1_before = runtime.health()["l1_clock"]
    report = runtime.run_living_active_agent_simulation(steps=10, grid_size=5)
    payload = report.as_payload()
    assert runtime.health()["l1_clock"] >= l1_before + payload["heartbeat_count"]
    assert runtime.world_model.self_model.observations == payload["heartbeat_count"]
    assert payload["average_prediction_error_bp"] >= 0
    assert payload["min_viability_bp"] > 0
    assert any(count > 0 for name, count in payload["action_counts"].items() if name.startswith("move_"))
