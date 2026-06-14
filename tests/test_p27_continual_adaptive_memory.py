from htce_origin import HTCERuntime


def test_p27_continual_adaptive_memory_accumulates_without_regression():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_continual_adaptive_memory_simulation(episodes=5, steps=18, grid_size=5)
    payload = report.as_payload()
    costs = [episode["adaptive_cost_raw"] for episode in payload["episodes"]]
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["simulation_only"] is True
    assert payload["single_runtime_loop"] is True
    assert payload["no_regression_passed"] is True
    assert payload["monotonic_cost_passed"] is True
    assert payload["proof_gates_passed"] is True
    assert payload["topology_gates_passed"] is True
    assert payload["babi_dialog_probes_passed"] is True
    assert payload["false_support_count"] == 0
    assert payload["wrong_turn_count"] == 0
    assert costs[1] < costs[0]
    assert all(costs[index] <= costs[index - 1] for index in range(1, len(costs)))
    assert "1_0" in payload["learned_avoid_cells_final"]
    assert set(payload["learned_dialog_slots_final"]) >= {"cuisine", "price", "stars"}
    assert payload["retained_l3_rules_final"] >= 4


def test_p27_episode_probes_stay_green_after_each_sleep_cycle():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_continual_adaptive_memory_simulation(episodes=3, steps=18, grid_size=5)
    for episode in report.as_payload()["episodes"]:
        assert episode["probe_failure_count"] == 0
        assert episode["probe_passed_count"] == episode["probe_total_count"]
        assert episode["proof_gate_passed"] is True
        assert episode["topology_gate_passed"] is True
        assert episode["l3_rule_regression_count"] == 0
        assert episode["false_support_count"] == 0
        assert episode["wrong_turns"] == 0
