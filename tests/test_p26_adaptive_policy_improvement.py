from htce_origin import HTCERuntime


def test_p26_adaptive_policy_improvement_runs_inside_one_runtime_trace():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_adaptive_policy_improvement_simulation(steps=18, grid_size=5)
    payload = report.as_payload()
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["simulation_only"] is True
    assert payload["single_runtime_loop"] is True
    assert payload["improvement_verified"] is True
    assert payload["episode_2"]["adaptive_cost_raw"] < payload["episode_1"]["adaptive_cost_raw"]
    assert payload["l3_rules_promoted_during_sleep"] >= 1
    assert payload["l3_provisional_rules_total"] >= 1
    assert "1_0" in payload["learned_avoid_cells"]


def test_p26_preserves_dialog_honesty_and_reduces_grid_recovery():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_adaptive_policy_improvement_simulation(steps=18, grid_size=5)
    ep1 = report.episode_1.as_payload()
    ep2 = report.episode_2.as_payload()
    assert ep1["wrong_turns"] == 0
    assert ep2["wrong_turns"] == 0
    assert ep1["false_support_count"] == 0
    assert ep2["false_support_count"] == 0
    assert ep1["recovery_actions"] > ep2["recovery_actions"]
    assert ep1["heartbeat_count"] > ep2["heartbeat_count"]
    assert ep1["living_dialog_report"]["dialog_metrics"]["total_turns"] == 8
    assert ep2["living_dialog_report"]["dialog_metrics"]["total_turns"] == 8
