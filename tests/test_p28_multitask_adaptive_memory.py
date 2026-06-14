from htce_origin import HTCERuntime


def test_p28_continual_multitask_adaptation_has_no_cross_domain_regression():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_continual_multitask_simulation(steps=18, grid_size=5)
    payload = report.as_payload()
    assert payload["passed"] is True
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["simulation_only"] is True
    assert payload["single_runtime_loop"] is True
    assert payload["no_cross_domain_regression"] is True
    assert payload["proof_gates_passed"] is True
    assert payload["topology_gates_passed"] is True
    assert payload["false_support_count"] == 0
    assert payload["wrong_turn_count"] == 0
    assert set(payload["domains_tested"]) == {"grid_nav", "dialog_slots", "babi_reasoning", "contradiction"}
    assert payload["total_l3_rules_promoted"] >= 4
    for domain, history in payload["domain_cost_history_raw"].items():
        assert history
        assert history[-1] <= history[0], domain
        assert all(cost >= 0 for cost in history)


def test_p28_probe_matrix_after_each_sleep_cycle_stays_green():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_continual_multitask_simulation(
        curriculum_domains=("grid_nav", "dialog_slots", "babi_reasoning", "contradiction"),
        steps=14,
        grid_size=5,
    )
    for episode in report.as_payload()["episodes"]:
        assert episode["probe_failure_count"] == 0
        assert episode["false_support_count"] == 0
        assert episode["wrong_turn_count"] == 0
        assert episode["retained_l3_rule_count"] > 0
        for probe in episode["probe_results"]:
            assert probe["probe_passed"] is True
            assert probe["regression_detected"] is False
            assert probe["false_support_count"] == 0
            assert probe["wrong_turn_count"] == 0
