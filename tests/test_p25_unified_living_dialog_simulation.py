from htce_origin import HTCERuntime


def test_p25_unified_living_dialog_runs_inside_one_simulation_loop():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_living_active_agent_simulation(steps=16, grid_size=5, include_dialog_policy=True)
    payload = report.as_payload()
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["unified_simulation"] is True
    assert payload["reached_goal"] is True
    assert payload["dialog_metrics"]["total_turns"] == 8
    assert payload["dialog_metrics"]["wrong_turns"] == 0
    assert payload["dialog_metrics"]["false_support_count"] == 0
    assert payload["dialog_metrics"]["act_simulated_count"] == 5
    assert payload["dialog_metrics"]["ask_clarification_count"] == 3
    assert payload["l2_fact_count_after"] > payload["l2_fact_count_before"]
    outputs = [turn["output"] for turn in payload["dialog_turns"]]
    assert any("domain=restaurant cuisine=chinese location=rome price=cheap" in out for out in outputs)
    assert any("domain=hotel location=paris stars=4" in out for out in outputs)
    assert any("missing required dialog slots: cuisine, price" in out for out in outputs)


def test_p25_domain_shift_does_not_mix_hotel_and_restaurant_slots():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_living_active_agent_simulation(steps=16, grid_size=5, include_dialog_policy=True)
    payload = report.as_payload()
    action_outputs = [turn["output"] for turn in payload["dialog_turns"] if turn["decision_kind"] == "act_simulated"]
    hotel_calls = [out for out in action_outputs if "domain=hotel" in out]
    restaurant_calls = [out for out in action_outputs if "domain=restaurant" in out]
    assert hotel_calls == ["api_call domain=hotel location=paris stars=4"]
    assert all("stars=" not in out for out in restaurant_calls)
    assert payload["domain_contexts"].get("restaurant") == 2
    assert payload["domain_contexts"].get("hotel") == 1
