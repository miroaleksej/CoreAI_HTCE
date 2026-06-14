from htce_origin import __version__, HTCERuntime


def test_v1_clean_system_external_revalidation_passes_inside_one_runtime():
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_v1_clean_system_revalidation(stress_steps=16, grid_size=5)
    payload = report.as_payload()
    assert __version__.startswith("1.0.0-")
    assert payload["passed"] is True
    assert payload["total_external_rows"] >= 12
    assert payload["external_rows_passed"] == payload["total_external_rows"]
    assert payload["external_false_support_count"] == 0
    assert payload["answer_key_visible_to_engine_count"] == 0
    assert payload["dialog_loader_strict_passed"] is True
    assert payload["no_external_regression"] is True
    assert payload["proof_gates_passed"] is True
    assert payload["topology_gates_passed"] is True
    assert payload["trace_verified"] is True
    assert payload["real_actions_allowed"] is False
    assert payload["simulation_only"] is True
    assert payload["clean_single_runtime_loop"] is True


def test_v1_external_rows_do_not_receive_gold_answers():
    runtime = HTCERuntime()
    runtime.wake()
    payload = runtime.run_v1_clean_system_revalidation(stress_steps=8, grid_size=5).as_payload()
    for row in payload["external_rows"]:
        assert row["answer_key_visible_to_engine"] == 0
        assert row["passed"] is True
        assert row["false_support"] == 0
        assert row["engine_input_hash"] != row["expected_digest"]
