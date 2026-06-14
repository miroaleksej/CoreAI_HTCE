from htce_origin.evaluation.benchmarks import BenchmarkDecision, BenchmarkHarness, MinimalBABIHarness, MinimalGoldMemory


def _result_by_name(report, name):
    return next(result for result in report.results if result.name == name)


def test_babi_task1_latest_state_correctness():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "babi_task1_latest_state")
    assert result.passed
    assert result.answer == "garden"
    assert result.decision == BenchmarkDecision.ANSWER


def test_babi_task15_deduction_correctness():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "babi_task15_basic_deduction")
    assert result.passed
    assert result.answer == "wolf"


def test_babi_task16_induction_smoke():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "babi_task16_basic_induction_smoke")
    assert result.passed
    assert result.answer == "white"
    assert result.decision == BenchmarkDecision.HYPOTHESIS


def test_dialog_babi_smoke():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "dialog_babi_slot_smoke")
    assert result.passed
    assert result.answer == "api_call italian rome cheap"


def test_unknown_questions_refused():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "unknown_location_refusal")
    assert result.passed
    assert result.decision == BenchmarkDecision.REFUSE
    assert result.answer is None
    assert result.unsupported_query


def test_memory_stress_preserves_latest_state():
    report = MinimalBABIHarness().run_all()
    result = _result_by_name(report, "memory_stress_latest_state")
    assert result.passed
    assert result.answer == "room_63"


def test_false_support_rate_is_zero_on_minimal_goldset():
    report = MinimalBABIHarness().run_all()
    assert report.passed
    assert report.false_support_rate_bp == 0
    false_support = _result_by_name(report, "false_support_blocking")
    assert false_support.passed
    assert not false_support.false_support


def test_benchmark_harness_backward_compatible_smoke_name():
    results = BenchmarkHarness().run_smoke()
    assert len(results) >= 7
    assert all(result.trace_id for result in results)


def test_minimal_gold_memory_digest_changes_with_commits():
    memory = MinimalGoldMemory()
    before = memory.digest()
    memory.commit("Mary", "location", "garden", "e1")
    after = memory.digest()
    assert before != after


def test_human_help_scenario_cards_do_not_leak_answers_or_rubric():
    harness = BenchmarkHarness()
    pack = harness.build_human_help_scenarios(20)
    assert pack.scenario_count == 20
    assert pack.no_answer_leakage_contract.passed()
    assert len(pack.hidden_evaluation_hashes) == 20
    for card in pack.public_cards:
        engine_input = card.engine_input()
        assert engine_input["contains_answer_key"] == 0
        assert engine_input["contains_template_answer"] == 0
        assert engine_input["contains_hidden_rubric_text"] == 0
        assert "hidden_criteria" not in engine_input
        assert "hidden_process_criteria" not in engine_input


def test_hidden_criteria_hash_committed_separately_from_engine_input():
    harness = BenchmarkHarness()
    pack = harness.build_human_help_scenarios(20)
    hidden_by_id = {item.scenario_id: item.hidden_criteria_hash for item in pack.hidden_evaluation_hashes}
    assert set(hidden_by_id) == {card.scenario_id for card in pack.public_cards}
    assert all(len(hidden_hash) == 64 for hidden_hash in hidden_by_id.values())
    for card in pack.public_cards:
        assert hidden_by_id[card.scenario_id] not in str(card.engine_input())


def test_no_answer_leakage_execution_passes_contract():
    report = BenchmarkHarness().run_no_answer_leakage_scenarios(20)
    assert report.no_answer_leakage_passed == 1
    assert report.target_met == 1
    assert report.false_fact_promoted_count == 0
    assert report.settled_truth_commit_from_web_count == 0
    assert report.min_ablation_margin_bp > 0
    assert report.answer_diversity_count >= 10
    assert all(row.no_leakage_passed() for row in report.rows)


def test_no_answer_leakage_rows_have_component_scores_and_ablation_margins():
    harness = BenchmarkHarness()
    report = harness.run_no_answer_leakage_scenarios(20)
    margins = harness.compute_ablation_margins(report.rows)
    assert margins["min_margin_bp"] == report.min_ablation_margin_bp
    assert margins["mean_margin_bp"] == report.mean_ablation_margin_bp
    for row in report.rows:
        assert row.memory_recall_bp > 0
        assert row.replay_use_bp > 0
        assert row.evidence_bridge_bp > 0
        assert row.world_model_bp > 0
        assert row.uncertainty_calibration_bp > 0
        assert row.faithfulness_bp > 0
        assert row.bounded_answer_bp > 0
        assert row.human_help_score_bp > max(row.ablation_scores_bp.values())
        assert row.trace_id


def test_false_support_rate_helper_stays_zero_on_minimal_goldset():
    harness = BenchmarkHarness()
    report = harness.run_all()
    assert harness.compute_false_support_rate(report.results) == 0



def test_hard_probe_pack_has_required_organs_and_forbidden_failures():
    harness = BenchmarkHarness()
    probes = harness.build_hard_probes()
    assert {probe.task_family for probe in probes} == {
        "claim_boundary_help",
        "contradiction_memory_help",
        "experience_replay_help",
        "missing_info_help",
        "uncertainty_calibration_help",
        "weak_evidence_help",
        "web_noise_resistance_help",
        "world_model_reuse_help",
    }
    allowed = {"memory", "replay", "evidence", "proof", "world", "uncertainty", "trace"}
    for probe in probes:
        assert set(probe.required_organs).issubset(allowed)
        assert probe.forbidden_failures
        assert probe.as_tuple()[0] == probe.task_family


def test_hard_probe_weighted_component_scores_are_positive():
    report = BenchmarkHarness().run_hard_probes()
    assert report.passed
    assert report.score_bp > 0
    assert report.memory_recall_bp > 0
    assert report.replay_use_bp > 0
    assert report.evidence_bridge_bp > 0
    assert report.world_model_reuse_bp > 0
    assert report.uncertainty_calibration_bp > 0
    assert report.faithfulness_bp > 0
    for row in report.rows:
        assert row.score_bp == (
            18 * row.memory_recall_bp
            + 14 * row.replay_use_bp
            + 18 * row.evidence_bridge_bp
            + 14 * row.world_model_reuse_bp
            + 18 * row.uncertainty_calibration_bp
            + 18 * row.faithfulness_bp
        ) // 100
        assert row.trace_id


def test_hard_probe_false_support_rate_is_zero():
    harness = BenchmarkHarness()
    report = harness.run_hard_probes()
    assert report.false_supported_answers == 0
    assert report.supported_answers > 0
    assert report.false_support_rate_bp == 0
    assert harness.compute_hard_probe_false_support_rate(report.rows) == 0


def test_hard_probe_forbidden_failures_do_not_fire():
    report = BenchmarkHarness().run_hard_probes()
    assert all(row.forbidden_failure_count == 0 for row in report.rows)
    assert all(row.passed() for row in report.rows)



def test_external_babi_loader_reads_task_rows_from_optional_path(tmp_path):
    dataset = tmp_path / "qa1_test.txt"
    dataset.write_text(
        "1 Mary went to the kitchen.\n"
        "2 John moved to the hallway.\n"
        "3 Mary journeyed to the garden.\n"
        "4 Where is Mary?\tgarden\t3\n",
        encoding="utf-8",
    )
    rows = BenchmarkHarness().load_babi_task(tmp_path, 1)
    assert len(rows) == 1
    assert rows[0].story[-1] == "Mary journeyed to the garden."
    assert rows[0].question == "Where is Mary?"
    assert rows[0].expected == "garden"
    assert rows[0].support_ids == (3,)


def test_external_babi_subset_scores_accuracy_and_false_support(tmp_path):
    (tmp_path / "qa1_test.txt").write_text(
        "1 Mary went to the kitchen.\n"
        "2 Mary moved to the garden.\n"
        "3 Where is Mary?\tgarden\t2\n",
        encoding="utf-8",
    )
    (tmp_path / "qa15_test.txt").write_text(
        "1 sheep are afraid of wolf.\n"
        "2 Gertrude is a sheep.\n"
        "3 What is Gertrude afraid of?\twolf\t1 2\n",
        encoding="utf-8",
    )
    (tmp_path / "qa16_test.txt").write_text(
        "1 Lily is a swan.\n"
        "2 Lily is white.\n"
        "3 Greg is a swan.\n"
        "4 Greg is white.\n"
        "5 Brian is a swan.\n"
        "6 What color is Brian?\twhite\t2 4\n",
        encoding="utf-8",
    )
    report = BenchmarkHarness().run_external_babi_subset(tmp_path, tasks=(1, 15, 16))
    assert report.passed
    assert report.total == 3
    assert report.accuracy_bp == 10000
    assert report.false_support_rate_bp == 0
    assert all(row.evidence_not_empty for row in report.rows)


def test_dialog_babi_loader_and_external_smoke_do_not_bundle_data(tmp_path):
    dialog = tmp_path / "dialog_babi_task1_sample.txt"
    dialog.write_text(
        "1 hello\thello what can I help you with today\n"
        "2 italian\twhat city should I search in\n"
        "3 rome\twhat price range do you want\n"
        "4 cheap\tapi_call italian rome cheap\n",
        encoding="utf-8",
    )
    rows = BenchmarkHarness().load_dialog_babi(tmp_path, 1)
    assert len(rows) == 4
    report = BenchmarkHarness().run_external_dialog_smoke(tmp_path, task_id=1)
    assert report.passed
    assert report.accuracy_bp == 10000
    assert report.false_support_rate_bp == 0
    assert report.rows[-1].answer == "api_call italian rome cheap"


def test_external_loader_missing_path_fails_closed(tmp_path):
    missing = tmp_path / "not_downloaded"
    try:
        BenchmarkHarness().load_babi_task(missing, 1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("external benchmark loader must fail closed when path is absent")


def test_evidence_anchor_boundary_smoke_keeps_web_anchors_evidence_only():
    report = BenchmarkHarness().run_evidence_anchor_boundary_smoke()
    assert report.passed
    assert report.claim_allowed == 1
    assert report.weak_source_downweighted == 1
    assert report.retracted_source_blocked == 1
    assert report.contradiction_blocked == 1
    assert report.web_anchor_settled_fact_count == 0
    assert report.trace_id


def test_claim_support_from_external_anchors_uses_integer_thresholds():
    harness = BenchmarkHarness()
    anchors = harness.build_evidence_anchors("claim:toroidal-runtime-is-integer-only")
    report = harness.compute_claim_support_from_anchors("claim:toroidal-runtime-is-integer-only", anchors)
    assert report.support_bp >= 6000
    assert report.contradiction_bp == 0
    assert report.claim_allowed == 1
    assert report.as_payload()["web_anchor_equals_settled_fact"] == 0



def test_evidence_threshold_calibration_in_benchmark_smoke_blocks_single_weak_source():
    report = BenchmarkHarness().run_evidence_anchor_boundary_smoke()
    assert report.passed
    assert report.primary_replicated_support_passes == 1
    assert report.single_weak_source_fails == 1
    assert report.support_threshold_bp > 0
    assert report.contradiction_threshold_bp > 0


def test_external_babi_all_reports_per_task_metrics(tmp_path):
    for task in range(1, 21):
        (tmp_path / f"qa{task}_test.txt").write_text(
            "1 Mary went to the kitchen.\n"
            "2 Mary moved to the garden.\n"
            "3 Where is Mary?\tgarden\t2\n",
            encoding="utf-8",
        )
    report = BenchmarkHarness().run_external_babi_all(tmp_path, max_examples_per_task=1)
    assert report.total == 20
    assert report.unsupported_answer_count == 0
    assert report.false_support_rate_bp == 0
    assert set(report.task_metrics) == {f"qa{i}" for i in range(1, 21)}
    for metrics in report.task_metrics.values():
        assert metrics["accuracy_bp"] == 10000
        assert metrics["false_support_rate_bp"] == 0
        assert metrics["unsupported_answer_count"] == 0
        assert "refusal_rate_bp" in metrics


def test_external_dialog_all_reports_per_task_metrics(tmp_path):
    for task in range(1, 7):
        (tmp_path / f"dialog_babi_task{task}_sample.txt").write_text(
            "1 hello\thello what can I help you with today\n"
            "2 italian\twhat city should I search in\n"
            "3 rome\twhat price range do you want\n"
            "4 cheap\tapi_call italian rome cheap\n",
            encoding="utf-8",
        )
    report = BenchmarkHarness().run_external_dialog_all(tmp_path, max_examples_per_task=4)
    assert report.total == 24
    assert report.unsupported_answer_count == 0
    assert report.false_support_rate_bp == 0
    assert set(report.task_metrics) == {f"dialog_babi_task{i}" for i in range(1, 7)}
    for metrics in report.task_metrics.values():
        assert metrics["accuracy_bp"] == 10000
        assert metrics["false_support_rate_bp"] == 0
        assert metrics["unsupported_answer_count"] == 0
        assert "refusal_rate_bp" in metrics
