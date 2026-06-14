from pathlib import Path
import json
import subprocess
import sys

from htce_origin import HTCERuntime, RuntimeRequest
from htce_origin.language.nlu_air_bridge import NluToAirBridge
from htce_origin.evaluation.benchmarks import ExternalBenchmarkRow
from htce_origin.evaluation.official_harness import P17OfficialBenchmarkHarness


ROOT = Path(__file__).resolve().parents[1]


def test_nlu_air_bridge_translates_plain_fact_and_query_through_runtime():
    runtime = HTCERuntime()
    runtime.wake()
    fact = runtime.tick(RuntimeRequest("Mary went to office."))
    query = runtime.tick(RuntimeRequest("Where is Mary?"))
    assert fact.output == "COMMIT: mary located_in office"
    assert query.output == "ANSWER: office"
    assert query.diagnostics["authorization"]["answer_allowed"] is True


def test_babi_object_tracking_uses_story_state_not_gold_answer():
    runtime = HTCERuntime()
    runtime.wake()
    for text in (
        "Mary went to office.",
        "Mary grabbed the football.",
        "Mary went to kitchen.",
    ):
        runtime.tick(RuntimeRequest(text))
    response = runtime.tick(RuntimeRequest("Where is the football?"))
    assert response.output == "ANSWER: kitchen"
    assert response.diagnostics["proof_id"]


def test_l3_class_rule_can_answer_through_proof_not_direct_l3_authority():
    runtime = HTCERuntime()
    runtime.wake()
    for text in (
        "Lily is a swan.",
        "Swan is afraid of wolf.",
    ):
        runtime.tick(RuntimeRequest(text))
    response = runtime.tick(RuntimeRequest("What is Lily afraid of?"))
    assert response.output == "ANSWER: wolf"
    assert response.diagnostics["authorization"]["answer_allowed"] is True
    assert "CLASS_RULE_DEDUCTION" in response.diagnostics["proof_path"]


def test_l3_induction_remains_hypothesis_not_answer():
    runtime = HTCERuntime()
    runtime.wake()
    for text in (
        "Lily is a swan.",
        "Bernice is a swan.",
        "Bernice is white.",
        "Gertrude is a swan.",
        "Gertrude is white.",
    ):
        runtime.tick(RuntimeRequest(text))
    response = runtime.tick(RuntimeRequest("What color is Lily?"))
    assert response.output == "HYPOTHESIS: white"
    assert response.diagnostics["authorization"]["answer_allowed"] is False
    assert response.diagnostics["authorization"]["hypothesis_allowed"] is True


def test_official_external_babi_row_calls_runtime_path_without_expected_leak():
    harness = P17OfficialBenchmarkHarness()
    row = ExternalBenchmarkRow(
        task_id="babi_task1",
        row_id="unit:1",
        story=("Mary went to the office.", "Mary went to the garden."),
        question="Where is Mary?",
        expected="garden",
        support_ids=(2,),
        source_path="unit",
    )
    decision, answer, evidence_path, trace_hash, false_support = harness._run_honest_babi_row(row)
    assert decision.value == "answer"
    assert answer == "garden"
    assert evidence_path
    assert trace_hash
    assert false_support == 0


def test_official_harness_source_no_minimal_oracle_in_external_methods():
    source = (ROOT / "htce_origin/evaluation/official_harness.py").read_text(encoding="utf-8")
    external_region = source[source.index("def run_external_babi_20"):source.index("def _report")]
    assert "_minimal_answer_babi_row" not in external_region
    assert "_minimal_answer_dialog_row" not in external_region
    assert "_run_honest_babi_row" in source
    assert "_run_honest_dialog_row" in source


def test_version_sync_gate_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/00_gates/check_version_sync.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_p22_discourse_coreference_and_carrier_chain_use_proof_path():
    runtime = HTCERuntime()
    runtime.wake()
    for text in (
        "John journeyed to hallway.",
        "After that he journeyed to garden.",
        "There he grabbed the football.",
        "Then he travelled to kitchen.",
    ):
        runtime.tick(RuntimeRequest(text))
    response = runtime.tick(RuntimeRequest("Where is the football?"))
    assert response.output == "ANSWER: kitchen"
    assert "TRANSITIVE_LOCATION" in response.diagnostics["proof_path"]
    assert len(response.diagnostics["evidence_ids"]) >= 2


def test_p22_irregular_lemmas_do_not_break_class_rule_deduction():
    runtime = HTCERuntime()
    runtime.wake()
    for text in (
        "Lily is a swan.",
        "Swans are afraid of wolves.",
    ):
        runtime.tick(RuntimeRequest(text))
    response = runtime.tick(RuntimeRequest("What is Lily afraid of?"))
    assert response.output == "ANSWER: wolf"
    assert "CLASS_RULE_DEDUCTION" in response.diagnostics["proof_path"]


def test_p22_unresolved_pronoun_refuses_without_guessing():
    runtime = HTCERuntime()
    runtime.wake()
    response = runtime.tick(RuntimeRequest("He went to office."))
    assert response.output.startswith("REFUSE:")
    assert response.diagnostics["air_error"]["nlu_air_bridge"] == "ambiguous_or_unsupported_translation"


def test_p22_dialog_loader_supports_usr_sys_and_strict_task_filter(tmp_path):
    (tmp_path / "dialog_babi_task1_sample.txt").write_text(
        "USR|hello\nSYS|hello what can I help you with today\nUSR|italian\nSYS|what city should I search in\n",
        encoding="utf-8",
    )
    (tmp_path / "dialog_babi_task5_sample.txt").write_text(
        "USR|hello\nSYS|task five only\n",
        encoding="utf-8",
    )
    from htce_origin.evaluation.benchmarks import BenchmarkHarness

    rows = BenchmarkHarness().load_dialog_babi(tmp_path, 1)
    assert len(rows) == 2
    assert all(row.task_id == "dialog_babi_task1" for row in rows)
    assert {row.expected for row in rows} == {"hello what can I help you with today", "what city should I search in"}
    assert all("task five only" not in " ".join(row.story) for row in rows)


def test_p22_target_babi_subset_closure_reports_outcome_categories():
    from htce_origin.evaluation.benchmarks import BenchmarkHarness

    report = BenchmarkHarness().run_external_babi_subset(
        ROOT / "data/official_benchmarks/babi/tasks_1-20_v1-2/en",
        tasks=(1, 2, 3, 6, 9, 10, 11, 15, 16),
        max_examples_per_task=2,
    )
    assert report.summary()["passed_count"] == report.summary()["total"]
    assert report.summary()["false_support_rate_bp"] == 0
    assert report.summary()["outcome_counts"]["wrong"] == 0
    assert report.summary()["outcome_counts"]["hypothesis"] >= 1


def test_p23_dialog_slot_correction_supersedes_and_api_call_is_proven():
    runtime = HTCERuntime()
    runtime.wake()
    runtime.tick(RuntimeRequest("I want italian food in Rome."))
    runtime.tick(RuntimeRequest("Actually, make it Chinese."))
    missing = runtime.tick(RuntimeRequest("Book a table."))
    assert missing.decision.kind.value == "ask_clarification"
    assert missing.output.startswith("ASK_CLARIFICATION:")
    assert "price" in missing.diagnostics["missing_slots"]

    runtime.tick(RuntimeRequest("cheap"))
    response = runtime.tick(RuntimeRequest("Book a table."))
    assert response.decision.kind.value == "act_simulated"
    assert response.output == "api_call domain=restaurant cuisine=chinese location=rome price=cheap"
    assert "API_CALL_READY" in response.diagnostics["proof_path"]
    cuisine_history = runtime.memory.history("current_dialog_restaurant_1", "has_slot_value_cuisine")
    assert len(cuisine_history) == 2
    assert cuisine_history[-2].status.value == "superseded"
    assert cuisine_history[-1].object_value == "chinese"


def test_p23_missing_dialog_slots_ask_clarification_not_refuse():
    runtime = HTCERuntime()
    runtime.wake()
    runtime.tick(RuntimeRequest("I want italian food."))
    response = runtime.tick(RuntimeRequest("Book a table."))
    assert response.decision.kind.value == "ask_clarification"
    assert response.output.startswith("ASK_CLARIFICATION:")
    assert set(response.diagnostics["missing_slots"]) == {"location", "price"}


def test_p23_multiple_valid_dialog_answers_and_api_call_surface_forms_match():
    from htce_origin.evaluation.benchmarks import BenchmarkHarness, ExternalBenchmarkCaseResult, BenchmarkDecision

    row = ExternalBenchmarkRow(
        task_id="dialog_babi_task1",
        row_id="p23:alternatives",
        story=(),
        question="cheap",
        expected=("api_call chinese rome cheap", "api_call cuisine=chinese location=rome price=cheap"),
        support_ids=(1,),
        source_path="unit",
    )
    result = ExternalBenchmarkCaseResult(
        row=row,
        decision=BenchmarkDecision.ANSWER,
        answer="api_call cuisine=chinese location=rome price=cheap",
        evidence_ids=(1,),
        trace_id="trace",
        false_support=0,
    )
    assert result.answer_match is True
    assert result.passed is True


def test_p23_dialog_external_smoke_uses_runtime_slot_memory_and_gold_after_response(tmp_path):
    from htce_origin.evaluation.benchmarks import BenchmarkHarness

    dialog = tmp_path / "dialog_babi_task1_sample.txt"
    dialog.write_text(
        "USR|hello\n"
        "SYS|hello what can I help you with today\n"
        "USR|italian\n"
        "SYS|what city should I search in\n"
        "USR|rome\n"
        "SYS|what price range do you want\n"
        "USR|cheap\n"
        "SYS|api_call italian rome cheap\n",
        encoding="utf-8",
    )
    report = BenchmarkHarness().run_external_dialog_smoke(tmp_path, task_id=1)
    assert report.summary()["passed_count"] == report.summary()["total"]
    assert report.summary()["false_support_rate_bp"] == 0
