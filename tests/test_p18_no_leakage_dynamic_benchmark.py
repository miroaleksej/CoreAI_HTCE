from htce_origin.evaluation.no_leakage import (
    DynamicTaskFamily,
    DynamicTaskGenerator,
    HiddenCriteriaCommitment,
    NoAnswerLeakageContract,
    P18NoLeakageProtocol,
)


def test_p18_public_task_card_contains_no_gold_answer():
    card, gold = DynamicTaskGenerator().latest_state("seed-alpha")
    public = card.public_payload()
    assert public["contains_gold_answer"] == 0
    assert public["contains_hidden_criteria"] == 0
    assert public["contains_template_answer"] == 0
    assert gold.gold_answer not in public.values()
    assert "gold_answer" not in public
    assert "acceptable_answers" not in public
    assert "hidden_criteria" not in public


def test_p18_hidden_criteria_commitment_is_pre_execution_hash_only():
    card, gold = DynamicTaskGenerator().deduction("seed-beta")
    commitment = HiddenCriteriaCommitment.create(card, gold)
    payload = commitment.as_payload()
    assert payload["committed_before_execution"] == 1
    assert payload["gold_commitment_hash"] == gold.commitment_hash()
    assert gold.gold_answer not in str(payload)
    assert len(payload["commitment_hash"]) == 64


def test_p18_counterfactual_rewrites_change_public_task_and_gold_commitment():
    generator = DynamicTaskGenerator()
    original_card, original_gold = generator.card_pair(DynamicTaskFamily.LATEST_STATE, "seed-gamma", counterfactual_index=0)
    rewrite_card, rewrite_gold = generator.card_pair(DynamicTaskFamily.LATEST_STATE, "seed-gamma", counterfactual_index=3)
    assert original_card.public_hash() != rewrite_card.public_hash()
    assert original_gold.commitment_hash() != rewrite_gold.commitment_hash()
    assert original_gold.gold_answer != rewrite_gold.gold_answer


def test_p18_no_answer_leakage_contract_passes():
    contract = NoAnswerLeakageContract()
    assert contract.passed() is True
    assert contract.as_payload()["engine_receives_gold_answer"] == 0
    assert contract.as_payload()["engine_receives_private_goldset"] == 0


def test_p18_protocol_report_has_no_visible_answer_keys_and_trace_hashes():
    report = P18NoLeakageProtocol(seed="seed-delta").run()
    assert report.passed is True
    assert report.total_count >= 10
    assert report.false_support_count == 0
    assert report.answer_key_visible_count == 0
    assert report.no_answer_leakage_contract.passed() is True
    assert all(row.answer_key_visible_to_engine == 0 for row in report.rows)
    assert all(row.hidden_gold_commitment_hash for row in report.rows)
    assert all(row.pre_execution_commitment_hash for row in report.rows)
    assert all(row.trace_hash for row in report.rows)
    assert all(item.passed for item in report.counterfactual_tests)
