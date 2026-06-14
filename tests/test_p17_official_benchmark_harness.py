from htce_origin.evaluation.official_harness import P17OfficialBenchmarkHarness, P17SuiteKind, build_p17_official_specs


def test_p17_specs_cover_required_suites_and_tasks():
    specs = build_p17_official_specs()
    ids = {spec.task_id for spec in specs}
    suites = {spec.suite for spec in specs}
    assert all(f"babi_qa_{i}" in ids for i in range(1, 21))
    assert all(f"dialog_babi_{i}" in ids for i in range(1, 7))
    assert all(f"modified_dialog_babi_{i}" in ids for i in range(1, 7))
    assert all(f"permuted_dialog_babi_{i}" in ids for i in range(1, 7))
    assert {"long_memory_10000", "long_memory_50000", "long_memory_100000"}.issubset(ids)
    assert "contradiction_retraction_smoke" in ids
    assert "arc_style_mini_symbolic" in ids
    assert "closed_loop_abstract_env" in ids
    assert P17SuiteKind.BABI_20 in suites
    assert P17SuiteKind.DIALOG_BABI_6 in suites
    assert P17SuiteKind.PERMUTED_DIALOG_BABI in suites


def test_p17_release_smoke_matrix_has_required_columns_and_no_leakage():
    report = P17OfficialBenchmarkHarness().run_release_smoke_matrix(long_memory_events=10000, closed_loop_steps=3)
    assert report.passed is True
    assert report.total_count >= 8
    for row in report.rows:
        payload = row.as_payload()
        for key in ("task", "required_capability", "htce_modules_used", "answer", "evidence_path", "refusal_correctness", "trace_hash"):
            assert key in payload
        assert row.answer_key_visible_to_engine == 0
        assert row.false_support == 0
        assert row.trace_hash
        assert row.evidence_path


def test_p17_long_memory_profiles_are_available():
    harness = P17OfficialBenchmarkHarness()
    for count in (10000, 50000, 100000):
        row = harness.run_long_memory_profile(count)
        assert row.task == f"long_memory_{count}"
        assert row.passed == 1
        assert row.false_support == 0
        assert "generated_events" in row.evidence_path[0]


def test_p17_external_specs_do_not_bundle_dataset_rows():
    payload = P17OfficialBenchmarkHarness().spec_payload()
    assert payload["schema_version"] == "htce-p17-official-benchmark-spec-v1"
    assert payload["spec_count"] >= 44
    assert all("story" not in spec and "expected" not in spec for spec in payload["specs"])
