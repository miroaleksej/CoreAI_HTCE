import json
from pathlib import Path
import pytest

from htce_origin.governance.evidence import (
    ClaimSupportReport,
    EvidenceAnchor,
    EvidenceError,
    EvidenceRelation,
    EvidenceWeigher,
    HashChain,
    PreExecutionCommitment,
    SourceManifest,
    calibrate_source_thresholds,
    TraceEvent,
    TraceVerifier,
    canonical_json,
    create_pre_execution_commitment,
    sha256_hex as evidence_sha256_hex,
    tamper_event,
    verify_pre_execution_commitment,
)


def test_canonical_json_is_deterministic_and_sorted():
    a = {"b": 2, "a": {"z": 1, "m": [3, 2, 1]}}
    b = {"a": {"m": [3, 2, 1], "z": 1}, "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a).startswith('{"a"')


def test_canonical_json_rejects_float_runtime_payload():
    with pytest.raises(EvidenceError):
        canonical_json({"confidence": 0.5})


def test_hash_chain_append_and_verify():
    chain = HashChain()
    ev1 = chain.append("answer", {"decision": "answer", "trace_required": True})
    ev2 = chain.append("refusal", {"decision": "refuse", "reason": "missing evidence"})
    assert ev1.previous_hash == "GENESIS"
    assert ev2.previous_hash == ev1.event_hash()
    assert chain.head == ev2.event_hash()
    assert chain.verify()


def test_snapshot_verifies():
    chain = HashChain()
    chain.append("answer", {"decision": "answer", "trace_required": True})
    chain.append("policy_decision", {"decision": "refuse", "reason": "unsupported claim"})
    snapshot = chain.snapshot()
    assert snapshot.count == 2
    assert snapshot.head == chain.head
    assert TraceVerifier.verify_snapshot(snapshot)


def test_tampered_event_breaks_chain_verification():
    chain = HashChain()
    chain.append("answer", {"decision": "answer"})
    second = chain.append("refusal", {"decision": "refuse", "reason": "missing evidence"})
    events = list(chain.events)
    events[0] = tamper_event(events[0], {"decision": "tampered"})
    # The second event still points to the original first-event hash, so tamper is detected.
    assert not TraceVerifier.verify(events)
    assert second.previous_hash != events[0].event_hash()


def test_reconstruct_hash_chain_rejects_wrong_previous_hash():
    ev = TraceEvent("answer", {"decision": "answer"}, previous_hash="BAD", sequence=0)
    with pytest.raises(EvidenceError):
        HashChain([ev])

from htce_origin.governance.snapshot import SnapshotError, SnapshotStore, corrupt_trace_event, verify_release_manifest_self_hash
from htce_origin.kernel.serialization import SerializationError, canonical_json_str, sha256_hex


def test_snapshot_release_snapshot_hash_verifies():
    chain = HashChain()
    chain.append("answer", {"decision": "answer", "trace_required": True})
    state = {"latest_state": {"mary.location": "garden"}, "release_line": "final_math"}
    bundle = SnapshotStore().create_bundle(state_payload=state, trace=chain, version="0.1.0-final_math")
    assert SnapshotStore().verify_bundle(bundle)
    assert bundle.manifest.state_hash == sha256_hex(state)
    assert bundle.manifest.trace_head == chain.head


def test_snapshot_release_restore_preserves_latest_state():
    chain = HashChain()
    chain.append("commit", {"key": "mary.location", "value": "garden"})
    state = {"latest_state": {"mary.location": "garden"}}
    store = SnapshotStore()
    bundle = store.create_bundle(state_payload=state, trace=chain, version="0.1.0-final_math")
    restored = store.restore(bundle, expected_version="0.1.0-final_math")
    assert restored["latest_state"]["mary.location"] == "garden"


def test_snapshot_release_corrupted_trace_rejected():
    chain = HashChain()
    chain.append("commit", {"key": "mary.location", "value": "garden"})
    bundle = SnapshotStore().create_bundle(
        state_payload={"latest_state": {"mary.location": "garden"}},
        trace=chain,
        version="0.1.0-final_math",
    )
    corrupted = corrupt_trace_event(bundle)
    assert not SnapshotStore().verify_bundle(corrupted)
    with pytest.raises(SnapshotError):
        SnapshotStore().restore(corrupted)


def test_snapshot_release_release_bundle_contains_manifest_report_hashes():
    chain = HashChain()
    chain.append("release", {"decision": "bundle"})
    store = SnapshotStore()
    bundle = store.create_bundle(state_payload={"latest_state": {}}, trace=chain, version="0.1.0-final_math")
    release = store.create_release_manifest(
        release_id="final_math-release",
        version="0.1.0-final_math",
        bundle=bundle,
        reports={"ACCEPTANCE_REPORT.md": "report text"},
        artifacts={"archive.zip": "abc123"},
    )
    payload = release.as_payload()
    assert payload["bundle_hash"] == bundle.bundle_hash
    assert payload["manifest_hash"] == sha256_hex(bundle.manifest.as_payload())
    assert "ACCEPTANCE_REPORT.md" in payload["report_hashes"]
    assert payload["artifact_hashes"]["archive.zip"] == "abc123"


def test_snapshot_release_canonical_serialization_rejects_float():
    with pytest.raises(SerializationError):
        canonical_json_str({"runtime_float": 0.5})



def _public_card():
    return {
        "prompt": "Help a user resolve a calendar contradiction without making unsupported claims.",
        "context": "The public card lists only observable messages and allowed tools.",
        "requirements": ["use memory", "use uncertainty", "do not invent evidence"],
        "difficulty": "medium",
    }


def _hidden_criteria():
    return {
        "expected_behavior": "ask_clarification_when_evidence_is_missing",
        "rubric_items": ["memory_recall", "uncertainty_calibration", "false_support_blocked"],
        "answer_key": "not visible to engine",
    }


def test_pre_execution_commitment_hashes_hidden_criteria_without_exposing_text():
    public_card = _public_card()
    hidden_criteria = _hidden_criteria()
    commitment = create_pre_execution_commitment(
        scenario_id="scenario-001",
        public_card=public_card,
        hidden_criteria=hidden_criteria,
        previous_trace_hash="GENESIS",
        experience_payload={"prior_run_count": 2},
        created_at="2026-06-14T12:00:00+00:00",
    )
    payload = commitment.as_payload()
    payload_json = canonical_json(payload)
    assert commitment.public_card_hash == evidence_sha256_hex(public_card)
    assert commitment.hidden_criteria_hash == evidence_sha256_hex(hidden_criteria)
    assert commitment.experience_hash == evidence_sha256_hex({"prior_run_count": 2})
    assert "not visible to engine" not in payload_json
    assert "ask_clarification_when_evidence_is_missing" not in payload_json
    assert payload["no_answer_leakage_contract"]["no_answer_leakage_pass"] == 1
    assert verify_pre_execution_commitment(
        commitment,
        public_card=public_card,
        hidden_criteria=hidden_criteria,
        experience_payload={"prior_run_count": 2},
    )


def test_public_card_with_answer_key_is_rejected_before_commitment():
    public_card = dict(_public_card())
    public_card["answer_key"] = "this must never be visible"
    with pytest.raises(EvidenceError):
        create_pre_execution_commitment(
            scenario_id="leaky-scenario",
            public_card=public_card,
            hidden_criteria=_hidden_criteria(),
            previous_trace_hash="GENESIS",
            created_at="2026-06-14T12:00:00+00:00",
        )


def test_hash_chain_commits_hidden_hash_before_scenario_execution():
    chain = HashChain()
    event, commitment = chain.append_pre_execution_commitment(
        scenario_id="scenario-002",
        public_card=_public_card(),
        hidden_criteria=_hidden_criteria(),
        experience_payload={"replay_hash": "abc"},
        created_at="2026-06-14T12:00:00+00:00",
    )
    assert event.event_type == "pre_execution_commitment"
    execution = chain.append_scenario_execution(
        run_commitment_hash=commitment.run_commitment_hash,
        scenario_id="scenario-002",
        public_output_hash=evidence_sha256_hex({"answer": "I need more evidence."}),
        decision="ask_clarification",
        metrics={"false_support": 0},
    )
    assert execution.previous_hash == event.event_hash()
    assert chain.verify()
    assert TraceVerifier.verify_hidden_commitments(chain.events)


def test_scenario_execution_without_prior_hidden_commitment_is_invalid():
    chain = HashChain()
    chain.append_scenario_execution(
        run_commitment_hash="0" * 64,
        scenario_id="scenario-003",
        public_output_hash=evidence_sha256_hex({"answer": "unsupported"}),
        decision="answer",
    )
    assert chain.verify()
    assert not TraceVerifier.verify_hidden_commitments(chain.events)


def test_commitment_depends_on_previous_trace_fragment():
    public_card = _public_card()
    hidden_criteria = _hidden_criteria()
    c1 = create_pre_execution_commitment(
        scenario_id="scenario-004",
        public_card=public_card,
        hidden_criteria=hidden_criteria,
        previous_trace_hash="GENESIS",
        created_at="2026-06-14T12:00:00+00:00",
    )
    c2 = create_pre_execution_commitment(
        scenario_id="scenario-004",
        public_card=public_card,
        hidden_criteria=hidden_criteria,
        previous_trace_hash="1" * 64,
        created_at="2026-06-14T12:00:00+00:00",
    )
    assert c1.trace_fragment_hash != c2.trace_fragment_hash
    assert c1.run_commitment_hash != c2.run_commitment_hash


def test_tampered_hidden_criteria_fails_commitment_verification():
    commitment = create_pre_execution_commitment(
        scenario_id="scenario-005",
        public_card=_public_card(),
        hidden_criteria=_hidden_criteria(),
        previous_trace_hash="GENESIS",
        created_at="2026-06-14T12:00:00+00:00",
    )
    changed_hidden = dict(_hidden_criteria())
    changed_hidden["answer_key"] = "changed after execution"
    assert not verify_pre_execution_commitment(
        commitment,
        public_card=_public_card(),
        hidden_criteria=changed_hidden,
    )


def test_source_manifest_weighting_downweights_weak_retracted_and_contradictory_sources():
    primary = SourceManifest(
        source_id="primary",
        uri="https://example.org/primary",
        title="Primary source",
        source_type="primary_paper",
        base_quality_bp=7200,
        primary_source=1,
        independent_replication_count=2,
    )
    weak = SourceManifest(
        source_id="weak",
        uri="https://example.org/weak",
        title="Weak web source",
        source_type="blog",
        base_quality_bp=7200,
        weak_source=1,
    )
    retracted = SourceManifest(
        source_id="retracted",
        uri="https://example.org/retracted",
        title="Retracted source",
        source_type="primary_paper",
        base_quality_bp=9000,
        primary_source=1,
        retracted=1,
    )
    contradictory = SourceManifest(
        source_id="contradictory",
        uri="https://example.org/contradictory",
        title="Contradictory source",
        source_type="replication_note",
        base_quality_bp=7200,
        contradiction_markers=2,
    )
    assert primary.source_weight_bp > weak.source_weight_bp
    assert retracted.source_weight_bp == 0
    assert contradictory.source_weight_bp < primary.source_weight_bp


def test_evidence_anchor_is_evidence_only_and_cannot_commit_settled_fact():
    source = SourceManifest(
        source_id="source",
        uri="https://example.org/source",
        title="Source",
        source_type="primary_paper",
        base_quality_bp=7000,
    )
    anchor = EvidenceAnchor(
        anchor_id="anchor-1",
        claim_id="claim-1",
        source=source,
        relation=EvidenceRelation.SUPPORT,
        quote_digest=evidence_sha256_hex({"quote": "support"}),
    )
    assert anchor.as_payload()["evidence_only_boundary"] == 1
    assert anchor.as_payload()["settled_fact_commit"] == 0
    with pytest.raises(EvidenceError):
        EvidenceAnchor(
            anchor_id="bad-anchor",
            claim_id="claim-1",
            source=source,
            relation=EvidenceRelation.SUPPORT,
            quote_digest=evidence_sha256_hex({"quote": "bad"}),
            settled_fact_commit=1,
        )


def test_claim_support_gate_allows_strong_support_and_blocks_retracted_or_contradicted_claims():
    primary = SourceManifest(
        source_id="primary",
        uri="https://example.org/primary",
        title="Primary source",
        source_type="primary_paper",
        base_quality_bp=7200,
        primary_source=1,
        independent_replication_count=1,
    )
    retracted = SourceManifest(
        source_id="retracted",
        uri="https://example.org/retracted",
        title="Retracted source",
        source_type="primary_paper",
        base_quality_bp=9000,
        primary_source=1,
        retracted=1,
    )
    contradictor = SourceManifest(
        source_id="contradictor",
        uri="https://example.org/contradictor",
        title="Contradictor",
        source_type="replication_note",
        base_quality_bp=7000,
        primary_source=1,
    )
    anchors = (
        EvidenceAnchor("a1", "claim-good", primary, EvidenceRelation.SUPPORT, evidence_sha256_hex("q1")),
        EvidenceAnchor("a2", "claim-retracted", retracted, EvidenceRelation.SUPPORT, evidence_sha256_hex("q2")),
        EvidenceAnchor("a3", "claim-contradicted", primary, EvidenceRelation.SUPPORT, evidence_sha256_hex("q3")),
        EvidenceAnchor("a4", "claim-contradicted", contradictor, EvidenceRelation.CONTRADICT, evidence_sha256_hex("q4")),
    )
    weigher = EvidenceWeigher(support_threshold_bp=6000, contradiction_threshold_bp=4000)
    good = weigher.score_claim("claim-good", anchors)
    retracted_report = weigher.score_claim("claim-retracted", anchors)
    contradicted = weigher.score_claim("claim-contradicted", anchors)
    assert isinstance(good, ClaimSupportReport)
    assert good.claim_allowed == 1
    assert retracted_report.claim_allowed == 0
    assert retracted_report.source_retracted == 1
    assert contradicted.claim_allowed == 0
    assert contradicted.contradiction_bp >= 4000


def test_hash_chain_appends_claim_support_report_as_evidence_only_trace_event():
    source = SourceManifest(
        source_id="primary",
        uri="https://example.org/primary",
        title="Primary source",
        source_type="primary_paper",
        base_quality_bp=7200,
        primary_source=1,
    )
    anchor = EvidenceAnchor("a1", "claim-good", source, EvidenceRelation.SUPPORT, evidence_sha256_hex("quote"))
    report = EvidenceWeigher().score_claim("claim-good", (anchor,))
    chain = HashChain()
    event = chain.append_claim_support_report(report)
    assert event.event_type == "claim_support_report"
    assert event.payload["web_anchor_equals_settled_fact"] == 0
    assert chain.verify()


def test_release_manifest_self_hash_protocol_verifies_current_manifest():
    import json
    from pathlib import Path

    manifest_path = Path(__file__).resolve().parents[1] / "RELEASE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_release_manifest_self_hash(payload)



def test_source_threshold_calibration_passes_primary_and_replication_blocks_weak_retracted_contradiction():
    primary = SourceManifest(
        source_id="primary",
        uri="https://example.org/primary",
        title="Primary source",
        source_type="primary_paper",
        base_quality_bp=7200,
        primary_source=1,
        independent_replication_count=1,
    )
    weak = SourceManifest(
        source_id="weak-blog",
        uri="https://example.org/blog",
        title="Weak blog",
        source_type="blog",
        base_quality_bp=4200,
        weak_source=1,
    )
    retracted = SourceManifest(
        source_id="retracted",
        uri="https://example.org/retracted",
        title="Retracted paper",
        source_type="primary_paper",
        base_quality_bp=7600,
        primary_source=1,
        retracted=1,
    )
    contradiction = SourceManifest(
        source_id="contradiction",
        uri="https://example.org/contradiction",
        title="Contradictory note",
        source_type="replication_note",
        base_quality_bp=6200,
        independent_replication_count=1,
        contradiction_markers=1,
    )
    calibration = calibrate_source_thresholds(
        strong_primary_support=(primary,),
        weak_blog_support=(weak,),
        retracted_support=(retracted,),
        contradicted_support=(contradiction,),
    )
    assert calibration.passed
    assert calibration.primary_replicated_support_passes == 1
    assert calibration.single_weak_source_fails == 1
    assert calibration.retracted_source_blocks == 1
    assert calibration.contradiction_source_blocks == 1
    weigher = EvidenceWeigher.from_calibration(calibration)
    strong_anchor = EvidenceAnchor("strong", "claim", primary, EvidenceRelation.SUPPORT, evidence_sha256_hex("strong"))
    weak_anchor = EvidenceAnchor("weak", "weak_claim", weak, EvidenceRelation.SUPPORT, evidence_sha256_hex("weak"))
    retracted_anchor = EvidenceAnchor("retracted", "retracted_claim", retracted, EvidenceRelation.SUPPORT, evidence_sha256_hex("retracted"))
    contradiction_anchor = EvidenceAnchor("contradict", "contradicted_claim", contradiction, EvidenceRelation.CONTRADICT, evidence_sha256_hex("contradict"))
    assert weigher.score_claim("claim", (strong_anchor,)).claim_allowed == 1
    assert weigher.score_claim("weak_claim", (weak_anchor,)).claim_allowed == 0
    assert weigher.score_claim("retracted_claim", (retracted_anchor,)).claim_allowed == 0
    assert weigher.score_claim("contradicted_claim", (contradiction_anchor,)).claim_allowed == 0


def test_release_manifest_and_hashes_match_real_release_file_set():
    from htce_origin.kernel.serialization import release_manifest_self_hash

    root = Path(__file__).resolve().parents[1]
    ignored_parts = {"__pycache__", ".pytest_cache", "__MACOSX", "artifacts"}

    def included(path: Path) -> bool:
        rel = path.relative_to(root).as_posix()
        if any(part in ignored_parts for part in path.parts):
            return False
        if path.name == ".DS_Store" or path.suffix == ".pyc":
            return False
        return path.is_file()

    real_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if included(path))
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_files = sorted(manifest["file_hashes_sha256"].keys())
    assert manifest_files == real_files
    assert manifest["file_hashes_sha256"]["RELEASE_MANIFEST.json"] == release_manifest_self_hash(manifest)

    hash_lines = [line.strip().split("  ", 1)[1] for line in (root / "HASHES.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sorted(hash_lines) == real_files
    assert not any("__pycache__" in name or ".pytest_cache" in name or name.endswith(".pyc") for name in manifest_files)
