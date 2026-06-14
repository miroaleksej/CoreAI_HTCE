"""P14 formal invariant contract tests.

Runtime invariants are asserted directly.  Structural invariants are delegated to
``scripts/00_gates/check_invariants.py``, which performs static AST checks and writes a
machine-readable verification report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from htce_origin.body.layers import L123Body
from htce_origin.body.runtime import HTCERuntime, RuntimeConfig, RuntimeRequest
from htce_origin.sensory.l1_encoder import RawSensorPacket
from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy
from htce_origin.kernel.core import EntityId, EvidenceId, FactFrame, RelationId, fact_delta
from htce_origin.kernel.q16 import DEFAULT_MODULUS, Q256_MODULUS, q_vector_sub
from htce_origin.kernel.serialization import SerializationError, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_invariants_json_is_machine_readable_and_complete():
    spec = json.loads((ROOT / "invariants.json").read_text(encoding="utf-8"))
    ids = [item["id"] for item in spec["invariants"]]

    assert spec["schema_version"] == "htce-invariants-v1"
    assert spec["modulus"] == "2^256"
    assert ids == [f"I{i}" for i in range(1, 17)]
    for item in spec["invariants"]:
        assert item["formula"]
        assert item["checker"].startswith("check_i")
        assert item["target_module"].startswith("htce_origin")


def test_i1_q256_default_and_bounded_state_coordinates():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.run_closed_loop_simulation(steps=3)

    assert DEFAULT_MODULUS == Q256_MODULUS
    assert runtime.body.modulus == Q256_MODULUS
    for vector in (runtime.body.l1.vector, runtime.body.l2.vector, runtime.body.l3.vector, runtime.body.l2_clean_vector()):
        assert all(0 <= value < Q256_MODULUS for value in vector)


def test_i2_protected_serialization_rejects_float_values():
    try:
        payload = {"value": float("3.14")}
        canonical_json_bytes(payload)
    except SerializationError:
        return
    raise AssertionError("protected serialization accepted a float")


def test_i3_l1_observation_updates_l1_only():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    before = runtime.health()
    response = runtime.observe_l1_packet(
        RawSensorPacket(
            modality="vision",
            samples=(1, 2, 3, 4),
            sample_min=0,
            sample_max=(1 << 16) - 1,
            evidence_id="ev_l1",
        ),
        source="test",
    )
    after = runtime.health()

    assert response.decision.trace_id
    assert after["l1_clock"] > before["l1_clock"]
    assert after["l2_clock"] == before["l2_clock"]
    assert after["l3_clock"] == before["l3_clock"]
    assert after["latest_fact_count"] == before["latest_fact_count"]


def test_i5_l2_clean_equals_raw_minus_tag_accumulator():
    body = L123Body(dimension=4, modulus=DEFAULT_MODULUS)
    fact = FactFrame(EntityId("alpha"), RelationId("located_in"), EntityId("lab"), EvidenceId("ev_i5"), confidence_bp=9000)
    body.commit_l2_fact(fact_delta(fact, dimension=4, modulus=DEFAULT_MODULUS))

    expected_clean = q_vector_sub(body.l2.vector, body.l2_episode_tag_accumulator, body.modulus)
    assert body.l2_clean_vector() == expected_clean


def test_i6_l2_supersession_keeps_one_active_residual():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT mary located_in office EVID ev_i6a", source="test"))
    clean_after_first = runtime.body.l2_clean_vector()
    runtime.tick(RuntimeRequest("FACT mary located_in garden EVID ev_i6b", source="test"))
    clean_after_second = runtime.body.l2_clean_vector()

    history = runtime.memory.history("mary", "located_in")
    assert any(record.status.value == "superseded" for record in history)
    assert len(runtime.body.l2_active_contributions) == 1
    assert clean_after_second != clean_after_first


def test_i9_planner_blocks_real_actions_runtime():
    policy = SimulationHabitatPolicy()
    gate = HabitatGateInput(action_class="real")
    decision = policy.evaluate(gate)

    assert policy.allowed_real_action(gate) is False
    assert decision.allowed is False
    assert decision.allowed_real_action is False


def test_i12_export_restore_preserves_trace_verification():
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT alice located_in office EVID ev_i12", source="test"))
    restored = HTCERuntime.restore_state(runtime.export_state())

    assert restored.trace.verify() is True
    assert len(restored.trace.events) == len(runtime.trace.events)


def test_p14_check_invariants_script_passes_and_writes_report():
    completed = subprocess.run(
        [sys.executable, "scripts/00_gates/check_invariants.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report_path = ROOT / "artifacts" / "invariants_verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["all_passed"] is True
    assert report["passed_count"] == report["total_count"] == 16
