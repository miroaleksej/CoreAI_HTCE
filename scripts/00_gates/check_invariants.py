#!/usr/bin/env python3
"""Machine-readable invariant checker for HTCE-Origin P14.

The checker loads ``invariants.json`` and verifies each invariant either by
runtime assertions or by static AST boundary checks.  It writes a canonical
JSON-compatible report to ``artifacts/invariants_verification_report.json``.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.body.layers import L123Body
from htce_origin.body.runtime import HTCERuntime, RuntimeConfig, RuntimeRequest
from htce_origin.sensory.l1_encoder import RawSensorPacket
from htce_origin.control.planner import HabitatGateInput, SimulationHabitatPolicy
from htce_origin.cognition.world import Q256WorldAction, Q256WorldModel
from htce_origin.cognition.l3_promotion import L3RuleCandidate
from htce_origin.governance.proof import Statement
from htce_origin.kernel.config import RuntimeConfig as KernelRuntimeConfig
from htce_origin.kernel.core import EntityId, EvidenceId, FactFrame, RelationId, TorusVector, fact_delta
from htce_origin.kernel.q16 import DEFAULT_MODULUS, Q256_MODULUS, q16_property_stress, q_vector_add, q_vector_sub
from htce_origin.kernel.uint256 import BOARD_MEASUREMENT_STATUS, HARDWARE_CLAIM_STATUS, generate_uint256_hardware_manifest, verify_uint256_arithmetic_model
from htce_origin.kernel.serialization import SerializationError, canonical_json_bytes

ARTIFACTS = ROOT / "artifacts"


class InvariantCheckError(RuntimeError):
    pass


def _module_path(module_rel: str) -> Path:
    return ROOT / (module_rel.replace(".", "/") + ".py")


def _source(module_rel: str) -> str:
    path = _module_path(module_rel)
    if not path.exists():
        raise InvariantCheckError(f"module source missing: {module_rel}")
    return path.read_text(encoding="utf-8")


def _tree(module_rel: str) -> ast.Module:
    return ast.parse(_source(module_rel))


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _class_def(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _attr_calls(node: ast.AST) -> set[str]:
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            calls.add(child.func.attr)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            calls.add(child.func.id)
    return calls


def _contains_call(node: ast.AST, call_name: str) -> bool:
    return call_name in _attr_calls(node)


def _assert_all_phases_bounded(values: object, modulus: int) -> bool:
    if isinstance(values, int):
        return 0 <= values < modulus
    if isinstance(values, (list, tuple)):
        return all(_assert_all_phases_bounded(item, modulus) for item in values)
    return True


def check_i1() -> tuple[bool, dict[str, object]]:
    stress = q16_property_stress(sample_count=2048, modulus=DEFAULT_MODULUS)
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.run_closed_loop_simulation(steps=3, verify_trace_each_step=True)
    body = runtime.body
    bounded = _assert_all_phases_bounded((body.l1.vector, body.l2.vector, body.l3.vector, body.l2_clean_vector()), body.modulus)
    return stress.passed and bounded and body.modulus == Q256_MODULUS, {
        "q256_modulus": body.modulus == Q256_MODULUS,
        "stress_passed": stress.passed,
        "stress_total_failures": stress.total_failures,
        "state_phases_bounded": bounded,
    }


def check_i2() -> tuple[bool, dict[str, object]]:
    try:
        canonical_json_bytes({"forbidden_float": float("3.14")})
    except SerializationError:
        return True, {"float_rejected": True}
    return False, {"float_rejected": False}


def check_i3() -> tuple[bool, dict[str, object]]:
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    before = runtime.health()
    response = runtime.observe_l1_packet(
        RawSensorPacket(
            modality="invariant_l1",
            samples=(1, 2, 3, 4),
            sample_min=0,
            sample_max=(1 << 16) - 1,
            evidence_id="inv_l1",
        ),
        source="invariant_checker",
    )
    after = runtime.health()
    runtime_pass = (
        response.decision.trace_id is not None
        and after["l1_clock"] != before["l1_clock"]
        and after["l2_clock"] == before["l2_clock"]
        and after["l3_clock"] == before["l3_clock"]
        and after["latest_fact_count"] == before["latest_fact_count"]
    )
    tree = _tree("htce_origin.body.runtime")
    forbidden = {"commit_l2_fact", "commit_l3_semantic_state", "commit_fact", "commit_negation"}
    static_ok = True
    details: dict[str, object] = {"runtime_l1_only": runtime_pass}
    for name in ("observe_l1_packet", "_commit_l1_observation", "observe_simulated"):
        node = _function_def(tree, name)
        if node is None:
            static_ok = False
            details[f"{name}_present"] = False
            continue
        calls = _attr_calls(node)
        bad = sorted(forbidden.intersection(calls))
        details[f"{name}_forbidden_calls"] = bad
        if bad:
            static_ok = False
    return runtime_pass and static_ok, details


def check_i4() -> tuple[bool, dict[str, object]]:
    tree = _tree("htce_origin.body.runtime")
    forbidden = {"_commit_fact", "_commit_negation", "commit_l2_fact", "commit_l3_semantic_state", "commit"}
    details: dict[str, object] = {}
    ok = True
    for name in ("observe_l1_packet", "_commit_l1_observation", "observe_simulated"):
        node = _function_def(tree, name)
        if node is None:
            ok = False
            details[f"{name}_present"] = False
            continue
        calls = _attr_calls(node)
        bad = sorted(forbidden.intersection(calls))
        details[f"{name}_fact_commit_calls"] = bad
        if bad:
            ok = False
    return ok, details


def check_i5() -> tuple[bool, dict[str, object]]:
    body = L123Body(dimension=4, modulus=DEFAULT_MODULUS)
    fact = FactFrame(EntityId("alpha"), RelationId("located_in"), EntityId("room"), EvidenceId("ev_i5"), confidence_bp=9000)
    body.commit_l2_fact(fact_delta(fact, dimension=4, modulus=DEFAULT_MODULUS))
    expected_clean = q_vector_sub(body.l2.vector, body.l2_episode_tag_accumulator, body.modulus)
    return body.l2_clean_vector() == expected_clean, {
        "clean_equals_raw_minus_tag_accumulator": body.l2_clean_vector() == expected_clean,
        "active_contribution_count": len(body.l2_active_contributions),
    }


def check_i6() -> tuple[bool, dict[str, object]]:
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT mary located_in office EVID ev_i6a", source="invariant_checker"))
    first_clean = runtime.body.l2_clean_vector()
    runtime.tick(RuntimeRequest("FACT mary located_in lab EVID ev_i6b", source="invariant_checker"))
    second_clean = runtime.body.l2_clean_vector()
    history = runtime.memory.history("mary", "located_in")
    has_superseded = any(record.status.value == "superseded" for record in history)
    active_count = len(runtime.body.l2_active_contributions)
    clean_changed = first_clean != second_clean
    return has_superseded and active_count == 1 and clean_changed, {
        "has_superseded_record": has_superseded,
        "active_working_contribution_count": active_count,
        "clean_state_changed_to_new_residual": clean_changed,
    }


def check_i7() -> tuple[bool, dict[str, object]]:
    learning_tree = _tree("htce_origin.cognition.learning")
    runtime_tree = _tree("htce_origin.body.runtime")
    bad_calls = []
    for cls_name in ("L3DeductiveEngine", "ToroidalSleepConsolidator"):
        cls = _class_def(learning_tree, cls_name)
        if cls is None:
            bad_calls.append(f"missing:{cls_name}")
            continue
        calls = _attr_calls(cls)
        for bad in ("authorize_query", "evaluate", "commit_l2_fact", "allowed_real_action"):
            if bad in calls:
                bad_calls.append(f"{cls_name}.{bad}")
    answer_fn = _function_def(runtime_tree, "_answer_query")
    answer_has_authorize = bool(answer_fn and _contains_call(answer_fn, "authorize_query"))
    return not bad_calls and answer_has_authorize, {
        "l3_forbidden_authority_calls": bad_calls,
        "answer_path_has_authorize_query": answer_has_authorize,
    }


def check_i8() -> tuple[bool, dict[str, object]]:
    tree = _tree("htce_origin.cognition.world")
    forbidden = {"commit", "commit_l2_fact", "commit_l3_semantic_state", "commit_fact", "commit_negation"}
    bad: list[str] = []
    for cls_name in ("Q256WorldModel", "AdaptiveQ256Dynamics"):
        cls = _class_def(tree, cls_name)
        if cls is None:
            bad.append(f"missing:{cls_name}")
            continue
        for node in ast.walk(cls):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "commit" in node.name:
                bad.append(f"{cls_name}.{node.name}")
        calls = _attr_calls(cls)
        for name in sorted(forbidden.intersection(calls)):
            bad.append(f"{cls_name}->{name}")
    return not bad, {"world_forbidden_commit_paths": bad}


def check_i9() -> tuple[bool, dict[str, object]]:
    policy = SimulationHabitatPolicy()
    gate = HabitatGateInput(action_class="real")
    decision = policy.evaluate(gate)
    allowed_real = policy.allowed_real_action(gate)
    return decision.allowed is False and allowed_real is False, {
        "evaluate_real_allowed": decision.allowed,
        "allowed_real_action": allowed_real,
        "decision_reason": decision.reason,
    }


def check_i10() -> tuple[bool, dict[str, object]]:
    tree = _tree("htce_origin.body.runtime")
    fn = _function_def(tree, "_answer_query")
    if fn is None:
        return False, {"_answer_query_present": False}
    calls = _attr_calls(fn)
    static_ok = {"policy.evaluate": "evaluate" in calls, "authorize_query": "authorize_query" in calls, "prove": ("prove" in calls or "prove_where" in calls), "trace_append": "append" in calls}
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT alice located_in office EVID ev_i10_fact", source="invariant_checker"))
    response = runtime.tick(RuntimeRequest("QUERY alice located_in EVID ev_i10_query", source="invariant_checker"))
    auth = response.diagnostics.get("authorization", {}) if isinstance(response.diagnostics, dict) else {}
    runtime_ok = response.output.startswith("ANSWER:") and bool(auth) and auth.get("answer_allowed") is True and bool(response.decision.trace_id)
    return all(static_ok.values()) and runtime_ok, {"static": static_ok, "runtime_answer_gated": runtime_ok}


def check_i11() -> tuple[bool, dict[str, object]]:
    runtime = HTCERuntime(RuntimeConfig())
    before = len(runtime.trace.events)
    runtime.wake()
    after_wake = len(runtime.trace.events)
    runtime.tick(RuntimeRequest("FACT bob located_in room EVID ev_i11", source="invariant_checker"))
    after_tick = len(runtime.trace.events)
    return after_wake > before and after_tick > after_wake, {"before": before, "after_wake": after_wake, "after_tick": after_tick}


def check_i12() -> tuple[bool, dict[str, object]]:
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    runtime.tick(RuntimeRequest("FACT carol located_in archive EVID ev_i12", source="invariant_checker"))
    exported = runtime.export_state()
    restored = HTCERuntime.restore_state(exported)
    verified = restored.trace.verify()
    return verified, {"restored_trace_verified": verified, "trace_event_count": len(restored.trace.events)}


def check_i13() -> tuple[bool, dict[str, object]]:
    model = Q256WorldModel(dimension=1, modulus=DEFAULT_MODULUS)
    state = TorusVector((0,), DEFAULT_MODULUS)
    target = TorusVector((8,), DEFAULT_MODULUS)
    near = Q256WorldAction("near", (8,), modulus=DEFAULT_MODULUS)
    far = Q256WorldAction("far", (DEFAULT_MODULUS // 2,), modulus=DEFAULT_MODULUS)
    selected = model.select_min_expected_free_energy_action(state, (far, near), target_state=target)
    world_tree = _tree("htce_origin.cognition.world")
    select_fn = _function_def(world_tree, "select_min_expected_free_energy_action")
    runtime_tree = _tree("htce_origin.body.runtime")
    loop_fn = _function_def(runtime_tree, "run_closed_loop_simulation")
    select_source = ast.get_source_segment(_source("htce_origin.cognition.world"), select_fn) if select_fn else ""
    loop_source = ast.get_source_segment(_source("htce_origin.body.runtime"), loop_fn) if loop_fn else ""
    static_ok = (
        selected.action_name == "near"
        and "expected_free_energy_raw" in (select_source or "")
        and "expected_free_energy_bp" not in (select_source or "")
        and "expected_free_energy_raw" in (loop_source or "")
    )
    return static_ok, {
        "selected_action": selected.action_name,
        "selector_uses_raw": "expected_free_energy_raw" in (select_source or ""),
        "selector_mentions_bp": "expected_free_energy_bp" in (select_source or ""),
        "closed_loop_records_raw": "expected_free_energy_raw" in (loop_source or ""),
    }


def check_i15() -> tuple[bool, dict[str, object]]:
    runtime = HTCERuntime(RuntimeConfig())
    runtime.wake()
    before = runtime.health()
    candidate = L3RuleCandidate(
        statement=Statement.atom("has_property", "cat", "fur"),
        support_count_raw=2,
        trace_ids=("l3_trace_a", "l3_trace_b"),
        source_rule_id="invariant_i15",
        l3_state_digest=runtime.body.l3.digest,
    )
    decision = runtime.promote_l3_candidate_rule(candidate, evidence_id="l3_i15", required_support_raw=2)
    after = runtime.health()
    runtime_ok = (
        decision.provisional_promoted
        and decision.may_answer is False
        and decision.may_commit_l2_fact is False
        and decision.may_execute_real_action is False
        and after["latest_fact_count"] == before["latest_fact_count"]
        and after["l2_clock"] == before["l2_clock"]
        and after["l3_clock"] == before["l3_clock"]
        and after["l3_provisional_rule_count"] == before.get("l3_provisional_rule_count", 0) + 1
        and runtime.theorem_layer.prove(candidate.statement).valid is False
    )
    runtime_tree = _tree("htce_origin.body.runtime")
    fn = _function_def(runtime_tree, "promote_l3_candidate_rule")
    calls = _attr_calls(fn) if fn is not None else set()
    forbidden = {"commit_l2_fact", "commit_l3_semantic_state", "promote_l3_rule", "_commit_fact", "_commit_negation", "run_closed_loop_simulation", "allowed_real_action"}
    bad = sorted(forbidden.intersection(calls))
    promotion_tree = _tree("htce_origin.cognition.l3_promotion")
    payload_source = _source("htce_origin.cognition.l3_promotion")
    static_ok = (
        fn is not None
        and not bad
        and "may_answer" in payload_source
        and "may_commit_l2_fact" in payload_source
        and "may_execute_real_action" in payload_source
    )
    return runtime_ok and static_ok, {
        "runtime_provisional_promoted": decision.provisional_promoted,
        "runtime_may_answer": decision.may_answer,
        "runtime_may_commit_l2_fact": decision.may_commit_l2_fact,
        "runtime_may_execute_real_action": decision.may_execute_real_action,
        "latest_fact_count_before": before["latest_fact_count"],
        "latest_fact_count_after": after["latest_fact_count"],
        "l2_clock_before": before["l2_clock"],
        "l2_clock_after": after["l2_clock"],
        "l3_clock_before": before["l3_clock"],
        "l3_clock_after": after["l3_clock"],
        "provisional_rule_count_after": after["l3_provisional_rule_count"],
        "theorem_proof_valid_after_association": runtime.theorem_layer.prove(candidate.statement).valid,
        "forbidden_runtime_calls": bad,
        "static_boundary_ok": static_ok,
    }


def check_i16() -> tuple[bool, dict[str, object]]:
    report = verify_uint256_arithmetic_model(sample_count=32)
    manifest = generate_uint256_hardware_manifest().as_payload()
    source = _source("htce_origin.kernel.uint256")
    ok = (
        report.passed
        and report.hardware_claim_status == HARDWARE_CLAIM_STATUS
        and report.board_measurement_status == BOARD_MEASUREMENT_STATUS
        and manifest["hardware_claim_status"] == HARDWARE_CLAIM_STATUS
        and manifest["board_measurement_status"] == BOARD_MEASUREMENT_STATUS
        and "general_256x256_multiplier_claim" in manifest["disallowed_operations"]
        and "board_measured_claim" in manifest["disallowed_operations"]
        and "UINT256_MASK" in source
        and "& UINT256_MASK" in source
    )
    return ok, {
        "hardware_claim_status": report.hardware_claim_status,
        "board_measurement_status": report.board_measurement_status,
        "failed_count": report.failed_count,
        "sample_count": report.sample_count,
        "manifest_disallowed_operations": manifest["disallowed_operations"],
        "explicit_mask_semantics_present": "& UINT256_MASK" in source,
    }


def check_i14() -> tuple[bool, dict[str, object]]:
    cfg = KernelRuntimeConfig()
    ok = DEFAULT_MODULUS == Q256_MODULUS and cfg.modulus == Q256_MODULUS
    return ok, {"DEFAULT_MODULUS": str(DEFAULT_MODULUS), "runtime_config_modulus": str(cfg.modulus), "q256": str(Q256_MODULUS)}


CHECKERS: dict[str, Callable[[], tuple[bool, dict[str, object]]]] = {
    "I1": check_i1,
    "I2": check_i2,
    "I3": check_i3,
    "I4": check_i4,
    "I5": check_i5,
    "I6": check_i6,
    "I7": check_i7,
    "I8": check_i8,
    "I9": check_i9,
    "I10": check_i10,
    "I11": check_i11,
    "I12": check_i12,
    "I13": check_i13,
    "I14": check_i14,
    "I15": check_i15,
    "I16": check_i16,
}


def load_spec() -> dict[str, object]:
    path = ROOT / "invariants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    spec = load_spec()
    results: list[dict[str, object]] = []
    all_passed = True
    for invariant in spec.get("invariants", []):
        inv = dict(invariant)
        inv_id = str(inv.get("id", ""))
        checker = CHECKERS.get(inv_id)
        if checker is None:
            passed = False
            detail = {"error": "missing checker"}
        else:
            try:
                passed, detail = checker()
            except Exception as exc:  # pragma: no cover - reported in CLI mode
                passed = False
                detail = {"exception": f"{type(exc).__name__}: {exc}"}
        all_passed = all_passed and passed
        results.append({
            "id": inv_id,
            "name": inv.get("name", ""),
            "passed": bool(passed),
            "check_type": inv.get("check_type", ""),
            "target_module": inv.get("target_module", ""),
            "detail": detail,
        })
    report = {
        "schema_version": "htce-invariant-verification-report-v1",
        "release_line": spec.get("release_line", ""),
        "modulus": spec.get("modulus", ""),
        "all_passed": all_passed,
        "passed_count": sum(1 for item in results if item["passed"]),
        "total_count": len(results),
        "results": results,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "invariants_verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{item['id']}] {item['name']}: {status}")
    print(f"invariants: {'PASS' if all_passed else 'FAIL'} ({report['passed_count']}/{report['total_count']})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
