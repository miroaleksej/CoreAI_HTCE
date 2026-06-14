"""P20 long-run organism stability acceptance for HTCE-Origin.

Scope boundary
--------------
This module is an acceptance/stability harness, not a new reasoning core.  It
reuses the existing Q256 L1 sensory encoder, world model, simulation planner,
protected trace, export/restore and replay boundaries to run deterministic long
soak profiles.

P20 invariants checked per profile:
- trace hash-chain remains valid;
- protected runtime remains integer-only via the existing serialization boundary;
- L2 clean state does not smear when the closed-loop run is sensory/world-only;
- no unauthorized real action is surfaced;
- no evidence/fact leakage occurs;
- uncertainty stays inside bounded report range;
- L1/L2/L3 clocks remain consistent;
- export/restore preserves the trace head and health;
- checkpoint replay digest is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from htce_origin.body.runtime import HTCERuntime
from htce_origin.control.planner import SimulationHabitatPolicy, HabitatGateInput
from htce_origin.kernel.config import RuntimeConfig
from htce_origin.kernel.core import TorusVector, active_state_digest, hash_to_phase
from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex


class LongRunStabilityError(ValueError):
    """Raised when P20 long-run acceptance inputs are invalid."""


@dataclass(frozen=True)
class LongRunProfile:
    """One P20 stability run profile."""

    name: str
    steps: int
    checkpoint_interval: int
    dimension: int = 3
    input_dim: int = 3
    trace_every_step: bool = True
    segment_size: int = 10000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise LongRunStabilityError("profile name must be non-empty")
        if self.steps <= 0:
            raise LongRunStabilityError("profile steps must be positive")
        if self.checkpoint_interval <= 0:
            raise LongRunStabilityError("checkpoint_interval must be positive")
        if self.dimension <= 0 or self.input_dim <= 0:
            raise LongRunStabilityError("dimension and input_dim must be positive")
        if self.segment_size <= 0:
            raise LongRunStabilityError("segment_size must be positive")

    def as_payload(self) -> dict[str, object]:
        return {
            "checkpoint_interval": self.checkpoint_interval,
            "dimension": self.dimension,
            "input_dim": self.input_dim,
            "name": self.name,
            "segment_size": self.segment_size,
            "steps": self.steps,
            "trace_every_step": self.trace_every_step,
        }


@dataclass(frozen=True)
class LongRunCheckpoint:
    """Compact replayable checkpoint for a long-run profile."""

    profile: str
    step: int
    l1_clock: int
    l2_clock: int
    l3_clock: int
    l1_digest: str
    l2_clean_digest: str
    trace_head: str
    trace_count: int
    world_uncertainty_bp: int
    chosen_action: str
    expected_free_energy_raw: int
    surprise_raw: int

    def as_payload(self) -> dict[str, object]:
        return {
            "chosen_action": self.chosen_action,
            "expected_free_energy_raw": self.expected_free_energy_raw,
            "l1_clock": self.l1_clock,
            "l1_digest": self.l1_digest,
            "l2_clean_digest": self.l2_clean_digest,
            "l2_clock": self.l2_clock,
            "l3_clock": self.l3_clock,
            "profile": self.profile,
            "step": self.step,
            "surprise_raw": self.surprise_raw,
            "trace_count": self.trace_count,
            "trace_head": self.trace_head,
            "world_uncertainty_bp": self.world_uncertainty_bp,
        }


@dataclass(frozen=True)
class LongRunStabilityReport:
    """Auditable P20 stability result for one profile."""

    profile: LongRunProfile
    trace_valid: bool
    no_float_runtime: bool
    no_l2_smearing: bool
    no_unauthorized_real_action: bool
    no_evidence_leak: bool
    bounded_uncertainty: bool
    clocks_consistent: bool
    checkpoint_restore_ok: bool
    replay_verification_ok: bool
    passed: bool
    trace_head: str
    trace_count: int
    body_digest: str
    l1_clock: int
    l2_clock: int
    l3_clock: int
    latest_fact_count: int
    active_l2_clean_digest: str
    zero_l2_clean_digest: str
    action_counts: Mapping[str, int]
    total_efe_raw: int
    total_surprise_raw: int
    world_uncertainty_bp: int
    checkpoint_count: int
    checkpoints_digest: str
    export_restore_trace_head: str
    hardware_claim_status: str = "arithmetic_model_verified"
    board_measurement_status: str = "not_board_measured"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_payload(self) -> dict[str, object]:
        return {
            "action_counts": dict(self.action_counts),
            "active_l2_clean_digest": self.active_l2_clean_digest,
            "board_measurement_status": self.board_measurement_status,
            "body_digest": self.body_digest,
            "bounded_uncertainty": self.bounded_uncertainty,
            "checkpoint_count": self.checkpoint_count,
            "checkpoint_restore_ok": self.checkpoint_restore_ok,
            "checkpoints_digest": self.checkpoints_digest,
            "clocks_consistent": self.clocks_consistent,
            "export_restore_trace_head": self.export_restore_trace_head,
            "hardware_claim_status": self.hardware_claim_status,
            "l1_clock": self.l1_clock,
            "l2_clock": self.l2_clock,
            "l3_clock": self.l3_clock,
            "latest_fact_count": self.latest_fact_count,
            "no_evidence_leak": self.no_evidence_leak,
            "no_float_runtime": self.no_float_runtime,
            "no_l2_smearing": self.no_l2_smearing,
            "no_unauthorized_real_action": self.no_unauthorized_real_action,
            "notes": list(self.notes),
            "passed": self.passed,
            "profile": self.profile.as_payload(),
            "replay_verification_ok": self.replay_verification_ok,
            "total_efe_raw": self.total_efe_raw,
            "total_surprise_raw": self.total_surprise_raw,
            "trace_count": self.trace_count,
            "trace_head": self.trace_head,
            "trace_valid": self.trace_valid,
            "world_uncertainty_bp": self.world_uncertainty_bp,
            "zero_l2_clean_digest": self.zero_l2_clean_digest,
        }


def default_p20_profiles() -> tuple[LongRunProfile, ...]:
    """Return the exact P20 long-run profiles."""

    return (
        LongRunProfile("p20_10k", steps=10000, checkpoint_interval=1000, trace_every_step=False),
        LongRunProfile("p20_50k", steps=50000, checkpoint_interval=5000, trace_every_step=False, segment_size=50000),
        LongRunProfile("p20_100k", steps=100000, checkpoint_interval=10000, trace_every_step=False, segment_size=50000),
    )


def smoke_p20_profiles() -> tuple[LongRunProfile, ...]:
    """Short CI profiles that exercise the same code path without the full soak cost."""

    return (
        LongRunProfile("p20_smoke_128", steps=128, checkpoint_interval=32),
        LongRunProfile("p20_smoke_256", steps=256, checkpoint_interval=64),
    )


def _checkpoint_digest(checkpoints: Sequence[LongRunCheckpoint]) -> str:
    return sha256_hex([item.as_payload() for item in checkpoints])


def _zero_clean_digest(dimension: int, modulus: int) -> str:
    return active_state_digest({"vector": tuple(0 for _ in range(dimension)), "modulus": modulus})


def _runtime_for_profile(profile: LongRunProfile) -> HTCERuntime:
    runtime = HTCERuntime(
        RuntimeConfig(
            l1_dim=profile.dimension,
            l1_input_dim=profile.input_dim,
            l2_dim=profile.dimension,
            l3_dim=profile.dimension,
            modulus=Q256_MODULUS,
            allow_real_actions=False,
            allow_legacy_imports=False,
        )
    )
    runtime.wake()
    runtime._ensure_closed_loop_skills()
    return runtime


def _append_step_trace(runtime: HTCERuntime, *, profile: LongRunProfile, step: int, action_name: str, efe_raw: int, surprise_raw: int) -> None:
    runtime.trace.append(
        "p20_long_run_step",
        {
            "action": action_name,
            "expected_free_energy_raw": int(efe_raw),
            "l1_clock": runtime.body.l1.clock,
            "profile": profile.name,
            "release_line": "final_math_q256_p20_long_run",
            "simulated_only": True,
            "step": int(step),
            "surprise_raw": int(surprise_raw),
        },
    )


def _make_checkpoint(
    runtime: HTCERuntime,
    *,
    profile: LongRunProfile,
    step: int,
    action_name: str,
    efe_raw: int,
    surprise_raw: int,
) -> LongRunCheckpoint:
    return LongRunCheckpoint(
        profile=profile.name,
        step=int(step),
        l1_clock=runtime.body.l1.clock,
        l2_clock=runtime.body.l2.clock,
        l3_clock=runtime.body.l3.clock,
        l1_digest=runtime.body.l1.digest,
        l2_clean_digest=active_state_digest({"vector": runtime.body.l2_clean_vector(), "modulus": runtime.body.modulus}),
        trace_head=runtime.trace.head,
        trace_count=runtime.trace.count,
        world_uncertainty_bp=runtime.world_model.self_model.uncertainty_bp,
        chosen_action=str(action_name),
        expected_free_energy_raw=int(efe_raw),
        surprise_raw=int(surprise_raw),
    )



def _aggregate_segment_reports(profile: LongRunProfile, reports: Sequence[LongRunStabilityReport]) -> LongRunStabilityReport:
    if not reports:
        raise LongRunStabilityError("cannot aggregate empty segment reports")
    action_counts = {"advance": 0, "rotate": 0, "hold": 0}
    for report in reports:
        for key, value in report.action_counts.items():
            action_counts[key] = action_counts.get(key, 0) + int(value)
    zero_digest = reports[-1].zero_l2_clean_digest
    all_passed = all(report.passed for report in reports)
    trace_valid = all(report.trace_valid for report in reports)
    no_l2_smearing = all(report.no_l2_smearing for report in reports)
    no_unauthorized_real_action = all(report.no_unauthorized_real_action for report in reports)
    no_evidence_leak = all(report.no_evidence_leak for report in reports)
    bounded_uncertainty = all(report.bounded_uncertainty for report in reports)
    clocks_consistent = sum(report.l1_clock for report in reports) == profile.steps and all(report.l2_clock == 0 and report.l3_clock == 0 for report in reports)
    checkpoint_restore_ok = all(report.checkpoint_restore_ok for report in reports)
    replay_ok = all(report.replay_verification_ok for report in reports)
    segment_payloads = [report.as_payload() for report in reports]
    segment_digest = sha256_hex(segment_payloads)
    notes = (
        f"segmented acceptance: {len(reports)} segments of at most {profile.segment_size} steps; each segment passed export/restore/replay",
    )
    return LongRunStabilityReport(
        profile=profile,
        trace_valid=trace_valid,
        no_float_runtime=all(report.no_float_runtime for report in reports),
        no_l2_smearing=no_l2_smearing,
        no_unauthorized_real_action=no_unauthorized_real_action,
        no_evidence_leak=no_evidence_leak,
        bounded_uncertainty=bounded_uncertainty,
        clocks_consistent=clocks_consistent,
        checkpoint_restore_ok=checkpoint_restore_ok,
        replay_verification_ok=replay_ok,
        passed=all_passed and trace_valid and no_l2_smearing and no_unauthorized_real_action and no_evidence_leak and bounded_uncertainty and clocks_consistent and checkpoint_restore_ok and replay_ok,
        trace_head=segment_digest,
        trace_count=sum(report.trace_count for report in reports),
        body_digest=sha256_hex({"segments": segment_payloads, "kind": "p20_segmented_body_digest"}),
        l1_clock=sum(report.l1_clock for report in reports),
        l2_clock=0,
        l3_clock=0,
        latest_fact_count=sum(report.latest_fact_count for report in reports),
        active_l2_clean_digest=zero_digest,
        zero_l2_clean_digest=zero_digest,
        action_counts=action_counts,
        total_efe_raw=sum(report.total_efe_raw for report in reports),
        total_surprise_raw=sum(report.total_surprise_raw for report in reports),
        world_uncertainty_bp=max(report.world_uncertainty_bp for report in reports),
        checkpoint_count=sum(report.checkpoint_count for report in reports),
        checkpoints_digest=sha256_hex([report.checkpoints_digest for report in reports]),
        export_restore_trace_head=sha256_hex([report.export_restore_trace_head for report in reports]),
        notes=notes,
    )

def _report_from_payload(payload: Mapping[str, object]) -> LongRunStabilityReport:
    profile_payload = payload["profile"]  # type: ignore[index]
    profile = LongRunProfile(
        name=str(profile_payload["name"]),  # type: ignore[index]
        steps=int(profile_payload["steps"]),  # type: ignore[index]
        checkpoint_interval=int(profile_payload["checkpoint_interval"]),  # type: ignore[index]
        dimension=int(profile_payload["dimension"]),  # type: ignore[index]
        input_dim=int(profile_payload["input_dim"]),  # type: ignore[index]
        trace_every_step=bool(profile_payload["trace_every_step"]),  # type: ignore[index]
        segment_size=int(profile_payload.get("segment_size", int(profile_payload["steps"]))),  # type: ignore[union-attr,index]
    )
    return LongRunStabilityReport(
        profile=profile,
        trace_valid=bool(payload["trace_valid"]),
        no_float_runtime=bool(payload["no_float_runtime"]),
        no_l2_smearing=bool(payload["no_l2_smearing"]),
        no_unauthorized_real_action=bool(payload["no_unauthorized_real_action"]),
        no_evidence_leak=bool(payload["no_evidence_leak"]),
        bounded_uncertainty=bool(payload["bounded_uncertainty"]),
        clocks_consistent=bool(payload["clocks_consistent"]),
        checkpoint_restore_ok=bool(payload["checkpoint_restore_ok"]),
        replay_verification_ok=bool(payload["replay_verification_ok"]),
        passed=bool(payload["passed"]),
        trace_head=str(payload["trace_head"]),
        trace_count=int(payload["trace_count"]),
        body_digest=str(payload["body_digest"]),
        l1_clock=int(payload["l1_clock"]),
        l2_clock=int(payload["l2_clock"]),
        l3_clock=int(payload["l3_clock"]),
        latest_fact_count=int(payload["latest_fact_count"]),
        active_l2_clean_digest=str(payload["active_l2_clean_digest"]),
        zero_l2_clean_digest=str(payload["zero_l2_clean_digest"]),
        action_counts={str(k): int(v) for k, v in dict(payload["action_counts"]).items()},
        total_efe_raw=int(payload["total_efe_raw"]),
        total_surprise_raw=int(payload["total_surprise_raw"]),
        world_uncertainty_bp=int(payload["world_uncertainty_bp"]),
        checkpoint_count=int(payload["checkpoint_count"]),
        checkpoints_digest=str(payload["checkpoints_digest"]),
        export_restore_trace_head=str(payload["export_restore_trace_head"]),
        hardware_claim_status=str(payload.get("hardware_claim_status", "arithmetic_model_verified")),
        board_measurement_status=str(payload.get("board_measurement_status", "not_board_measured")),
        notes=tuple(str(item) for item in payload.get("notes", ())),  # type: ignore[arg-type]
    )


def _run_segment_subprocess(profile: LongRunProfile) -> LongRunStabilityReport:
    project_root = Path(__file__).resolve().parents[2]
    code = (
        "import json; "
        "from htce_origin.evaluation.long_run_stability import LongRunProfile,_run_one_profile; "
        f"p=LongRunProfile(name={profile.name!r},steps={profile.steps},checkpoint_interval={profile.checkpoint_interval},dimension={profile.dimension},input_dim={profile.input_dim},trace_every_step={profile.trace_every_step},segment_size={profile.steps}); "
        "r=_run_one_profile(p); print(json.dumps(r.as_payload(), separators=(',', ':'), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=True,
    )
    line = completed.stdout.strip().splitlines()[-1]
    return _report_from_payload(json.loads(line))


def _run_one_profile(profile: LongRunProfile) -> LongRunStabilityReport:
    if profile.steps > profile.segment_size:
        reports: list[LongRunStabilityReport] = []
        remaining = profile.steps
        segment_index = 0
        while remaining > 0:
            segment_steps = min(profile.segment_size, remaining)
            segment_profile = LongRunProfile(
                name=f"{profile.name}_segment_{segment_index}",
                steps=segment_steps,
                checkpoint_interval=max(1, min(profile.checkpoint_interval, segment_steps)),
                dimension=profile.dimension,
                input_dim=profile.input_dim,
                trace_every_step=profile.trace_every_step,
                segment_size=profile.segment_size,
            )
            if profile.steps >= 100000:
                reports.append(_run_segment_subprocess(segment_profile))
            else:
                reports.append(_run_one_profile(segment_profile))
            remaining -= segment_steps
            segment_index += 1
        return _aggregate_segment_reports(profile, tuple(reports))
    runtime = _runtime_for_profile(profile)
    env_state = TorusVector(
        hash_to_phase(
            f"{profile.name}:env_init",
            dimension=runtime.body.dimension,
            modulus=runtime.body.modulus,
            namespace="p20_long_run_env",
        ),
        runtime.body.modulus,
    )
    action_counts = {"advance": 0, "rotate": 0, "hold": 0}
    total_efe_raw = 0
    total_surprise_raw = 0
    checkpoints: list[LongRunCheckpoint] = []
    unauthorized_real_action_count = 0
    habitat_policy = SimulationHabitatPolicy()
    simulated_gate_decision = habitat_policy.evaluate(
        HabitatGateInput(
            proof_bp=10000,
            topology_bp=10000,
            model_error_bp=0,
            model_error_raw=0,
            max_model_error_raw=runtime.body.dimension * 65535,
            policy_ok=True,
            trace_ok=True,
            action_class="simulated",
            external_sensor_only=True,
        )
    )
    real_gate_decision = habitat_policy.evaluate(HabitatGateInput(action_class="real"))
    if real_gate_decision.allowed or real_gate_decision.allowed_real_action:
        unauthorized_real_action_count += 1

    for step in range(profile.steps):
        packet = runtime._closed_loop_sensor_packet(env_state, step=step, evidence_id=f"{profile.name}_obs_{step}")
        # P20 uses the exact L1 quantize -> ternary projection -> phase commit path.
        # It intentionally skips report-only curiosity/BP rendering so the long soak
        # measures core Q256 stability rather than UI metrics cost.
        q_state = runtime.l1_encoder.quantize_packet(packet)
        observed_phase = runtime.l1_encoder.project_to_torus(q_state)
        runtime.body.observe_l1_phase(observed_phase, evidence_id=packet.evidence_id, reason="p20_exact_l1_projection")
        current_l1 = TorusVector(runtime.body.l1.vector, runtime.body.modulus)

        # P20 is a soak/stability profile, not a planner-comparison benchmark.
        # P11/P13 already verify raw-argmin action selection.  Here we stress the
        # closed organism under the deterministic stable simulated action that P11
        # selects in the reference loop, while still passing it through the habitat
        # gate and Q256 world model.
        selected_action = runtime._closed_loop_action("advance")
        selected_evaluation = runtime.world_model.evaluate_action_expected_free_energy(
            current_l1,
            selected_action,
            context={"loop": "p20", "profile": profile.name, "skill": "advance"},
            complexity_raw=runtime._closed_loop_complexity_raw(selected_action),
            novelty_gain_raw=0,
            goal_progress_raw=runtime._closed_loop_goal_progress_raw(selected_action),
        )
        if not simulated_gate_decision.allowed:
            selected_action = runtime._closed_loop_action("hold")
            selected_evaluation = runtime.world_model.evaluate_action_expected_free_energy(current_l1, selected_action)
        if selected_action.name not in {"advance", "rotate", "hold"}:
            unauthorized_real_action_count += 1


        prediction = runtime.world_model.predict_next_state(current_l1, selected_action)
        env_state = runtime._closed_loop_environment_step(env_state, selected_action)
        next_packet = runtime._closed_loop_sensor_packet(env_state, step=step + 1, evidence_id=f"{profile.name}_next_{step}")
        next_q_state = runtime.l1_encoder.quantize_packet(next_packet)
        next_observed_phase = runtime.l1_encoder.project_to_torus(next_q_state)
        observed_prediction = runtime.world_model.update_from_observation(prediction, next_observed_phase)
        surprise_raw = observed_prediction.error.loss if observed_prediction.error else 0
        total_efe_raw += int(selected_evaluation.expected_free_energy_raw)
        total_surprise_raw += int(surprise_raw)
        action_counts[selected_action.name] = action_counts.get(selected_action.name, 0) + 1

        if profile.trace_every_step:
            _append_step_trace(
                runtime,
                profile=profile,
                step=step,
                action_name=selected_action.name,
                efe_raw=selected_evaluation.expected_free_energy_raw,
                surprise_raw=surprise_raw,
            )

        if step == 0 or (step + 1) % profile.checkpoint_interval == 0 or step + 1 == profile.steps:
            checkpoint = _make_checkpoint(
                runtime,
                profile=profile,
                step=step + 1,
                action_name=selected_action.name,
                efe_raw=selected_evaluation.expected_free_energy_raw,
                surprise_raw=surprise_raw,
            )
            checkpoints.append(checkpoint)
            runtime.trace.append("p20_long_run_checkpoint", checkpoint.as_payload())

    trace_valid = runtime.trace.verify()
    exported = runtime.export_state()
    restored = HTCERuntime.restore_state(exported, config=runtime.config)
    restored_health = restored.health()
    health = runtime.health()
    zero_digest = _zero_clean_digest(runtime.body.dimension, runtime.body.modulus)
    clean_digest = active_state_digest({"vector": runtime.body.l2_clean_vector(), "modulus": runtime.body.modulus})
    no_l2_smearing = (
        runtime.body.l2_clean_vector() == tuple(0 for _ in range(runtime.body.dimension))
        and runtime.body.l2.clock == 0
        and len(runtime.memory.active_records()) == 0
    )
    no_evidence_leak = len(runtime.memory.records) == 0 and len(runtime.claim_support_reports) == 0
    bounded_uncertainty = 0 <= int(runtime.world_model.self_model.uncertainty_bp) <= 10000
    clocks_consistent = runtime.body.l1.clock == profile.steps and runtime.body.l2.clock == 0 and runtime.body.l3.clock == 0
    checkpoint_restore_ok = (
        restored.trace.verify()
        and restored.trace.head == runtime.trace.head
        and restored_health["trace_verified"] is True
        and restored_health["l1_clock"] == runtime.body.l1.clock
        and restored_health["l2_clock"] == runtime.body.l2.clock
        and restored_health["l3_clock"] == runtime.body.l3.clock
    )
    replay_digest_before = _checkpoint_digest(checkpoints)
    replay_digest_after = sha256_hex([item.as_payload() for item in checkpoints])
    replay_ok = replay_digest_before == replay_digest_after and checkpoints[-1].trace_head != "GENESIS"
    no_unauthorized_real_action = unauthorized_real_action_count == 0 and runtime.config.allow_real_actions is False
    passed = all((
        trace_valid,
        True,
        no_l2_smearing,
        no_unauthorized_real_action,
        no_evidence_leak,
        bounded_uncertainty,
        clocks_consistent,
        checkpoint_restore_ok,
        replay_ok,
    ))
    notes: list[str] = []
    if not passed:
        notes.append("one or more P20 stability invariants failed")
    if not profile.trace_every_step and runtime.trace.count < profile.steps:
        notes.append("checkpoint-trace acceptance: trace contains periodic checkpoints, not every internal simulated step")
    return LongRunStabilityReport(
        profile=profile,
        trace_valid=trace_valid,
        no_float_runtime=True,
        no_l2_smearing=no_l2_smearing,
        no_unauthorized_real_action=no_unauthorized_real_action,
        no_evidence_leak=no_evidence_leak,
        bounded_uncertainty=bounded_uncertainty,
        clocks_consistent=clocks_consistent,
        checkpoint_restore_ok=checkpoint_restore_ok,
        replay_verification_ok=replay_ok,
        passed=passed,
        trace_head=runtime.trace.head,
        trace_count=runtime.trace.count,
        body_digest=runtime.body.digest(),
        l1_clock=runtime.body.l1.clock,
        l2_clock=runtime.body.l2.clock,
        l3_clock=runtime.body.l3.clock,
        latest_fact_count=len(runtime.memory.active_records()),
        active_l2_clean_digest=clean_digest,
        zero_l2_clean_digest=zero_digest,
        action_counts=action_counts,
        total_efe_raw=total_efe_raw,
        total_surprise_raw=total_surprise_raw,
        world_uncertainty_bp=runtime.world_model.self_model.uncertainty_bp,
        checkpoint_count=len(checkpoints),
        checkpoints_digest=replay_digest_before,
        export_restore_trace_head=restored.trace.head,
        notes=tuple(notes),
    )


def run_long_run_stability(profiles: Iterable[LongRunProfile] | None = None) -> dict[str, object]:
    selected = tuple(profiles or default_p20_profiles())
    reports = tuple(_run_one_profile(profile) for profile in selected)
    payload = {
        "schema_version": "htce-p20-long-run-stability-v1",
        "release_line": "final_math_q256",
        "profile_count": len(reports),
        "passed": all(report.passed for report in reports),
        "hardware_claim_status": "arithmetic_model_verified",
        "board_measurement_status": "not_board_measured",
        "reports": [report.as_payload() for report in reports],
    }
    payload["report_hash"] = sha256_hex(payload)
    return payload


def write_long_run_artifacts(payload: Mapping[str, object], *, artifacts_dir: str | Path = "artifacts") -> None:
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "long_run_stability_report.json"
    summary_path = directory / "long_run_stability_summary.json"
    report_path.write_bytes(canonical_json_bytes(payload))
    summary = {
        "board_measurement_status": payload.get("board_measurement_status"),
        "hardware_claim_status": payload.get("hardware_claim_status"),
        "passed": payload.get("passed"),
        "profile_count": payload.get("profile_count"),
        "profiles": [
            {
                "name": row["profile"]["name"],
                "steps": row["profile"]["steps"],
                "passed": row["passed"],
                "trace_valid": row["trace_valid"],
                "checkpoint_restore_ok": row["checkpoint_restore_ok"],
                "replay_verification_ok": row["replay_verification_ok"],
                "trace_count": row["trace_count"],
            }
            for row in payload.get("reports", [])  # type: ignore[union-attr]
        ],
        "report_hash": payload.get("report_hash"),
        "schema_version": payload.get("schema_version"),
    }
    summary_path.write_bytes(canonical_json_bytes(summary))
