#!/usr/bin/env python3
"""Bounded one-command v1.0 acceptance orchestrator.

The script avoids long-lived make dependency chains by running each release stage
as an explicit subprocess with a local timeout.  It preserves the existing
scripts/* pipeline; it does not replace any gate or benchmark.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PY = sys.executable

COMMANDS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("clean", ("make", "clean"), 120),
    ("compile", (PY, "-m", "compileall", "-q", "htce_origin", "scripts", "tests"), 180),
    ("pipeline-tree", ("make", "pipeline-tree"), 60),
    ("scan-float", (PY, "scripts/00_gates/scan_float_literals.py"), 60),
    ("version-sync", (PY, "scripts/00_gates/check_version_sync.py"), 60),
    ("invariants", (PY, "scripts/00_gates/check_invariants.py"), 120),
    ("organism-sanity", (PY, "scripts/01_sanity/run_organism.py"), 120),
    ("data-readiness", (PY, "scripts/02_benchmarks/prepare_training_data.py"), 120),
    ("benchmark", (PY, "scripts/02_benchmarks/run_benchmark.py"), 120),
    ("official-harness", (PY, "scripts/02_benchmarks/run_official_harness.py", "--max-examples-per-task", "15", "--long-memory-events", "10000", "--closed-loop-steps", "15"), 180),
    ("no-leakage", (PY, "scripts/02_benchmarks/run_no_leakage.py"), 120),
    ("topology", (PY, "scripts/03_topology_and_hardware/run_topology_acceptance.py"), 120),
    ("interaction-report", (PY, "scripts/03_topology_and_hardware/generate_interaction_report.py"), 180),
    ("hardware-width", (PY, "scripts/03_topology_and_hardware/run_hardware_width.py"), 120),
    ("stability", (PY, "scripts/04_stability/run_long_run_stability.py", "--smoke"), 180),
    ("artifacts", (PY, "scripts/05_artifacts/export_artifacts.py"), 180),
    ("trace-verify", (PY, "scripts/05_artifacts/verify_trace.py"), 60),
    ("manifest-verify", (PY, "scripts/05_artifacts/verify_manifest.py"), 60),
)


def run_stage(name: str, command: tuple[str, ...], timeout_seconds: int) -> None:
    print(f"[v1.0 acceptance] {name}: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=timeout_seconds,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)



def run_unified_living_adaptive_memory_stage() -> None:
    print("[v1.0 acceptance] unified living/adaptive/v1-revalidation smoke: in-process HTCERuntime", flush=True)
    from htce_origin.body.runtime import HTCERuntime
    from htce_origin.kernel.serialization import canonical_json_str

    runtime = HTCERuntime()
    runtime.wake()
    p25 = runtime.run_living_active_agent_simulation(steps=8, grid_size=5, include_dialog_policy=True).as_payload()
    print("[v1.0 acceptance] p25 stage PASS", flush=True)
    if not p25["trace_verified"] or p25["real_actions_allowed"] or not p25["unified_simulation"]:
        raise SystemExit("p25 unified living/dialog stage failed")
    if p25["dialog_metrics"].get("wrong_turns") != 0 or p25["dialog_metrics"].get("false_support_count") != 0:
        raise SystemExit("p25 dialog/action policy regressed")

    adaptive_runtime = HTCERuntime()
    adaptive_runtime.wake()
    p26 = adaptive_runtime.run_adaptive_policy_improvement_simulation(steps=12, grid_size=5).as_payload()
    print("[v1.0 acceptance] p26 stage PASS", flush=True)
    if not p26["trace_verified"] or p26["real_actions_allowed"] or not p26["single_runtime_loop"]:
        raise SystemExit("p26 adaptive stage boundary failed")
    if not p26["improvement_verified"]:
        raise SystemExit("p26 adaptive improvement was not verified")

    v1_runtime = HTCERuntime()
    v1_runtime.wake()
    v1 = v1_runtime.run_v1_clean_system_revalidation(stress_steps=4, grid_size=5).as_payload()
    print("[v1.0 acceptance] v1 clean-system revalidation stage PASS", flush=True)
    if not v1["passed"]:
        raise SystemExit("v1.0 clean system revalidation failed")
    if v1["external_false_support_count"] != 0 or v1["answer_key_visible_to_engine_count"] != 0:
        raise SystemExit("v1.0 external revalidation leaked answer key or produced false support")

    print(canonical_json_str({
        "p25_unified_living_dialog_simulation": "PASS",
        "p26_adaptive_policy_improvement": "PASS",
        "p26_episode_1_cost_raw": p26["episode_1"]["adaptive_cost_raw"],
        "p26_episode_2_cost_raw": p26["episode_2"]["adaptive_cost_raw"],
        "p26_improvement_margin_raw": p26["improvement_margin_raw"],
        "v1_clean_system_revalidation": "PASS",
        "v1_external_rows_passed": v1["external_rows_passed"],
        "v1_total_external_rows": v1["total_external_rows"],
        "v1_no_external_regression": v1["no_external_regression"],
        "v1_false_support_count": v1["external_false_support_count"],
        "trace_verified": p25["trace_verified"] and p26["trace_verified"] and v1["trace_verified"],
        "simulation_only": True,
    }), flush=True)

def main() -> None:
    for name, command, timeout_seconds in COMMANDS:
        run_stage(name, command, timeout_seconds)
        if name == "organism-sanity":
            run_unified_living_adaptive_memory_stage()
    print("HTCE-Origin v1.0 final_math Q256 clean system acceptance PASS", flush=True)


if __name__ == "__main__":
    main()
