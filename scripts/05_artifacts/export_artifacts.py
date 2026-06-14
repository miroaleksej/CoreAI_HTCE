#!/usr/bin/env python3
"""Export buyer-safe Q256 acceptance artifacts with canonical JSON."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

from htce_origin.body.runtime import HTCERuntime, RuntimeRequest
from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex
ARTIFACTS.mkdir(exist_ok=True)


def write_canonical(name: str, payload: dict[str, object]) -> str:
    data = canonical_json_bytes(payload)
    digest = sha256_hex(data)
    enriched = dict(payload)
    enriched["artifact_sha256"] = digest
    path = ARTIFACTS / name
    path.write_bytes(canonical_json_bytes(enriched))
    return digest


def main() -> None:
    runtime = HTCERuntime()
    runtime.wake()
    closed_loop = runtime.run_closed_loop_simulation(steps=32)
    for text in (
        "FACT Mary located_in office EVID p12_fact_1",
        "QUERY Mary location EVID p12_query_1",
        "FACT Mary located_in garden EVID p12_fact_2",
        "QUERY Mary location EVID p12_query_2",
        "QUERY John location EVID p12_query_unknown",
    ):
        runtime.tick(RuntimeRequest(text, source="p12_artifact_export"))
    state_export = runtime.export_state()
    payload = {
        "artifact_type": "closed_loop_trace_export",
        "release_line": "v1.0_final_math_q256_clean",
        "modulus": Q256_MODULUS,
        "closed_loop": closed_loop.as_payload(),
        "runtime_state": state_export,
        "trace_verified": runtime.trace.verify(),
        "health": runtime.health(),
    }
    digest = write_canonical("closed_loop_trace_export.json", payload)
    print(f"closed_loop_trace_export: artifacts/closed_loop_trace_export.json sha256={digest}")

    living_runtime = HTCERuntime()
    living_runtime.wake()
    living_report = living_runtime.run_living_active_agent_simulation(steps=16, grid_size=5, include_dialog_policy=True)
    living_payload = {
        "artifact_type": "p25_unified_living_dialog_simulation_report",
        "release_line": "p25_unified_living_dialog_simulation_only_q256",
        "modulus": Q256_MODULUS,
        "report": living_report.as_payload(),
        "trace_verified": living_runtime.trace.verify(),
        "health": living_runtime.health(),
        "claim_boundary": {
            "consciousness_claimed": False,
            "multiple_shells": False,
            "qualia_claimed": False,
            "real_actions_allowed": False,
            "simulation_only": True,
            "single_runtime_loop": True,
        },
    }
    living_digest = write_canonical("p25_unified_living_dialog_simulation_report.json", living_payload)
    print(f"p25_unified_living_dialog_simulation_report: artifacts/p25_unified_living_dialog_simulation_report.json sha256={living_digest}")

    adaptive_runtime = HTCERuntime()
    adaptive_runtime.wake()
    adaptive_report = adaptive_runtime.run_adaptive_policy_improvement_simulation(steps=18, grid_size=5)
    adaptive_payload = {
        "artifact_type": "p26_adaptive_policy_improvement_report",
        "release_line": "p26_adaptive_policy_improvement_inside_single_simulation_q256",
        "modulus": Q256_MODULUS,
        "report": adaptive_report.as_payload(),
        "trace_verified": adaptive_runtime.trace.verify(),
        "health": adaptive_runtime.health(),
        "claim_boundary": {
            "consciousness_claimed": False,
            "multiple_shells": False,
            "qualia_claimed": False,
            "real_actions_allowed": False,
            "simulation_only": True,
            "single_runtime_loop": True,
        },
    }
    adaptive_digest = write_canonical("p26_adaptive_policy_improvement_report.json", adaptive_payload)
    print(f"p26_adaptive_policy_improvement_report: artifacts/p26_adaptive_policy_improvement_report.json sha256={adaptive_digest}")


    continual_runtime = HTCERuntime()
    continual_runtime.wake()
    continual_report = continual_runtime.run_continual_adaptive_memory_simulation(episodes=5, steps=18, grid_size=5)
    continual_payload = {
        "artifact_type": "p27_continual_adaptive_memory_report",
        "release_line": "p27_continual_adaptive_memory_without_regression_q256",
        "modulus": Q256_MODULUS,
        "report": continual_report.as_payload(),
        "trace_verified": continual_runtime.trace.verify(),
        "health": continual_runtime.health(),
        "claim_boundary": {
            "consciousness_claimed": False,
            "multiple_shells": False,
            "qualia_claimed": False,
            "real_actions_allowed": False,
            "simulation_only": True,
            "single_runtime_loop": True,
        },
    }
    continual_digest = write_canonical("p27_continual_adaptive_memory_report.json", continual_payload)
    print(f"p27_continual_adaptive_memory_report: artifacts/p27_continual_adaptive_memory_report.json sha256={continual_digest}")


    multitask_runtime = HTCERuntime()
    multitask_runtime.wake()
    multitask_report = multitask_runtime.run_continual_multitask_simulation(steps=18, grid_size=5)
    multitask_payload = {
        "artifact_type": "p28_continual_multitask_adaptive_memory_report",
        "release_line": "p28_continual_multitask_adaptive_memory_without_cross_domain_regression_q256",
        "modulus": Q256_MODULUS,
        "report": multitask_report.as_payload(),
        "trace_verified": multitask_runtime.trace.verify(),
        "health": multitask_runtime.health(),
        "claim_boundary": {
            "consciousness_claimed": False,
            "multiple_shells": False,
            "qualia_claimed": False,
            "real_actions_allowed": False,
            "simulation_only": True,
            "single_runtime_loop": True,
            "no_multitask_general_intelligence_claim": True,
        },
    }
    multitask_digest = write_canonical("p28_continual_multitask_adaptive_memory_report.json", multitask_payload)
    print(f"p28_continual_multitask_adaptive_memory_report: artifacts/p28_continual_multitask_adaptive_memory_report.json sha256={multitask_digest}")



    v1_runtime = HTCERuntime()
    v1_runtime.wake()
    v1_report = v1_runtime.run_v1_clean_system_revalidation(stress_steps=32, grid_size=5)
    v1_payload = {
        "artifact_type": "v1_clean_system_revalidation_report",
        "release_line": "v1.0_final_math_q256_clean",
        "modulus": Q256_MODULUS,
        "report": v1_report.as_payload(),
        "trace_verified": v1_runtime.trace.verify(),
        "health": v1_runtime.health(),
        "claim_boundary": {
            "consciousness_claimed": False,
            "multiple_shells": False,
            "qualia_claimed": False,
            "real_actions_allowed": False,
            "simulation_only": True,
            "single_runtime_loop": True,
            "external_gold_visible_to_engine": False,
            "no_agi_claim": True,
        },
    }
    v1_digest = write_canonical("v1_clean_system_revalidation_report.json", v1_payload)
    print(f"v1_clean_system_revalidation_report: artifacts/v1_clean_system_revalidation_report.json sha256={v1_digest}")

    capabilities = (ROOT / "capabilities.json").read_text(encoding="utf-8")
    cap_path = ARTIFACTS / "capabilities.json"
    cap_path.write_text(capabilities, encoding="utf-8")
    print("capabilities_export: artifacts/capabilities.json")


if __name__ == "__main__":
    main()