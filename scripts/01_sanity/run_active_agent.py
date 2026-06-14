#!/usr/bin/env python3
"""Run v1.0 unified living/adaptive/multitask/external revalidation sanity loops."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.body.runtime import HTCERuntime
from htce_origin.kernel.serialization import canonical_json_str


def main() -> None:
    runtime = HTCERuntime()
    runtime.wake()
    report = runtime.run_living_active_agent_simulation(steps=16, grid_size=5, include_dialog_policy=True)
    payload = report.as_payload()
    if not payload["trace_verified"]:
        raise SystemExit("p25 trace verification failed")
    if payload["real_actions_allowed"]:
        raise SystemExit("p25 must remain simulation-only")
    if payload["final_goal_distance"] >= payload["start_goal_distance"]:
        raise SystemExit("p25 unified active loop did not reduce goal distance")
    if not payload["unified_simulation"]:
        raise SystemExit("p25 must execute dialog/action policy inside the same simulation")
    if payload["dialog_metrics"].get("wrong_turns") != 0:
        raise SystemExit("p25 unified dialog/action policy had wrong turns")
    if payload["dialog_metrics"].get("false_support_count") != 0:
        raise SystemExit("p25 unified dialog/action policy produced false support")

    adaptive_runtime = HTCERuntime()
    adaptive_runtime.wake()
    adaptive = adaptive_runtime.run_adaptive_policy_improvement_simulation(steps=18, grid_size=5)
    adaptive_payload = adaptive.as_payload()
    if not adaptive_payload["trace_verified"]:
        raise SystemExit("p26 trace verification failed")
    if adaptive_payload["real_actions_allowed"]:
        raise SystemExit("p26 must remain simulation-only")
    if not adaptive_payload["single_runtime_loop"]:
        raise SystemExit("p26 must run inside one HTCERuntime loop")
    if not adaptive_payload["improvement_verified"]:
        raise SystemExit("p26 adaptive improvement was not verified")

    continual_runtime = HTCERuntime()
    continual_runtime.wake()
    continual = continual_runtime.run_continual_adaptive_memory_simulation(episodes=5, steps=18, grid_size=5)
    continual_payload = continual.as_payload()
    if not continual_payload["trace_verified"]:
        raise SystemExit("p27 trace verification failed")
    if continual_payload["real_actions_allowed"]:
        raise SystemExit("p27 must remain simulation-only")
    if not continual_payload["single_runtime_loop"]:
        raise SystemExit("p27 must run inside one HTCERuntime loop")
    if not continual_payload["no_regression_passed"]:
        raise SystemExit("p27 continual adaptive memory regressed")


    multitask_runtime = HTCERuntime()
    multitask_runtime.wake()
    multitask = multitask_runtime.run_continual_multitask_simulation(steps=18, grid_size=5)
    multitask_payload = multitask.as_payload()
    if not multitask_payload["trace_verified"]:
        raise SystemExit("p28 trace verification failed")
    if multitask_payload["real_actions_allowed"]:
        raise SystemExit("p28 must remain simulation-only")
    if not multitask_payload["single_runtime_loop"]:
        raise SystemExit("p28 must run inside one HTCERuntime loop")
    if not multitask_payload["no_cross_domain_regression"]:
        raise SystemExit("p28 cross-domain regression detected")
    if multitask_payload["false_support_count"] != 0 or multitask_payload["wrong_turn_count"] != 0:
        raise SystemExit("p28 produced false support or wrong turns")



    v1_runtime = HTCERuntime()
    v1_runtime.wake()
    v1_report = v1_runtime.run_v1_clean_system_revalidation(stress_steps=16, grid_size=5)
    v1_payload = v1_report.as_payload()
    if not v1_payload["passed"]:
        raise SystemExit("v1.0 clean-system revalidation failed")
    if v1_payload["external_false_support_count"] != 0 or v1_payload["answer_key_visible_to_engine_count"] != 0:
        raise SystemExit("v1.0 external revalidation leaked gold or false support")

    print(canonical_json_str({
        "dialog_metrics": payload["dialog_metrics"],
        "final_goal_distance": payload["final_goal_distance"],
        "heartbeat_count": payload["heartbeat_count"],
        "p25_unified_living_dialog_simulation": "PASS",
        "p26_adaptive_policy_improvement": "PASS",
        "p26_episode_1_cost_raw": adaptive_payload["episode_1"]["adaptive_cost_raw"],
        "p26_episode_2_cost_raw": adaptive_payload["episode_2"]["adaptive_cost_raw"],
        "p26_improvement_margin_raw": adaptive_payload["improvement_margin_raw"],
        "p26_l3_rules_promoted_during_sleep": adaptive_payload["l3_rules_promoted_during_sleep"],
        "p26_learned_avoid_cells": adaptive_payload["learned_avoid_cells"],
        "p27_continual_adaptive_memory": "PASS",
        "p27_adaptive_costs_raw": [episode["adaptive_cost_raw"] for episode in continual_payload["episodes"]],
        "p27_false_support_count": continual_payload["false_support_count"],
        "p27_no_regression_passed": continual_payload["no_regression_passed"],
        "p27_retained_l3_rules_final": continual_payload["retained_l3_rules_final"],
        "p28_continual_multitask_adaptive_memory": "PASS",
        "p28_domain_cost_history_raw": multitask_payload["domain_cost_history_raw"],
        "p28_false_support_count": multitask_payload["false_support_count"],
        "p28_no_cross_domain_regression": multitask_payload["no_cross_domain_regression"],
        "v1_clean_system_revalidation": "PASS",
        "v1_external_rows_passed": v1_payload["external_rows_passed"],
        "v1_total_external_rows": v1_payload["total_external_rows"],
        "v1_no_external_regression": v1_payload["no_external_regression"],
        "reached_goal": payload["reached_goal"],
        "simulation_only": True,
        "trace_verified": payload["trace_verified"] and adaptive_payload["trace_verified"] and continual_payload["trace_verified"] and multitask_payload["trace_verified"] and v1_payload["trace_verified"],
        "unified_simulation": payload["unified_simulation"] and adaptive_payload["single_runtime_loop"] and continual_payload["single_runtime_loop"] and multitask_payload["single_runtime_loop"],
    }))


if __name__ == "__main__":
    main()
