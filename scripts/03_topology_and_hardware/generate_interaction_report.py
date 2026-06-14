#!/usr/bin/env python3
"""Generate an integer-only L1/L2/L3 interaction report from exported state."""
from __future__ import annotations

import json
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
EXPORT = ARTIFACTS / "closed_loop_trace_export.json"

from htce_origin.kernel.core import TorusVector, active_state_digest
from htce_origin.kernel.q16 import q_vector_add
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex
from htce_origin.topology.guard import l3_betti_1_skeleton, validate_l3_semantic_window
def main() -> None:
    if not EXPORT.exists():
        runpy.run_path(str(ROOT / "scripts" / "05_artifacts" / "export_artifacts.py"), run_name="__main__")
    payload = json.loads(EXPORT.read_text(encoding="utf-8"))
    state = payload["runtime_state"]
    body = state["body"]
    layers = body["layers"]
    modulus = int(body["modulus"])
    l1 = layers["L1"]
    l2 = layers["L2"]
    l3 = layers["L3"]
    l2_work = body.get("l2_working_memory", {})
    l2_clean = tuple(int(v) for v in l2_work.get("clean_vector", l2["vector"]))
    l3_vector = tuple(int(v) for v in l3["vector"])

    l2_window = state.get("topology_guard", {}).get("l2_live_window", [])
    l2_window_count = len(l2_window) if isinstance(l2_window, list) else 0

    l3_points: list[tuple[int, ...]] = [l3_vector]
    if l3_vector:
        # Deterministic integer-neighborhood probe for the report only. It does
        # not mutate runtime state and it is labelled as diagnostic in payload.
        for index in range(1, 4):
            offset = tuple((index * (coord + 1)) % modulus for coord in range(len(l3_vector)))
            l3_points.append(q_vector_add(l3_vector, offset, modulus))
    beta0, beta1, point_count, edge_count = l3_betti_1_skeleton(l3_points, modulus=modulus)
    l3_validation = validate_l3_semantic_window(l3_points, modulus=modulus)

    report = {
        "report_type": "L1_L2_L3_Interaction_Report",
        "release_line": "p15_full_topology_acceptance_layer_q256",
        "modulus": modulus,
        "integer_only": True,
        "l1": {
            "clock": int(l1["clock"]),
            "digest": str(l1["digest"]),
            "vector_digest": active_state_digest(TorusVector(tuple(int(v) for v in l1["vector"]), modulus)),
        },
        "l2": {
            "clock": int(l2["clock"]),
            "digest": str(l2["digest"]),
            "clean_digest": active_state_digest(TorusVector(l2_clean, modulus)),
            "active_working_count": len(l2_work.get("active_contributions", [])),
            "episode_index": int(l2_work.get("episode_index", 0)),
            "episode_fact_count": int(l2_work.get("episode_fact_count", 0)),
            "live_window_count": l2_window_count,
        },
        "l3": {
            "clock": int(l3["clock"]),
            "digest": str(l3["digest"]),
            "vector_digest": active_state_digest(TorusVector(l3_vector, modulus)),
            "topology_beta0": int(beta0),
            "topology_beta1": int(beta1),
            "topology_point_count": int(point_count),
            "topology_edge_count": int(edge_count),
            "topology_passed": bool(l3_validation.passed),
            "topology_reason": l3_validation.reason,
        },
        "world_model": state["world_model"]["self_model"],
        "trace_event_count": len(state["trace_events"]),
        "trace_verified": bool(payload["trace_verified"]),
        "diagnostic_probe_mutates_runtime": False,
    }
    digest = sha256_hex(canonical_json_bytes(report))
    report["artifact_sha256"] = digest
    path = ARTIFACTS / "l1_l2_l3_interaction_report.json"
    path.write_bytes(canonical_json_bytes(report))
    print(f"l1_l2_l3_interaction_report: artifacts/l1_l2_l3_interaction_report.json sha256={digest}")


if __name__ == "__main__":
    main()