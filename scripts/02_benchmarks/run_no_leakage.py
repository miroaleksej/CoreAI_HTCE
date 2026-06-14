#!/usr/bin/env python3
"""Run P18 no-leakage dynamic benchmark protocol and export artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.evaluation.no_leakage import P18NoLeakageProtocol
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P18 no-leakage dynamic benchmark protocol")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", default="htce-p18-private-seed")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    report = P18NoLeakageProtocol(seed=args.seed).run()
    payload = report.as_payload()
    payload["artifact_sha256"] = sha256_hex(payload)

    _write(artifacts / "no_leakage_protocol_report.json", payload)
    _write(artifacts / "seeded_private_goldset_commitments.json", {
        "schema_version": "htce-p18-seeded-private-goldset-commitments-v1",
        "commitment_count": len(payload["commitments"]),
        "commitments": payload["commitments"],
    })
    _write(artifacts / "counterfactual_rewrite_report.json", {
        "schema_version": "htce-p18-counterfactual-rewrite-report-v1",
        "counterfactual_tests": payload["counterfactual_tests"],
        "passed": all(item["passed"] for item in payload["counterfactual_tests"]),
    })
    _write(artifacts / "dynamic_task_public_cards.json", {
        "schema_version": "htce-p18-public-task-cards-v1",
        "rows": [
            {
                "task": row["task"],
                "family": row["family"],
                "engine_input_hash": row["engine_input_hash"],
                "public_task_hash": row["public_task_hash"],
                "answer_key_visible_to_engine": row["answer_key_visible_to_engine"],
            }
            for row in payload["rows"]
        ],
    })

    print("P18 no-leakage dynamic benchmark protocol")
    print(f"rows: {report.total_count}")
    print(f"passed: {report.passed}")
    print(f"false_support_count: {report.false_support_count}")
    print(f"answer_key_visible_count: {report.answer_key_visible_count}")
    print(f"trace_head: {report.trace_head}")
    print(f"artifacts_dir: {artifacts}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
