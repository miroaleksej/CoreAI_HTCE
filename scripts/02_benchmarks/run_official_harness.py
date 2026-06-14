#!/usr/bin/env python3
"""P17 official/near-official benchmark harness runner.

Default mode generates release-safe traceable matrix artifacts without bundling
external datasets. Optional --babi-path / --dialog-path / --modified-dialog-path /
--permuted-dialog-path execute external official rows from user-supplied paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from htce_origin.evaluation.official_harness import P17OfficialBenchmarkHarness, P17SuiteKind
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P17 official benchmark harness")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--long-memory-events", type=int, default=10000, choices=(10000, 50000, 100000))
    parser.add_argument("--closed-loop-steps", type=int, default=15)
    parser.add_argument("--babi-path")
    parser.add_argument("--dialog-path")
    parser.add_argument("--modified-dialog-path")
    parser.add_argument("--permuted-dialog-path")
    parser.add_argument("--max-examples-per-task", type=int, default=25)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    harness = P17OfficialBenchmarkHarness()

    spec_payload = harness.spec_payload()
    _write_json(artifacts / "official_benchmark_specs.json", spec_payload)

    matrix = harness.run_release_smoke_matrix(
        long_memory_events=args.long_memory_events,
        closed_loop_steps=args.closed_loop_steps,
    )
    matrix_payload = matrix.as_payload()
    matrix_payload["artifact_sha256"] = sha256_hex(matrix_payload)
    _write_json(artifacts / "official_benchmark_matrix.json", matrix_payload)

    external_reports: dict[str, object] = {}
    if args.babi_path:
        report = harness.run_external_babi_20(args.babi_path, max_examples_per_task=args.max_examples_per_task)
        external_reports["babi_20"] = report.as_payload()
        _write_json(artifacts / "official_benchmark_babi20_external.json", report.as_payload())
    if args.dialog_path:
        report = harness.run_external_dialog_suite(args.dialog_path, suite_kind=P17SuiteKind.DIALOG_BABI_6, max_examples_per_task=args.max_examples_per_task)
        external_reports["dialog_babi_1_6"] = report.as_payload()
        _write_json(artifacts / "official_benchmark_dialog_babi_external.json", report.as_payload())
    if args.modified_dialog_path:
        report = harness.run_external_dialog_suite(args.modified_dialog_path, suite_kind=P17SuiteKind.MODIFIED_DIALOG_BABI, max_examples_per_task=args.max_examples_per_task)
        external_reports["modified_dialog_babi"] = report.as_payload()
        _write_json(artifacts / "official_benchmark_modified_dialog_babi_external.json", report.as_payload())
    if args.permuted_dialog_path:
        report = harness.run_external_dialog_suite(args.permuted_dialog_path, suite_kind=P17SuiteKind.PERMUTED_DIALOG_BABI, max_examples_per_task=args.max_examples_per_task)
        external_reports["permuted_dialog_babi"] = report.as_payload()
        _write_json(artifacts / "official_benchmark_permuted_dialog_babi_external.json", report.as_payload())

    summary = {
        "schema_version": "htce-p17-official-benchmark-summary-v1",
        "release_smoke_matrix_passed": matrix.passed,
        "release_smoke_row_count": matrix.total_count,
        "external_reports_present": sorted(external_reports.keys()),
        "trace_head": matrix.trace_head,
        "spec_count": len(spec_payload["specs"]),
    }
    _write_json(artifacts / "official_benchmark_summary.json", summary)

    print("P17 official benchmark harness")
    print(f"spec_count: {len(spec_payload['specs'])}")
    print(f"release_smoke_rows: {matrix.total_count}")
    print(f"release_smoke_passed: {matrix.passed}")
    print(f"trace_head: {matrix.trace_head}")
    print(f"artifacts_dir: {artifacts}")
    return 0 if matrix.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
