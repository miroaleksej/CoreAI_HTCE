from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_active_scripts_are_pipeline_organized() -> None:
    expected = {
        "scripts/00_gates/scan_float_literals.py",
        "scripts/00_gates/check_invariants.py",
        "scripts/00_gates/check_version_sync.py",
        "scripts/01_sanity/run_organism.py",
        "scripts/01_sanity/run_active_agent.py",
        "scripts/02_benchmarks/prepare_training_data.py",
        "scripts/02_benchmarks/run_benchmark.py",
        "scripts/02_benchmarks/run_official_harness.py",
        "scripts/02_benchmarks/run_no_leakage.py",
        "scripts/03_topology_and_hardware/run_topology_acceptance.py",
        "scripts/03_topology_and_hardware/generate_interaction_report.py",
        "scripts/03_topology_and_hardware/run_hardware_width.py",
        "scripts/04_stability/run_long_run_stability.py",
        "scripts/05_artifacts/export_artifacts.py",
        "scripts/05_artifacts/run_acceptance.py",
        "scripts/05_artifacts/verify_manifest.py",
        "scripts/05_artifacts/verify_trace.py",
    }
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / "scripts").rglob("*.py")}
    assert actual == expected


def test_deprecated_flat_entrypoints_are_removed() -> None:
    assert not (ROOT / "scripts" / "run_smoke.py").exists()
    assert not (ROOT / "scripts" / "run_closed_loop.py").exists()


def test_pipeline_tree_make_target_passes() -> None:
    subprocess.run(["make", "pipeline-tree"], cwd=ROOT, check=True)
