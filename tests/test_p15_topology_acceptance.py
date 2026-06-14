import json
import subprocess
import sys
from pathlib import Path

from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.topology.acceptance import (
    analyze_cross_layer_consistency,
    analyze_topology_window,
    build_acceptance_windows,
    run_topology_acceptance,
    vietoris_rips_2_skeleton_scan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p15_vr_2_skeleton_scan_returns_integer_betti() -> None:
    modulus = 1 << 16
    points = (
        (0, 0),
        (modulus // 8, 0),
        (0, modulus // 8),
        (modulus // 8, modulus // 8),
    )
    report = vietoris_rips_2_skeleton_scan(points, epsilon=modulus // 4, modulus=modulus)
    assert isinstance(report.beta0, int)
    assert isinstance(report.beta1, int)
    assert isinstance(report.beta2_2_skeleton, int)
    assert report.point_count == 4
    assert report.passed_bounds is True


def test_p15_builds_l1_l2_l3_world_windows_q256() -> None:
    windows = build_acceptance_windows(steps=6, dimension=4, modulus=Q256_MODULUS)
    assert set(windows) == {"l1", "l2", "l3", "world"}
    assert len(windows["l1"]) == 6
    assert len(windows["l2"]) >= 6
    assert len(windows["l3"]) == 6
    assert len(windows["world"]) == 6
    for points in windows.values():
        for point in points:
            assert point.modulus == Q256_MODULUS
            assert point.dimension == 4


def test_p15_analyzes_windows_and_cross_layer_consistency() -> None:
    windows = build_acceptance_windows(steps=5, dimension=3, modulus=Q256_MODULUS)
    l1_report = analyze_topology_window("L1", windows["l1"], artifact_name="topology_acceptance_l1.json", modulus=Q256_MODULUS)
    assert l1_report.passed is True
    assert l1_report.scale_reports
    assert all(isinstance(item.beta0, int) for item in l1_report.scale_reports)
    cross = analyze_cross_layer_consistency(windows, modulus=Q256_MODULUS)
    assert cross.same_dimension is True
    assert cross.same_modulus is True
    assert cross.passed is True
    assert all(isinstance(value, int) for value in cross.cross_layer_distances_raw.values())


def test_p15_script_generates_required_artifacts() -> None:
    subprocess.run([sys.executable, "scripts/03_topology_and_hardware/run_topology_acceptance.py"], cwd=ROOT, check=True)
    required = (
        "topology_acceptance_l1.json",
        "topology_acceptance_l2.json",
        "topology_acceptance_l3.json",
        "topology_acceptance_world.json",
        "topology_acceptance_cross_layer.json",
        "topology_acceptance_summary.json",
    )
    for name in required:
        path = ROOT / "artifacts" / name
        assert path.exists(), name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["integer_only"] is True
        assert "artifact_sha256" in payload
    summary = json.loads((ROOT / "artifacts" / "topology_acceptance_summary.json").read_text(encoding="utf-8"))
    assert summary["topology_acceptance_passed"] is True
    assert summary["runtime_fast_guard_unchanged"] is True
