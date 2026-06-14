import ast
import json
import subprocess
import sys
from pathlib import Path

from htce_origin import __version__
from htce_origin.body.runtime import HTCERuntime
from htce_origin.cognition.world import Q16WorldModel, Q256WorldAction, Q256WorldModel
from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.kernel.serialization import verify_release_manifest_self_hash

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_p12_version_and_q256_world_default():
    assert __version__.startswith("1.0.0-")
    model = Q256WorldModel(dimension=2)
    assert model.modulus == Q256_MODULUS
    assert Q16WorldModel is Q256WorldModel
    action = Q256WorldAction("step", (1, 2))
    assert action.modulus == Q256_MODULUS


def test_no_runtime_placeholder_or_stub_terms():
    banned = ("placeholder", "stub", "mock")
    scan_paths = list((ROOT / "htce_origin").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    hits = []
    for path in scan_paths:
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            if word in text:
                hits.append((path.relative_to(ROOT).as_posix(), word))
    assert hits == []


def test_no_float_literals_in_htce_origin():
    offenders = []
    for path in (ROOT / "htce_origin").rglob("*.py"):
        count = _float_constant_count(path)
        if count:
            offenders.append((path.relative_to(ROOT).as_posix(), count))
    assert offenders == []


def test_capability_matrix_is_machine_readable_q256():
    payload = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
    assert payload["version"].startswith("1.0.0-")
    assert payload["release_line"] == "v1.0_final_math_q256_clean"
    assert payload["modulus"] == "2^256"
    assert payload["integer_only_runtime"] is True
    assert payload["verified_capabilities"]["l1_l2_l3_toroidal_state"] is True
    assert payload["verified_capabilities"]["continual_multitask_no_cross_domain_regression"] is True
    assert payload["verified_capabilities"]["external_shaped_revalidation_no_leakage"] is True
    assert payload["claim_boundary"]["consciousness_claimed"] is False


def test_release_manifest_self_hash_protocol():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert verify_release_manifest_self_hash(manifest)


def test_artifact_export_scripts_generate_canonical_files(tmp_path):
    subprocess.run([sys.executable, "scripts/05_artifacts/export_artifacts.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/03_topology_and_hardware/generate_interaction_report.py"], cwd=ROOT, check=True)
    export_path = ROOT / "artifacts" / "closed_loop_trace_export.json"
    report_path = ROOT / "artifacts" / "l1_l2_l3_interaction_report.json"
    assert export_path.exists()
    assert report_path.exists()
    export = json.loads(export_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert export["modulus"] == Q256_MODULUS
    assert export["trace_verified"] is True
    assert report["integer_only"] is True
    assert isinstance(report["l3"]["topology_beta0"], int)
    assert isinstance(report["l3"]["topology_beta1"], int)


def test_runtime_export_contains_q256_l1_l2_l3_contract():
    runtime = HTCERuntime()
    runtime.wake()
    runtime.run_closed_loop_simulation(steps=4)
    state = runtime.export_state()
    assert state["body"]["modulus"] == Q256_MODULUS
    assert "l2_working_memory" in state["body"]
    assert state["topology_guard"]["l2_window_is_clean_state"] is True
    assert state["world_model"]["modulus"] == Q256_MODULUS
