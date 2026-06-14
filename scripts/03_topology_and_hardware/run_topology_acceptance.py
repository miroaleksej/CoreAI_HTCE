#!/usr/bin/env python3
"""Run P15 offline topology acceptance and write canonical JSON artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.topology.acceptance import run_topology_acceptance


def main() -> None:
    summary = run_topology_acceptance(artifacts_dir=ARTIFACTS, steps=8, modulus=Q256_MODULUS)
    print("topology_acceptance: PASS" if summary["topology_acceptance_passed"] else "topology_acceptance: FAIL")
    print("artifacts/topology_acceptance_l1.json")
    print("artifacts/topology_acceptance_l2.json")
    print("artifacts/topology_acceptance_l3.json")
    print("artifacts/topology_acceptance_world.json")
    print("artifacts/topology_acceptance_cross_layer.json")


if __name__ == "__main__":
    main()
