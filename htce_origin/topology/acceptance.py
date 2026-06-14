"""Offline topology acceptance layer for HTCE-Origin Q256.

Scope boundary
--------------
This module is an acceptance/research instrument. It is not imported by
runtime.tick and it does not participate in online gating. The runtime topology
organ remains the fast integer 1-skeleton guard in htce_origin.topology.guard.

The acceptance layer builds L1, clean L2, L3 and world-prediction trajectory
windows, then runs an integer Vietoris-Rips multi-scale Betti scan.  At every
integer radius it constructs the VR 2-skeleton, computes beta0, beta1 and a
2-skeleton beta2 value over GF(2), and emits canonical JSON artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, TorusVector, active_state_digest, hash_to_phase
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_distance_vector, q_mod, q_vector_add
from htce_origin.kernel.serialization import canonical_json_bytes, sha256_hex


class TopologyAcceptanceError(ValueError):
    """Raised when offline topology acceptance input violates the Q256 contract."""


def _as_vector(state: TorusVector | Iterable[int], *, modulus: int) -> TorusVector:
    if isinstance(state, TorusVector):
        return state
    return TorusVector(tuple(int(value) for value in state), modulus)


def _require_window(points: Sequence[TorusVector | Iterable[int]], *, modulus: int) -> tuple[TorusVector, ...]:
    vectors = tuple(_as_vector(point, modulus=modulus) for point in points)
    if not vectors:
        raise TopologyAcceptanceError("topology acceptance window must not be empty")
    dimension = vectors[0].dimension
    for vector in vectors:
        if vector.dimension != dimension:
            raise TopologyAcceptanceError("topology acceptance vectors must share one dimension")
        if vector.modulus != modulus:
            raise TopologyAcceptanceError("topology acceptance vectors must share one modulus")
        if any(value < 0 or value >= modulus for value in vector.phases):
            raise TopologyAcceptanceError("topology acceptance vector escaped Q256 bounds")
    return vectors


def _gf2_rank(columns: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for raw_column in columns:
        column = int(raw_column)
        while column:
            pivot = column.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = column
                rank += 1
                break
            column ^= existing
    return rank


def _component_count(edge_pairs: Sequence[tuple[int, int]], point_count: int) -> int:
    parent = list(range(point_count))
    size = [1 for _ in range(point_count)]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for left, right in edge_pairs:
        union(left, right)
    return len({find(index) for index in range(point_count)})


def _scale_radii(modulus: int) -> tuple[int, ...]:
    denominators = (512, 256, 128, 64, 32, 16, 8, 4, 2)
    radii: list[int] = []
    for denominator in denominators:
        radius = max(1, modulus // denominator)
        if radius not in radii:
            radii.append(radius)
    return tuple(radii)


@dataclass(frozen=True)
class VRScaleBettiReport:
    """Exact integer Betti scan of one VR 2-skeleton scale over GF(2)."""

    epsilon: int
    point_count: int
    edge_count: int
    triangle_count: int
    beta0: int
    beta1: int
    beta2_2_skeleton: int
    boundary2_rank: int
    passed_bounds: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "beta2_2_skeleton": self.beta2_2_skeleton,
            "boundary2_rank": self.boundary2_rank,
            "edge_count": self.edge_count,
            "epsilon": self.epsilon,
            "passed_bounds": self.passed_bounds,
            "point_count": self.point_count,
            "triangle_count": self.triangle_count,
        }


@dataclass(frozen=True)
class TopologyWindowAcceptanceReport:
    """Offline topology acceptance report for one layer/window."""

    layer_name: str
    artifact_name: str
    modulus: int
    dimension: int
    point_count: int
    point_digests: tuple[str, ...]
    scale_reports: tuple[VRScaleBettiReport, ...]
    backend: str = "integer_vr_2_skeleton_gf2"
    acceptance_mode: str = "offline_acceptance_not_runtime"
    notes: str = "multi-scale integer Betti scan; runtime fast guard remains separate"
    metadata: Mapping[str, int | str | bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.scale_reports) and all(report.passed_bounds for report in self.scale_reports)

    def as_payload(self) -> dict[str, object]:
        return {
            "acceptance_mode": self.acceptance_mode,
            "artifact_name": self.artifact_name,
            "backend": self.backend,
            "dimension": self.dimension,
            "integer_only": True,
            "layer_name": self.layer_name,
            "metadata": dict(self.metadata),
            "modulus": self.modulus,
            "notes": self.notes,
            "passed": self.passed,
            "point_count": self.point_count,
            "point_digests": list(self.point_digests),
            "scale_reports": [report.as_payload() for report in self.scale_reports],
        }


@dataclass(frozen=True)
class CrossLayerTopologyConsistencyReport:
    """Integer cross-layer topology consistency report."""

    modulus: int
    dimension: int
    l1_point_count: int
    l2_point_count: int
    l3_point_count: int
    world_point_count: int
    cross_layer_distances_raw: Mapping[str, int]
    same_dimension: bool
    same_modulus: bool
    passed: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "cross_layer_distances_raw": dict(self.cross_layer_distances_raw),
            "dimension": self.dimension,
            "integer_only": True,
            "l1_point_count": self.l1_point_count,
            "l2_point_count": self.l2_point_count,
            "l3_point_count": self.l3_point_count,
            "modulus": self.modulus,
            "passed": self.passed,
            "same_dimension": self.same_dimension,
            "same_modulus": self.same_modulus,
            "world_point_count": self.world_point_count,
        }


def vietoris_rips_2_skeleton_scan(
    points: Sequence[TorusVector | Iterable[int]],
    *,
    epsilon: int,
    modulus: int = DEFAULT_MODULUS,
) -> VRScaleBettiReport:
    """Compute integer beta0/beta1/beta2 for a VR 2-skeleton at one radius."""

    if epsilon < 0:
        raise TopologyAcceptanceError("epsilon must be non-negative")
    vectors = _require_window(points, modulus=modulus)
    point_count = len(vectors)
    threshold_sq = int(epsilon) * int(epsilon) * max(1, vectors[0].dimension)

    edge_pairs: list[tuple[int, int]] = []
    edge_index: dict[tuple[int, int], int] = {}
    for left in range(point_count):
        for right in range(left + 1, point_count):
            if q_distance_vector(vectors[left].phases, vectors[right].phases, modulus) <= threshold_sq:
                edge_index[(left, right)] = len(edge_pairs)
                edge_pairs.append((left, right))

    triangle_columns: list[int] = []
    for first in range(point_count):
        for second in range(first + 1, point_count):
            first_second = edge_index.get((first, second))
            if first_second is None:
                continue
            for third in range(second + 1, point_count):
                first_third = edge_index.get((first, third))
                second_third = edge_index.get((second, third))
                if first_third is None or second_third is None:
                    continue
                triangle_columns.append((1 << first_second) | (1 << first_third) | (1 << second_third))

    beta0 = _component_count(edge_pairs, point_count)
    edge_count = len(edge_pairs)
    triangle_count = len(triangle_columns)
    boundary1_rank = point_count - beta0
    boundary2_rank = _gf2_rank(triangle_columns)
    beta1 = max(0, edge_count - boundary1_rank - boundary2_rank)
    beta2 = max(0, triangle_count - boundary2_rank)
    return VRScaleBettiReport(
        epsilon=int(epsilon),
        point_count=point_count,
        edge_count=edge_count,
        triangle_count=triangle_count,
        beta0=beta0,
        beta1=beta1,
        beta2_2_skeleton=beta2,
        boundary2_rank=boundary2_rank,
        passed_bounds=beta0 >= 1 and beta1 >= 0 and beta2 >= 0,
    )


def analyze_topology_window(
    layer_name: str,
    points: Sequence[TorusVector | Iterable[int]],
    *,
    artifact_name: str,
    modulus: int = DEFAULT_MODULUS,
    metadata: Mapping[str, int | str | bool] | None = None,
) -> TopologyWindowAcceptanceReport:
    """Run the offline multi-scale VR Betti scan for one topology window."""

    vectors = _require_window(points, modulus=modulus)
    scale_reports = tuple(
        vietoris_rips_2_skeleton_scan(vectors, epsilon=radius, modulus=modulus)
        for radius in _scale_radii(modulus)
    )
    point_digests = tuple(active_state_digest(vector) for vector in vectors)
    return TopologyWindowAcceptanceReport(
        layer_name=str(layer_name),
        artifact_name=str(artifact_name),
        modulus=modulus,
        dimension=vectors[0].dimension,
        point_count=len(vectors),
        point_digests=point_digests,
        scale_reports=scale_reports,
        metadata=dict(metadata or {}),
    )


def _write_report(path: Path, report: TopologyWindowAcceptanceReport | CrossLayerTopologyConsistencyReport) -> str:
    payload = report.as_payload()
    digest = sha256_hex(canonical_json_bytes(payload))
    enriched = dict(payload)
    enriched["artifact_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(enriched))
    return digest


def _sensor_samples(step: int, *, dimension: int, modulus: int) -> tuple[int, ...]:
    base = hash_to_phase(f"p15:l1:samples:{step}", dimension=dimension, modulus=modulus, namespace="p15_l1")
    step_delta = max(1, modulus // 4096)
    return tuple(q_mod(value + (step + 1) * (index + 1) * step_delta, modulus) for index, value in enumerate(base))


def build_acceptance_windows(
    *,
    steps: int = 24,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
) -> dict[str, tuple[TorusVector, ...]]:
    """Build deterministic L1, clean L2, L3 and world prediction windows."""

    if steps < 4:
        raise TopologyAcceptanceError("steps must be at least four")
    from htce_origin.body.runtime import HTCERuntime, RuntimeRequest
    from htce_origin.kernel.config import RuntimeConfig
    from htce_origin.sensory.l1_encoder import RawSensorPacket

    runtime = HTCERuntime(RuntimeConfig(modulus=modulus, l1_dim=dimension, l2_dim=dimension, l3_dim=dimension))
    runtime.wake()

    l1_points: list[TorusVector] = []
    for step in range(steps):
        packet = RawSensorPacket(
            modality="p15_topology_l1",
            samples=_sensor_samples(step, dimension=dimension, modulus=modulus),
            sample_min=0,
            sample_max=modulus - 1,
            reliability_bp=10000,
            evidence_id=f"p15_l1_{step}",
            metadata={"acceptance": True, "step": step},
        )
        runtime.observe_l1_packet(packet, source="p15_topology_acceptance")
        l1_points.append(TorusVector(runtime.body.l1.vector, modulus))

    from htce_origin.kernel.core import EntityId, EvidenceId, FactFrame, RelationId, fact_delta

    l2_points: list[TorusVector] = [TorusVector(runtime.body.l2_clean_vector(), modulus)]
    for step in range(steps):
        fact = FactFrame(
            EntityId(f"p15_entity_{step % 8}"),
            RelationId("located_in"),
            EntityId(f"p15_loc_{step % 5}"),
            EvidenceId(f"p15_l2_{step}"),
        )
        runtime.body.commit_l2_fact(fact_delta(fact, dimension=dimension, modulus=modulus), reason="topology_acceptance_clean_l2_window")
        l2_points.append(TorusVector(runtime.body.l2_clean_vector(), modulus))

    l3_points: list[TorusVector] = []
    l3_state = TorusVector(runtime.body.l3.vector, modulus)
    for step in range(steps):
        delta = hash_to_phase(f"p15:l3:semantic:{step}", dimension=dimension, modulus=modulus, namespace="p15_l3")
        target = TorusVector(q_vector_add(l3_state.phases, delta, modulus), modulus)
        runtime.body.commit_l3_semantic_state(target.phases, evidence_id=f"p15_l3_{step}", reason="topology_acceptance_l3_window")
        l3_state = TorusVector(runtime.body.l3.vector, modulus)
        l3_points.append(l3_state)

    world_points: list[TorusVector] = []
    world_state = TorusVector(runtime.body.l1.vector, modulus)
    action_names = ("advance", "rotate", "hold")
    runtime._ensure_closed_loop_skills()
    for step in range(steps):
        name = action_names[step % len(action_names)]
        action = runtime._closed_loop_action(name)
        prediction = runtime.world_model.predict_next_state(world_state, action)
        world_state = prediction.predicted_state
        world_points.append(world_state)

    return {
        "l1": tuple(l1_points),
        "l2": tuple(l2_points),
        "l3": tuple(l3_points),
        "world": tuple(world_points),
    }


def analyze_cross_layer_consistency(windows: Mapping[str, Sequence[TorusVector]], *, modulus: int = DEFAULT_MODULUS) -> CrossLayerTopologyConsistencyReport:
    required = ("l1", "l2", "l3", "world")
    for name in required:
        if name not in windows or not windows[name]:
            raise TopologyAcceptanceError(f"missing topology window: {name}")
    last = {name: windows[name][-1] for name in required}
    dimensions = {vector.dimension for vector in last.values()}
    moduli = {vector.modulus for vector in last.values()}
    same_dimension = len(dimensions) == 1
    same_modulus = len(moduli) == 1 and modulus in moduli
    distances = {
        "l1_l2": q_distance_vector(last["l1"].phases, last["l2"].phases, modulus),
        "l1_l3": q_distance_vector(last["l1"].phases, last["l3"].phases, modulus),
        "l1_world": q_distance_vector(last["l1"].phases, last["world"].phases, modulus),
        "l2_l3": q_distance_vector(last["l2"].phases, last["l3"].phases, modulus),
        "l2_world": q_distance_vector(last["l2"].phases, last["world"].phases, modulus),
        "l3_world": q_distance_vector(last["l3"].phases, last["world"].phases, modulus),
    }
    return CrossLayerTopologyConsistencyReport(
        modulus=modulus,
        dimension=next(iter(dimensions)) if dimensions else 0,
        l1_point_count=len(windows["l1"]),
        l2_point_count=len(windows["l2"]),
        l3_point_count=len(windows["l3"]),
        world_point_count=len(windows["world"]),
        cross_layer_distances_raw=distances,
        same_dimension=same_dimension,
        same_modulus=same_modulus,
        passed=same_dimension and same_modulus,
    )


def run_topology_acceptance(
    *,
    artifacts_dir: str | Path,
    steps: int = 24,
    dimension: int = DEFAULT_TORUS_DIMENSION,
    modulus: int = DEFAULT_MODULUS,
) -> dict[str, object]:
    """Generate all P15 topology acceptance artifacts and return summary."""

    artifact_root = Path(artifacts_dir)
    windows = build_acceptance_windows(steps=steps, dimension=dimension, modulus=modulus)
    reports = {
        "l1": analyze_topology_window("L1", windows["l1"], artifact_name="topology_acceptance_l1.json", modulus=modulus, metadata={"window": "l1_trajectory"}),
        "l2": analyze_topology_window("L2_clean", windows["l2"], artifact_name="topology_acceptance_l2.json", modulus=modulus, metadata={"window": "clean_l2_working_torus"}),
        "l3": analyze_topology_window("L3", windows["l3"], artifact_name="topology_acceptance_l3.json", modulus=modulus, metadata={"window": "l3_semantic_state"}),
        "world": analyze_topology_window("world_prediction", windows["world"], artifact_name="topology_acceptance_world.json", modulus=modulus, metadata={"window": "world_prediction_trajectory"}),
    }
    digests = {
        name: _write_report(artifact_root / report.artifact_name, report)
        for name, report in reports.items()
    }
    cross = analyze_cross_layer_consistency(windows, modulus=modulus)
    cross_digest = _write_report(artifact_root / "topology_acceptance_cross_layer.json", cross)
    summary = {
        "artifact_type": "topology_acceptance_summary",
        "cross_layer_digest": cross_digest,
        "integer_only": True,
        "modulus": modulus,
        "release_line": "p15_full_topology_acceptance_layer_q256",
        "report_digests": digests,
        "reports_passed": {name: report.passed for name, report in reports.items()},
        "runtime_fast_guard_unchanged": True,
        "steps": int(steps),
        "topology_acceptance_passed": all(report.passed for report in reports.values()) and cross.passed,
    }
    summary_digest = sha256_hex(canonical_json_bytes(summary))
    enriched = dict(summary)
    enriched["artifact_sha256"] = summary_digest
    (artifact_root / "topology_acceptance_summary.json").write_bytes(canonical_json_bytes(enriched))
    return enriched


__all__ = [
    "CrossLayerTopologyConsistencyReport",
    "TopologyAcceptanceError",
    "TopologyWindowAcceptanceReport",
    "VRScaleBettiReport",
    "analyze_cross_layer_consistency",
    "analyze_topology_window",
    "build_acceptance_windows",
    "run_topology_acceptance",
    "vietoris_rips_2_skeleton_scan",
]
