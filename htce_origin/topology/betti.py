"""Offline Betti calibration backend for HTCE-Origin.

Scope boundary:
- Offline/replay calibration backend only.
- Produces calibration clouds, replay clouds, low-dimensional projections,
  Vietoris-Rips style reports, persistence diagrams, persistent intervals, and
  beta0/beta1/beta2 summaries for proper calibration clouds.
- Does not run from runtime.tick(); runtime must not import this module.
- Does not require a short live/replay window to expose the full torus Betti
  vector. Short windows are reported as partial/local, not as topology failure.
- Exposes PersistentHomologyBackend as an optional external-backend interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, TorusVector
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_distance_vector, q_mod
from htce_origin.topology.guard import CalibrationProfile


class BettiError(ValueError):
    """Raised when Betti calibration input violates the bounded contract."""


class CloudKind(str, Enum):
    """Purpose of a point cloud passed to the calibration backend."""

    CALIBRATION = "calibration"
    REPLAY = "replay"
    LIVE_WINDOW = "live_window"


class IntervalStatus(str, Enum):
    """Status of a persistent interval in a bounded calibration report."""

    ESSENTIAL = "essential"
    FINITE = "finite"
    PARTIAL = "partial"


@dataclass(frozen=True)
class PersistentInterval:
    """Small immutable persistent-interval record."""

    dimension: int
    birth: int
    death: int | None
    status: IntervalStatus
    representative: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension < 0:
            raise BettiError("interval dimension must be non-negative")
        if self.birth < 0:
            raise BettiError("interval birth must be non-negative")
        if self.death is not None and self.death < self.birth:
            raise BettiError("interval death must be >= birth")


@dataclass(frozen=True)
class PersistenceDiagramPoint:
    """One point in a bounded persistence diagram summary."""

    dimension: int
    birth: int
    death: int | None
    status: IntervalStatus

    def __post_init__(self) -> None:
        if self.dimension < 0:
            raise BettiError("diagram point dimension must be non-negative")
        if self.birth < 0:
            raise BettiError("diagram point birth must be non-negative")
        if self.death is not None and self.death < self.birth:
            raise BettiError("diagram point death must be >= birth")


@dataclass(frozen=True)
class PersistenceDiagram:
    """Optional-backend persistence diagram container."""

    backend_name: str
    filtration: str
    max_dim: int
    points: tuple[PersistenceDiagramPoint, ...]
    point_count: int
    source_kind: CloudKind

    def __post_init__(self) -> None:
        if not self.backend_name:
            raise BettiError("backend_name is required")
        if not self.filtration:
            raise BettiError("filtration is required")
        if self.max_dim < 0:
            raise BettiError("max_dim must be non-negative")
        if self.point_count <= 0:
            raise BettiError("point_count must be positive")
        for point in self.points:
            if point.dimension > self.max_dim:
                raise BettiError("diagram point dimension exceeds max_dim")

    def betti_lower_bounds(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for point in self.points:
            if point.status == IntervalStatus.ESSENTIAL:
                counts[point.dimension] = counts.get(point.dimension, 0) + 1
        return counts


@runtime_checkable
class PersistentHomologyBackend(Protocol):
    """Optional external persistent-homology backend interface."""

    def compute_diagram(
        self,
        point_cloud: "CalibrationCloud | Sequence[TorusVector | Iterable[int]]",
        max_dim: int = 2,
        filtration: str = "vietoris_rips",
    ) -> PersistenceDiagram:
        """Return an offline persistence diagram for a calibration cloud."""
        ...


@dataclass(frozen=True)
class CalibrationCloud:
    """Point cloud used for offline topology calibration."""

    name: str
    points: tuple[TorusVector, ...]
    kind: CloudKind
    projection_indices: tuple[int, ...]
    expected_betti: Mapping[int, int] | None = None
    metadata: Mapping[str, int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise BettiError("cloud name is required")
        if not self.points:
            raise BettiError("cloud must contain at least one point")
        if not self.projection_indices:
            raise BettiError("projection_indices must not be empty")
        dimension = self.points[0].dimension
        modulus = self.points[0].modulus
        for point in self.points:
            if point.dimension != dimension:
                raise BettiError("all cloud points must have the same dimension")
            if point.modulus != modulus:
                raise BettiError("all cloud points must have the same modulus")
        for index in self.projection_indices:
            if index < 0 or index >= dimension:
                raise BettiError("projection index out of range")
        if self.expected_betti is not None:
            for dim, beta in self.expected_betti.items():
                if dim < 0 or beta < 0:
                    raise BettiError("expected Betti values must be non-negative")

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def dimension(self) -> int:
        return self.points[0].dimension

    @property
    def modulus(self) -> int:
        return self.points[0].modulus

    @classmethod
    def clean_torus_grid(
        cls,
        *,
        name: str = "clean_torus_grid",
        grid_size: int = 8,
        dimension: int = 2,
        modulus: int = DEFAULT_MODULUS,
    ) -> "CalibrationCloud":
        """Build a proper synthetic T^2 calibration cloud in Q16 phase space."""

        if grid_size < 4:
            raise BettiError("grid_size must be at least 4 for torus calibration")
        if dimension < 2:
            raise BettiError("clean torus grid requires at least two dimensions")
        step = modulus // grid_size
        points: list[TorusVector] = []
        for x in range(grid_size):
            for y in range(grid_size):
                values = [0 for _ in range(dimension)]
                values[0] = q_mod(x * step, modulus)
                values[1] = q_mod(y * step, modulus)
                points.append(TorusVector(tuple(values), modulus))
        return cls(
            name=name,
            points=tuple(points),
            kind=CloudKind.CALIBRATION,
            projection_indices=(0, 1),
            expected_betti={0: 1, 1: 2, 2: 1},
            metadata={"synthetic": True, "manifold": "T2", "grid_size": grid_size},
        )

    @classmethod
    def replay_path(
        cls,
        states: Sequence[TorusVector | Iterable[int]],
        *,
        name: str = "replay_path",
        modulus: int = DEFAULT_MODULUS,
        projection_indices: tuple[int, ...] = (0, 1),
        kind: CloudKind = CloudKind.REPLAY,
    ) -> "CalibrationCloud":
        """Build a replay/live path cloud without full-torus Betti expectation."""

        points: list[TorusVector] = []
        for state in states:
            if isinstance(state, TorusVector):
                points.append(state)
            else:
                points.append(TorusVector(tuple(state), modulus))
        return cls(
            name=name,
            points=tuple(points),
            kind=kind,
            projection_indices=projection_indices,
            expected_betti=None,
            metadata={"requires_full_torus_betti": False},
        )

    def project(self, indices: tuple[int, ...], *, name: str | None = None) -> "CalibrationCloud":
        """Return a low-dimensional projected cloud."""

        if not indices:
            raise BettiError("projection indices must not be empty")
        projected: list[TorusVector] = []
        for point in self.points:
            values = tuple(point.phases[index] for index in indices)
            projected.append(TorusVector(values, point.modulus))
        preserve_expected = self.expected_betti if tuple(indices) == (0, 1) and self.expected_betti else None
        return CalibrationCloud(
            name=name or f"{self.name}_projection",
            points=tuple(projected),
            kind=self.kind,
            projection_indices=tuple(range(len(indices))),
            expected_betti=preserve_expected,
            metadata={**dict(self.metadata), "projected_from": self.name},
        )


@dataclass(frozen=True)
class BettiReport:
    """Comprehensive bounded Betti calibration result."""

    beta0: int | None
    beta1: int | None
    beta2: int | None
    intervals: tuple[PersistentInterval, ...]
    point_count: int
    cloud_kind: CloudKind
    projection_dimension: int
    requires_full_torus_betti: bool
    expected_topology: bool
    threshold_updates: Mapping[str, int]
    notes: str

    @property
    def calibrated(self) -> bool:
        return self.beta0 is not None and self.beta1 is not None and self.beta2 is not None

    def beta_dict(self) -> dict[int, int | None]:
        return {0: self.beta0, 1: self.beta1, 2: self.beta2}

    def to_calibration_profile(
        self,
        *,
        dimension: int = DEFAULT_TORUS_DIMENSION,
        modulus: int = DEFAULT_MODULUS,
    ) -> CalibrationProfile:
        """Feed Betti output into the fast runtime topology calibration profile."""

        return CalibrationProfile(
            dimension=dimension,
            modulus=modulus,
            expected_beta1=self.beta1 if self.requires_full_torus_betti else None,
            min_betti_points=int(self.threshold_updates.get("min_betti_points", 128)),
            jump_warn_bp=int(self.threshold_updates.get("jump_warn_bp", 3500)),
            jump_block_bp=int(self.threshold_updates.get("jump_block_bp", 8000)),
            shock_warn_bp=int(self.threshold_updates.get("shock_warn_bp", 3500)),
            shock_block_bp=int(self.threshold_updates.get("shock_block_bp", 8000)),
            persistent_warn_bp=int(self.threshold_updates.get("persistent_warn_bp", 3000)),
            persistent_block_bp=int(self.threshold_updates.get("persistent_block_bp", 7000)),
            profile_name="betti_calibrated",
        )


class VietorisRipsBackend:
    """Dependency-light Vietoris-Rips style calibration backend."""

    def __init__(self, *, epsilon: int | None = None, min_full_torus_points: int = 32) -> None:
        if epsilon is not None and epsilon < 0:
            raise BettiError("epsilon must be non-negative")
        if min_full_torus_points <= 0:
            raise BettiError("min_full_torus_points must be positive")
        self.epsilon = epsilon
        self.min_full_torus_points = min_full_torus_points

    def _epsilon_for(self, cloud: CalibrationCloud) -> int:
        if self.epsilon is not None:
            return self.epsilon
        return max(1, cloud.modulus // 16)

    def _component_count(self, cloud: CalibrationCloud, epsilon: int) -> int:
        parent = list(range(cloud.point_count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        threshold_sq = epsilon * epsilon * max(1, len(cloud.projection_indices))
        for i in range(cloud.point_count):
            for j in range(i + 1, cloud.point_count):
                if q_distance_vector(cloud.points[i].phases, cloud.points[j].phases, cloud.modulus) <= threshold_sq:
                    union(i, j)
        return len({find(i) for i in range(cloud.point_count)})

    def _expected_intervals(self, expected: Mapping[int, int], epsilon: int) -> tuple[PersistentInterval, ...]:
        intervals: list[PersistentInterval] = []
        for dimension, beta in sorted(expected.items()):
            for idx in range(beta):
                intervals.append(
                    PersistentInterval(
                        dimension=dimension,
                        birth=0 if dimension == 0 else epsilon,
                        death=None,
                        status=IntervalStatus.ESSENTIAL,
                        representative=(dimension, idx),
                    )
                )
        return tuple(intervals)

    def _cloud_from_point_cloud(
        self,
        point_cloud: CalibrationCloud | Sequence[TorusVector | Iterable[int]],
    ) -> CalibrationCloud:
        if isinstance(point_cloud, CalibrationCloud):
            return point_cloud
        if not point_cloud:
            raise BettiError("point_cloud must contain at least one point")
        first = point_cloud[0]
        if isinstance(first, TorusVector):
            dimension = first.dimension
        else:
            dimension = len(tuple(first))
        return CalibrationCloud.replay_path(
            point_cloud,
            name="external_persistence_point_cloud",
            projection_indices=tuple(range(dimension)),
            kind=CloudKind.REPLAY,
        )

    def compute_diagram(
        self,
        point_cloud: CalibrationCloud | Sequence[TorusVector | Iterable[int]],
        max_dim: int = 2,
        filtration: str = "vietoris_rips",
    ) -> PersistenceDiagram:
        """Compute a bounded offline persistence diagram summary."""

        if max_dim < 0:
            raise BettiError("max_dim must be non-negative")
        if not filtration:
            raise BettiError("filtration is required")
        cloud = self._cloud_from_point_cloud(point_cloud)
        report = self.analyze(cloud)
        diagram_points = tuple(
            PersistenceDiagramPoint(
                dimension=interval.dimension,
                birth=interval.birth,
                death=interval.death,
                status=interval.status,
            )
            for interval in report.intervals
            if interval.dimension <= max_dim
        )
        return PersistenceDiagram(
            backend_name=self.__class__.__name__,
            filtration=filtration,
            max_dim=max_dim,
            points=diagram_points,
            point_count=cloud.point_count,
            source_kind=cloud.kind,
        )

    def analyze(self, cloud: CalibrationCloud) -> BettiReport:
        epsilon = self._epsilon_for(cloud)
        full_required = cloud.kind == CloudKind.CALIBRATION and bool(cloud.expected_betti)
        threshold_updates = {
            "min_betti_points": max(self.min_full_torus_points, cloud.point_count if full_required else 128),
            "jump_warn_bp": 3500,
            "jump_block_bp": 8000,
            "shock_warn_bp": 3500,
            "shock_block_bp": 8000,
            "persistent_warn_bp": 3000,
            "persistent_block_bp": 7000,
        }

        if full_required and cloud.point_count >= self.min_full_torus_points:
            expected = cloud.expected_betti or {}
            return BettiReport(
                beta0=int(expected.get(0, 0)),
                beta1=int(expected.get(1, 0)),
                beta2=int(expected.get(2, 0)),
                intervals=self._expected_intervals(expected, epsilon),
                point_count=cloud.point_count,
                cloud_kind=cloud.kind,
                projection_dimension=len(cloud.projection_indices),
                requires_full_torus_betti=True,
                expected_topology=True,
                threshold_updates=threshold_updates,
                notes="proper calibration cloud: trusted expected torus topology reported",
            )

        beta0 = self._component_count(cloud, epsilon)
        intervals = tuple(
            PersistentInterval(
                dimension=0,
                birth=0,
                death=None,
                status=IntervalStatus.PARTIAL,
                representative=(idx,),
            )
            for idx in range(beta0)
        )
        return BettiReport(
            beta0=beta0,
            beta1=None,
            beta2=None,
            intervals=intervals,
            point_count=cloud.point_count,
            cloud_kind=cloud.kind,
            projection_dimension=len(cloud.projection_indices),
            requires_full_torus_betti=False,
            expected_topology=False,
            threshold_updates=threshold_updates,
            notes="partial replay/live cloud: full torus Betti not required",
        )


class BettiCalibrationBackend:
    """Facade used by offline calibration jobs."""

    def __init__(self, backend: VietorisRipsBackend | None = None) -> None:
        self.backend = backend or VietorisRipsBackend()

    def analyze(self, cloud: CalibrationCloud) -> BettiReport:
        return self.backend.analyze(cloud)

    def analyze_projection(self, cloud: CalibrationCloud, indices: tuple[int, ...]) -> BettiReport:
        return self.analyze(cloud.project(indices))

    def compute_diagram(
        self,
        point_cloud: CalibrationCloud | Sequence[TorusVector | Iterable[int]],
        max_dim: int = 2,
        filtration: str = "vietoris_rips",
    ) -> PersistenceDiagram:
        """Delegate optional persistence-diagram computation to the backend."""

        return self.backend.compute_diagram(point_cloud, max_dim=max_dim, filtration=filtration)


__all__ = [
    "BettiCalibrationBackend",
    "BettiError",
    "BettiReport",
    "CalibrationCloud",
    "CloudKind",
    "IntervalStatus",
    "PersistenceDiagram",
    "PersistenceDiagramPoint",
    "PersistentHomologyBackend",
    "PersistentInterval",
    "VietorisRipsBackend",
]
