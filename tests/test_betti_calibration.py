from __future__ import annotations

import inspect

from htce_origin.topology.betti import (
    BettiCalibrationBackend,
    CalibrationCloud,
    CloudKind,
    IntervalStatus,
    PersistenceDiagram,
    PersistentHomologyBackend,
    VietorisRipsBackend,
)
from htce_origin.kernel.q16 import DEFAULT_MODULUS
from htce_origin.topology.guard import CalibrationProfile
import htce_origin.body.runtime as runtime


def test_clean_torus_calibration_gives_expected_topology_on_proper_cloud() -> None:
    cloud = CalibrationCloud.clean_torus_grid(grid_size=8, dimension=2)
    report = BettiCalibrationBackend(VietorisRipsBackend(min_full_torus_points=16)).analyze(cloud)

    assert report.calibrated is True
    assert report.beta0 == 1
    assert report.beta1 == 2
    assert report.beta2 == 1
    assert report.expected_topology is True
    assert report.requires_full_torus_betti is True
    assert any(interval.dimension == 1 and interval.status == IntervalStatus.ESSENTIAL for interval in report.intervals)


def test_short_live_window_does_not_require_full_torus_betti() -> None:
    short_path = CalibrationCloud.replay_path(
        [
            (0, 0),
            (64, 0),
            (128, 0),
            (192, 0),
        ],
        kind=CloudKind.LIVE_WINDOW,
    )
    report = BettiCalibrationBackend().analyze(short_path)

    assert report.requires_full_torus_betti is False
    assert report.beta0 is not None
    assert report.beta1 is None
    assert report.beta2 is None
    assert "full torus Betti not required" in report.notes


def test_low_dimensional_projection_preserves_proper_t2_report_for_canonical_projection() -> None:
    cloud = CalibrationCloud.clean_torus_grid(grid_size=8, dimension=4)
    report = BettiCalibrationBackend(VietorisRipsBackend(min_full_torus_points=16)).analyze_projection(cloud, (0, 1))

    assert report.projection_dimension == 2
    assert report.beta0 == 1
    assert report.beta1 == 2
    assert report.beta2 == 1


def test_replay_projection_drops_full_torus_expectation() -> None:
    cloud = CalibrationCloud.replay_path(
        [
            (0, 0, 0),
            (100, 0, 0),
            (200, 0, 0),
            (300, 0, 0),
        ],
        projection_indices=(0, 1, 2),
    )
    projected = cloud.project((0, 1))
    report = BettiCalibrationBackend().analyze(projected)

    assert projected.expected_betti is None
    assert report.requires_full_torus_betti is False
    assert report.beta1 is None


def test_betti_output_feeds_calibration_thresholds() -> None:
    cloud = CalibrationCloud.clean_torus_grid(grid_size=8, dimension=2)
    report = BettiCalibrationBackend(VietorisRipsBackend(min_full_torus_points=16)).analyze(cloud)
    profile = report.to_calibration_profile(dimension=2, modulus=DEFAULT_MODULUS)

    assert isinstance(profile, CalibrationProfile)
    assert profile.expected_beta1 == 2
    assert profile.min_betti_points >= cloud.point_count
    assert profile.jump_warn_bp < profile.jump_block_bp


def test_partial_replay_report_does_not_force_runtime_expected_beta1() -> None:
    cloud = CalibrationCloud.replay_path([(0, 0), (32, 0), (64, 0), (96, 0)])
    report = BettiCalibrationBackend().analyze(cloud)
    profile = report.to_calibration_profile(dimension=2)

    assert report.requires_full_torus_betti is False
    assert profile.expected_beta1 is None


def test_runtime_does_not_import_heavy_betti_each_tick() -> None:
    source = inspect.getsource(runtime)
    assert "import betti" not in source
    assert "from .betti" not in source


def test_betti_backend_has_no_runtime_float_constants() -> None:
    import htce_origin.topology.betti as betti

    source = inspect.getsource(betti)
    assert ".0" not in source
    assert ".5" not in source



def test_optional_persistent_homology_backend_interface_returns_diagram() -> None:
    backend: PersistentHomologyBackend = VietorisRipsBackend(min_full_torus_points=4)
    cloud = CalibrationCloud.clean_torus_grid(grid_size=4, dimension=2)

    diagram = backend.compute_diagram(cloud, max_dim=2, filtration="vietoris_rips")

    assert isinstance(diagram, PersistenceDiagram)
    assert diagram.backend_name == "VietorisRipsBackend"
    assert diagram.filtration == "vietoris_rips"
    assert diagram.max_dim == 2
    assert diagram.point_count == cloud.point_count
    assert diagram.betti_lower_bounds()[1] == 2


def test_betti_calibration_facade_exposes_optional_diagram_interface() -> None:
    facade = BettiCalibrationBackend(VietorisRipsBackend(min_full_torus_points=4))
    diagram = facade.compute_diagram([(0, 0), (32, 0), (64, 0)], max_dim=1)

    assert diagram.source_kind == CloudKind.REPLAY
    assert all(point.dimension <= 1 for point in diagram.points)


def test_runtime_still_does_not_import_betti_after_optional_backend_interface() -> None:
    source = inspect.getsource(runtime)
    assert "htce_origin.topology.betti" not in source
    assert "BettiCalibrationBackend" not in source
    assert "PersistentHomologyBackend" not in source
