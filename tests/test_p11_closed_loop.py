import ast
from pathlib import Path

from htce_origin import HTCERuntime
from htce_origin.kernel.q16 import Q256_MODULUS

ROOT = Path(__file__).resolve().parents[1]


def _float_constant_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float))


def test_p11_closed_loop_runtime_float_count_remains_zero():
    assert _float_constant_count(ROOT / "htce_origin" / "body" / "runtime.py") == 0


def test_p11_closed_loop_runs_l1_world_planner_trace_q256():
    runtime = HTCERuntime()
    runtime.wake()

    report = runtime.run_closed_loop_simulation(steps=5)

    assert len(report.steps) == 5
    assert report.modulus == Q256_MODULUS
    assert report.trace_verified is True
    assert all(step.gate_allowed for step in report.steps)
    assert runtime.body.l1.clock == 5
    assert runtime.body.l2.clock == 0
    assert runtime.body.l3.clock == 0
    assert runtime.health()["latest_fact_count"] == 0
    assert any(step.chosen_action in {"advance", "rotate", "hold"} for step in report.steps)
    assert all(0 <= step.efe_bp <= 10000 for step in report.steps)
    assert all(0 <= step.surprise_bp <= 10000 for step in report.steps)


def test_p11_closed_loop_is_deterministic_for_same_initial_runtime():
    first = HTCERuntime()
    second = HTCERuntime()
    first.wake()
    second.wake()

    report_a = first.run_closed_loop_simulation(steps=4)
    report_b = second.run_closed_loop_simulation(steps=4)

    assert [step.chosen_action for step in report_a.steps] == [step.chosen_action for step in report_b.steps]
    assert [step.efe_bp for step in report_a.steps] == [step.efe_bp for step in report_b.steps]
    assert [step.surprise_bp for step in report_a.steps] == [step.surprise_bp for step in report_b.steps]
    assert report_a.trace_verified and report_b.trace_verified
