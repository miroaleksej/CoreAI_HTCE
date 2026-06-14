from htce_origin.body.runtime import HTCERuntime
from htce_origin.kernel.config import RuntimeConfig
from htce_origin.kernel.q16 import Q256_MODULUS
from htce_origin.kernel.serialization import canonical_json_bytes


def test_1000_step_closed_loop_stability_q256():
    runtime = HTCERuntime(RuntimeConfig(l1_dim=3, l1_input_dim=3, l2_dim=3, l3_dim=3))
    runtime.wake()
    report = runtime.run_closed_loop_simulation(steps=1000, verify_trace_each_step=False)

    assert report.modulus == Q256_MODULUS
    assert len(report.steps) == 1000
    assert report.trace_verified is True
    assert runtime.trace.verify() is True
    assert runtime.health()["trace_verified"] is True
    assert runtime.health()["l1_clock"] >= 1000
    assert runtime.health()["latest_fact_count"] == 0
    assert runtime.health()["l2_clock"] == 0
    assert runtime.health()["l3_clock"] == 0
    assert 0 <= report.average_efe_bp <= 10000
    assert 0 <= report.average_surprise_bp <= 10000
    assert report.advance_count + report.rotate_count + report.hold_count == 1000
    canonical_json_bytes(runtime.export_state())
