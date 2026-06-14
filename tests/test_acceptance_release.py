from pathlib import Path
from htce_origin import HTCERuntime, RuntimeRequest
from htce_origin.kernel.config import RuntimeConfig
from htce_origin.interface.api import is_allowed_endpoint, ALLOWED_ENDPOINTS

ROOT = Path(__file__).resolve().parents[1]


def test_claim_boundary_contains_prohibited_claims_section():
    text = (ROOT / "CLAIM_BOUNDARY.md").read_text(encoding="utf-8")
    assert "Prohibited claims" in text
    assert "architecture freeze" in text
    assert "If a feature has no test" in text


def test_runtime_blocks_real_actions_and_legacy_by_default():
    cfg = RuntimeConfig()
    assert cfg.simulation_first is True
    assert cfg.allow_real_actions is False
    assert cfg.allow_legacy_imports is False
    cfg.validate()


def test_operator_surface_is_short_and_explicit():
    assert len(ALLOWED_ENDPOINTS) <= 10
    assert is_allowed_endpoint("/wake")
    assert not is_allowed_endpoint("/legacy/release_ladder")


def test_plain_supported_nlu_runtime_produces_trace_and_checked_commit():
    rt = HTCERuntime()
    rt.wake()
    response = rt.tick(RuntimeRequest("Mary is in office"))
    assert response.decision.trace_id
    assert response.decision.kind.value == "answer"
    assert response.output == "COMMIT: mary located_in office"
    assert rt.health()["latest_fact_count"] == 1
    assert rt.health()["legacy_imports_allowed"] is False
