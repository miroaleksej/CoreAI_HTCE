from htce_origin.evaluation.long_run_stability import LongRunProfile, run_long_run_stability, smoke_p20_profiles
from htce_origin.kernel.q16 import Q256_MODULUS


def test_p20_smoke_profiles_pass_and_preserve_invariants():
    payload = run_long_run_stability(smoke_p20_profiles())
    assert payload["passed"] is True
    assert payload["hardware_claim_status"] == "arithmetic_model_verified"
    assert payload["board_measurement_status"] == "not_board_measured"
    assert payload["profile_count"] == 2
    for report in payload["reports"]:
        assert report["passed"] is True
        assert report["trace_valid"] is True
        assert report["no_float_runtime"] is True
        assert report["no_l2_smearing"] is True
        assert report["no_unauthorized_real_action"] is True
        assert report["no_evidence_leak"] is True
        assert report["bounded_uncertainty"] is True
        assert report["clocks_consistent"] is True
        assert report["checkpoint_restore_ok"] is True
        assert report["replay_verification_ok"] is True
        assert report["profile"]["steps"] == report["l1_clock"]
        assert report["l2_clock"] == 0
        assert report["l3_clock"] == 0
        assert report["latest_fact_count"] == 0
        assert report["active_l2_clean_digest"] == report["zero_l2_clean_digest"]
        assert report["trace_count"] >= report["profile"]["steps"]


def test_p20_custom_profile_runs_q256():
    profile = LongRunProfile("unit_64", steps=64, checkpoint_interval=16, dimension=3, input_dim=3)
    payload = run_long_run_stability((profile,))
    row = payload["reports"][0]
    assert payload["passed"] is True
    assert row["profile"]["steps"] == 64
    assert row["profile"]["dimension"] == 3
    assert row["profile"]["trace_every_step"] is True
    assert row["l1_clock"] == 64
    assert int(Q256_MODULUS) == 2 ** 256
