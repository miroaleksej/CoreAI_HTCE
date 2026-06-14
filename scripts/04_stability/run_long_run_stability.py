#!/usr/bin/env python3
"""Run P20 long-run organism stability acceptance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from htce_origin.evaluation.long_run_stability import (
    LongRunProfile,
    default_p20_profiles,
    run_long_run_stability,
    smoke_p20_profiles,
    write_long_run_artifacts,
)


def _profiles_from_args(args: argparse.Namespace) -> tuple[LongRunProfile, ...]:
    if args.smoke:
        return smoke_p20_profiles()
    if args.steps:
        return tuple(
            LongRunProfile(
                name=f"p20_custom_{step}",
                steps=int(step),
                checkpoint_interval=max(1, int(step) // max(1, args.checkpoints)),
                trace_every_step=not args.checkpoint_trace_only,
            )
            for step in args.steps
        )
    return default_p20_profiles()


def main() -> int:
    parser = argparse.ArgumentParser(description="P20 long-run organism stability acceptance")
    parser.add_argument("--smoke", action="store_true", help="run short CI profiles")
    parser.add_argument("--steps", nargs="*", type=int, help="custom step counts")
    parser.add_argument("--checkpoints", type=int, default=10, help="target checkpoint count for custom profiles")
    parser.add_argument("--checkpoint-trace-only", action="store_true", help="trace only checkpoints for custom profiles")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    profiles = _profiles_from_args(args)
    payload = run_long_run_stability(profiles)
    write_long_run_artifacts(payload, artifacts_dir=args.artifacts_dir)
    print("P20 Long-Run Organism Stability")
    print(f"profiles: {payload['profile_count']}")
    print(f"passed: {payload['passed']}")
    print(f"hardware_claim_status: {payload['hardware_claim_status']}")
    print(f"board_measurement_status: {payload['board_measurement_status']}")
    for row in payload["reports"]:  # type: ignore[index]
        print(
            f"{row['profile']['name']}: steps={row['profile']['steps']} "
            f"passed={row['passed']} trace_valid={row['trace_valid']} "
            f"restore={row['checkpoint_restore_ok']} replay={row['replay_verification_ok']} "
            f"trace_count={row['trace_count']}"
        )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
