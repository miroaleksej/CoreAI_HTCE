"""Immutable runtime configuration for final clean release."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    modulus: int = 2 ** 256
    q_format: str = "Q256"
    l1_dim: int = 64
    l1_input_dim: int = 256
    l2_dim: int = 64
    l3_dim: int = 64
    simulation_first: bool = True
    allow_legacy_imports: bool = False
    allow_real_actions: bool = False
    trace_required: bool = True

    def validate(self) -> None:
        if self.modulus <= 0 or self.modulus & (self.modulus - 1) != 0:
            raise ValueError("modulus must be a positive power of two")
        if self.l1_dim <= 0 or self.l1_input_dim <= 0 or self.l2_dim <= 0 or self.l3_dim <= 0:
            raise ValueError("runtime dimensions must be positive")
        if self.allow_legacy_imports:
            raise ValueError("legacy imports are forbidden in clean runtime")
        if self.allow_real_actions:
            raise ValueError("real actions are blocked in final clean release")
