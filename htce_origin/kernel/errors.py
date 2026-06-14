"""Typed fail-closed errors for HTCE-Origin final clean release."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    code: str
    message: str
    recoverable: bool = False


class HTCEError(Exception):
    """Base error. final clean release errors must be explicit, not silent."""


class ContractError(HTCEError):
    """Raised when an interface contract is violated."""


class GateRejected(HTCEError):
    """Raised when a gate blocks a request."""


class UnsafeActionBlocked(HTCEError):
    """Raised when a real-world action is attempted in simulation-first mode."""
