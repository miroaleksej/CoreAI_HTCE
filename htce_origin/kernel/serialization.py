"""Q16 kernel4 canonical serialization helpers.

Protected release artifacts use deterministic JSON and SHA-256 hashes. Runtime
float values are rejected to preserve the integer/Q16 protected-path contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass, fields
from enum import Enum
from typing import Any, Mapping, Sequence


class SerializationError(ValueError):
    """Raised when payloads cannot be canonically serialized."""


def _to_json_safe(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise SerializationError(f"float value is forbidden in protected serialization at {path}")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_json_safe(getattr(value, field.name), path=f"{path}.{field.name}") for field in fields(value)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(f"canonical JSON keys must be strings at {path}")
            out[key] = _to_json_safe(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    raise SerializationError(f"unsupported value type at {path}: {type(value).__name__}")


def canonical_json_str(payload: Any) -> str:
    return json.dumps(_to_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(payload: Any) -> bytes:
    return canonical_json_str(payload).encode("utf-8")


def sha256_hex(payload: bytes | str | Mapping[str, Any] | Sequence[Any] | Any) -> str:
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = canonical_json_bytes(payload)
    return hashlib.sha256(data).hexdigest()


# Backward-compatible alias used by earlier final clean release files.
canonical_bytes = canonical_json_bytes



def release_manifest_without_self_hash(manifest_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical manifest payload with its self-referential hash removed.

    ``file_hashes_sha256["RELEASE_MANIFEST.json"]`` cannot be a raw hash of the
    complete manifest file, because writing that digest changes the file.  The
    protected-path convention is therefore:

    manifest_self_hash = SHA256(canonical_json(RELEASE_MANIFEST minus
        file_hashes_sha256["RELEASE_MANIFEST.json"]))

    The returned object is JSON-safe and deterministic.  It does not mutate the
    caller's mapping.
    """

    safe = _to_json_safe(manifest_payload)
    if not isinstance(safe, dict):
        raise SerializationError("release manifest payload must be a JSON object")
    file_hashes = safe.get("file_hashes_sha256")
    if isinstance(file_hashes, dict):
        file_hashes.pop("RELEASE_MANIFEST.json", None)
    return safe


def release_manifest_self_hash(manifest_payload: Mapping[str, Any]) -> str:
    """Compute the self-hash defined by the release-manifest protocol."""

    return sha256_hex(release_manifest_without_self_hash(manifest_payload))


def verify_release_manifest_self_hash(manifest_payload: Mapping[str, Any]) -> bool:
    """Verify ``file_hashes_sha256["RELEASE_MANIFEST.json"]`` self-hash.

    The comparison is against the canonical payload with the self-hash field
    removed, not against the raw bytes of the complete manifest file.
    """

    safe = _to_json_safe(manifest_payload)
    if not isinstance(safe, dict):
        return False
    file_hashes = safe.get("file_hashes_sha256")
    if not isinstance(file_hashes, dict):
        return False
    expected = file_hashes.get("RELEASE_MANIFEST.json")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    try:
        return expected == release_manifest_self_hash(safe)
    except SerializationError:
        return False
