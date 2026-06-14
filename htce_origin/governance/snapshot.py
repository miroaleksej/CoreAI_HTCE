"""Q16 kernel4 snapshot/restore and release-state boundary.

Snapshots bind a canonical state payload to a verified protected trace snapshot.
They are deterministic release artifacts, not a live L2/L3 storage engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from htce_origin.governance.evidence import HashChain, TraceEvent, TraceSnapshot, TraceVerifier
from htce_origin.kernel.serialization import canonical_json_str, release_manifest_self_hash, release_manifest_without_self_hash, sha256_hex, verify_release_manifest_self_hash as _verify_release_manifest_self_hash


class SnapshotError(ValueError):
    """Raised when a snapshot, restore or release manifest fails validation."""


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    version: str
    state_hash: str
    trace_hash: str
    trace_head: str
    trace_count: int
    schema_version: str = "snapshot-v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "state_hash": self.state_hash,
            "trace_count": self.trace_count,
            "trace_hash": self.trace_hash,
            "trace_head": self.trace_head,
            "version": self.version,
        }


@dataclass(frozen=True)
class SnapshotBundle:
    manifest: SnapshotManifest
    state_payload: Mapping[str, Any]
    trace_events: tuple[TraceEvent, ...]
    bundle_hash: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "bundle_hash": self.bundle_hash,
            "manifest": self.manifest.as_payload(),
            "state_payload": dict(self.state_payload),
            "trace_hashes": [event.event_hash() for event in self.trace_events],
        }


@dataclass(frozen=True)
class ReleaseManifest:
    release_id: str
    version: str
    bundle_hash: str
    manifest_hash: str
    report_hashes: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    schema_version: str = "release-v1"

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact_hashes": dict(self.artifact_hashes),
            "bundle_hash": self.bundle_hash,
            "manifest_hash": self.manifest_hash,
            "release_id": self.release_id,
            "report_hashes": dict(self.report_hashes),
            "schema_version": self.schema_version,
            "version": self.version,
        }


class SnapshotStore:
    """Create and validate deterministic snapshot bundles."""

    def create_manifest(
        self,
        state_hash: str,
        trace_head: str,
        version: str,
        *,
        trace_hash: str | None = None,
        trace_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> SnapshotManifest:
        """Backward-compatible manifest constructor from final clean release/14."""

        trace_digest = trace_hash or sha256_hex({"trace_head": trace_head, "trace_count": trace_count})
        payload = {
            "metadata": dict(metadata or {}),
            "schema_version": "snapshot-v1",
            "state_hash": state_hash,
            "trace_count": trace_count,
            "trace_hash": trace_digest,
            "trace_head": trace_head,
            "version": version,
        }
        snapshot_id = sha256_hex(payload)
        return SnapshotManifest(
            snapshot_id=snapshot_id,
            version=version,
            state_hash=state_hash,
            trace_hash=trace_digest,
            trace_head=trace_head,
            trace_count=trace_count,
            metadata=dict(metadata or {}),
        )

    def create_bundle(
        self,
        *,
        state_payload: Mapping[str, Any],
        trace: HashChain,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SnapshotBundle:
        trace_snapshot = trace.snapshot()
        state_hash = sha256_hex(state_payload)
        trace_hash = trace_snapshot.snapshot_hash
        manifest = self.create_manifest(
            state_hash=state_hash,
            trace_head=trace_snapshot.head,
            version=version,
            trace_hash=trace_hash,
            trace_count=trace_snapshot.count,
            metadata=metadata,
        )
        bundle_payload = {
            "manifest": manifest.as_payload(),
            "state_hash": state_hash,
            "trace_hash": trace_hash,
        }
        bundle_hash = sha256_hex(bundle_payload)
        return SnapshotBundle(
            manifest=manifest,
            state_payload=dict(state_payload),
            trace_events=tuple(trace_snapshot.events),
            bundle_hash=bundle_hash,
        )

    def verify_bundle(self, bundle: SnapshotBundle) -> bool:
        try:
            state_hash = sha256_hex(bundle.state_payload)
            if state_hash != bundle.manifest.state_hash:
                return False
            trace_snapshot = TraceSnapshot(
                events=tuple(bundle.trace_events),
                head=bundle.manifest.trace_head,
                count=bundle.manifest.trace_count,
                snapshot_hash=bundle.manifest.trace_hash,
            )
            if not TraceVerifier.verify_snapshot(trace_snapshot):
                return False
            expected_manifest = self.create_manifest(
                state_hash=bundle.manifest.state_hash,
                trace_head=bundle.manifest.trace_head,
                version=bundle.manifest.version,
                trace_hash=bundle.manifest.trace_hash,
                trace_count=bundle.manifest.trace_count,
                metadata=bundle.manifest.metadata,
            )
            if expected_manifest.snapshot_id != bundle.manifest.snapshot_id:
                return False
            expected_bundle_hash = sha256_hex({
                "manifest": bundle.manifest.as_payload(),
                "state_hash": bundle.manifest.state_hash,
                "trace_hash": bundle.manifest.trace_hash,
            })
            return expected_bundle_hash == bundle.bundle_hash
        except Exception:
            return False

    def restore(self, bundle: SnapshotBundle, *, expected_version: str | None = None) -> Mapping[str, Any]:
        if expected_version is not None and bundle.manifest.version != expected_version:
            raise SnapshotError("snapshot version mismatch")
        if not self.verify_bundle(bundle):
            raise SnapshotError("snapshot verification failed")
        return dict(bundle.state_payload)

    def create_release_manifest(
        self,
        *,
        release_id: str,
        version: str,
        bundle: SnapshotBundle,
        reports: Mapping[str, str],
        artifacts: Mapping[str, str],
    ) -> ReleaseManifest:
        report_hashes = {name: sha256_hex(content) for name, content in sorted(reports.items())}
        artifact_hashes = {name: digest for name, digest in sorted(artifacts.items())}
        manifest_hash = sha256_hex(bundle.manifest.as_payload())
        return ReleaseManifest(
            release_id=release_id,
            version=version,
            bundle_hash=bundle.bundle_hash,
            manifest_hash=manifest_hash,
            report_hashes=report_hashes,
            artifact_hashes=artifact_hashes,
        )


def corrupt_trace_event(bundle: SnapshotBundle, *, index: int = 0, key: str = "tampered") -> SnapshotBundle:
    """Testing helper: return a bundle with a modified trace event but same manifest."""

    events = list(bundle.trace_events)
    event = events[index]
    payload = dict(event.payload)
    payload[key] = True
    events[index] = replace(event, payload=payload)
    return SnapshotBundle(
        manifest=bundle.manifest,
        state_payload=bundle.state_payload,
        trace_events=tuple(events),
        bundle_hash=bundle.bundle_hash,
    )



def verify_release_manifest_self_hash(manifest_payload: Mapping[str, Any]) -> bool:
    """Verify the protected self-hash protocol for a release manifest payload."""

    return _verify_release_manifest_self_hash(manifest_payload)



def release_manifest_protocol_payload(manifest_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical preimage used for the release manifest self-hash."""

    return release_manifest_without_self_hash(manifest_payload)
