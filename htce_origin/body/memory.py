"""Fact-as-delta L2 memory for HTCE-Origin Clean Body L1/L2/L3 body.

The memory organ separates two things that must not be conflated:

1. The active toroidal state vector, maintained by ``layers.L123Body``.
2. The semantic latest-state index, maintained here as traceable records.

Old facts are not deleted.  New facts over the same semantic key supersede old
facts.  Direct A / NOT A conflicts quarantine the affected records instead of
silently selecting one side.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

from htce_origin.kernel.core import EvidenceId, FactDelta, FactFrame, EntityId, RelationId, active_state_digest, fact_delta
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_mod, q_toroidal_loss_vector


class MemoryError(ValueError):
    """Raised when fact memory input violates the L1/L2/L3 body contract."""


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"
    NEGATED = "negated"


@dataclass(frozen=True, order=True)
class FactKey:
    subject: str
    relation: str

    @classmethod
    def from_frame(cls, fact: FactFrame) -> "FactKey":
        return cls(fact.subject.value, fact.relation.value)

    def as_payload(self) -> dict[str, str]:
        return {"relation": self.relation, "subject": self.subject}


@dataclass(frozen=True)
class MemoryRecord:
    delta: FactDelta
    status: FactStatus
    trace_id: str
    sequence: int = 0
    revision: int = 1
    supersedes: tuple[str, ...] = ()
    negated: bool = False
    reason: str = ""
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise MemoryError("trace_id must be non-empty")
        if self.sequence < 0:
            raise MemoryError("sequence must be non-negative")
        if self.revision <= 0:
            raise MemoryError("revision must be positive")
        rid = self.record_id or active_state_digest({
            "delta": self.delta.delta,
            "evidence": self.delta.fact.evidence.value,
            "negated": self.negated,
            "object": self.delta.fact.object.value,
            "relation": self.delta.fact.relation.value,
            "revision": self.revision,
            "sequence": self.sequence,
            "status": self.status.value,
            "subject": self.delta.fact.subject.value,
            "supersedes": self.supersedes,
        })
        object.__setattr__(self, "record_id", rid)
        object.__setattr__(self, "supersedes", tuple(self.supersedes))

    @property
    def key(self) -> FactKey:
        return FactKey.from_frame(self.delta.fact)

    @property
    def object_value(self) -> str:
        return self.delta.fact.object.value

    def statement_text(self) -> str:
        return f"{self.delta.fact.relation.value}({self.delta.fact.subject.value},{self.delta.fact.object.value})"

    def as_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.delta.fact.evidence.value,
            "key": self.key.as_payload(),
            "negated": self.negated,
            "object": self.object_value,
            "reason": self.reason,
            "record_id": self.record_id,
            "revision": self.revision,
            "sequence": self.sequence,
            "status": self.status.value,
            "supersedes": list(self.supersedes),
            "trace_id": self.trace_id,
        }

    def snapshot_payload(self) -> dict[str, object]:
        payload = self.as_payload()
        payload.update({
            "confidence_bp": self.delta.fact.confidence_bp,
            "delta": list(self.delta.delta),
            "fact_metadata": dict(self.delta.fact.metadata),
            "weight": self.delta.weight,
            "modulus": self.delta.modulus,
        })
        return payload


@dataclass(frozen=True)
class QueryResult:
    subject: str
    relation: str
    status: FactStatus | str
    answer: str | None = None
    record_id: str | None = None
    evidence_id: str | None = None
    reason: str = ""
    trace_id: str | None = None

    @property
    def answered(self) -> bool:
        return self.status == FactStatus.ACTIVE and self.answer is not None

    def as_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "evidence_id": self.evidence_id,
            "reason": self.reason,
            "record_id": self.record_id,
            "relation": self.relation,
            "status": self.status.value if isinstance(self.status, FactStatus) else self.status,
            "subject": self.subject,
            "trace_id": self.trace_id,
        }


class ActiveFactIndex:
    def __init__(self) -> None:
        self._active: dict[FactKey, MemoryRecord] = {}

    def get(self, key: FactKey) -> MemoryRecord | None:
        return self._active.get(key)

    def set(self, record: MemoryRecord) -> None:
        self._active[record.key] = record

    def pop(self, key: FactKey) -> MemoryRecord | None:
        return self._active.pop(key, None)

    def items(self) -> tuple[tuple[FactKey, MemoryRecord], ...]:
        return tuple(sorted(self._active.items(), key=lambda item: (item[0].subject, item[0].relation)))


class SupersededFactIndex:
    def __init__(self) -> None:
        self._items: dict[FactKey, list[MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> None:
        self._items.setdefault(record.key, []).append(record)

    def get(self, key: FactKey) -> tuple[MemoryRecord, ...]:
        return tuple(self._items.get(key, ()))


class ConflictIndex:
    def __init__(self) -> None:
        self._items: dict[FactKey, list[MemoryRecord]] = {}

    def add(self, *records: MemoryRecord) -> None:
        for record in records:
            self._items.setdefault(record.key, []).append(record)

    def has_conflict(self, key: FactKey) -> bool:
        return bool(self._items.get(key))

    def get(self, key: FactKey) -> tuple[MemoryRecord, ...]:
        return tuple(self._items.get(key, ()))


class EvidenceIndex:
    def __init__(self) -> None:
        self._by_evidence: dict[str, list[MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> None:
        self._by_evidence.setdefault(record.delta.fact.evidence.value, []).append(record)

    def get(self, evidence_id: str) -> tuple[MemoryRecord, ...]:
        return tuple(self._by_evidence.get(str(evidence_id).lower(), ()))


class EpisodeBuffer:
    def __init__(self) -> None:
        self._events: list[MemoryRecord] = []

    def append(self, record: MemoryRecord) -> None:
        self._events.append(record)

    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._events)


class FactDeltaStore:
    """Traceable latest-state memory with supersession and quarantine."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self.active = ActiveFactIndex()
        self.superseded = SupersededFactIndex()
        self.conflicts = ConflictIndex()
        self.evidence = EvidenceIndex()
        self.episodes = EpisodeBuffer()

    def commit(self, record: MemoryRecord | FactDelta, trace_id: str | None = None) -> MemoryRecord:
        if isinstance(record, MemoryRecord):
            base = record
        else:
            if trace_id is None:
                raise MemoryError("trace_id is required when committing a FactDelta")
            base = MemoryRecord(record, FactStatus.ACTIVE, trace_id=trace_id)
        key = base.key
        prior = self.active.get(key)
        supersedes: tuple[str, ...] = ()
        revision = 1
        if prior is not None:
            prior_superseded = replace(prior, status=FactStatus.SUPERSEDED, reason="superseded by newer fact")
            self._replace_record(prior, prior_superseded)
            self.superseded.add(prior_superseded)
            supersedes = (prior_superseded.record_id,)
            revision = prior.revision + 1
        committed = replace(
            base,
            status=FactStatus.ACTIVE,
            sequence=len(self.records),
            revision=revision,
            supersedes=supersedes,
            negated=False,
            reason=base.reason or "active latest fact",
            record_id="",
        )
        self._append_record(committed)
        self.active.set(committed)
        return committed

    def commit_fact(self, fact: FactFrame, *, trace_id: str, dimension: int = 64) -> MemoryRecord:
        return self.commit(fact_delta(fact, dimension=dimension), trace_id=trace_id)

    def commit_negation(self, fact: FactFrame, *, trace_id: str, dimension: int = 64) -> MemoryRecord:
        delta = fact_delta(fact, dimension=dimension)
        key = FactKey.from_frame(fact)
        active = self.active.get(key)
        negation = MemoryRecord(
            delta=delta,
            status=FactStatus.QUARANTINED,
            trace_id=trace_id,
            sequence=len(self.records),
            revision=(active.revision + 1 if active else 1),
            negated=True,
            reason="negative fact quarantined pending contradiction resolution",
        )
        self._append_record(negation)
        if active is not None and active.object_value == fact.object.value:
            quarantined_active = replace(active, status=FactStatus.QUARANTINED, reason="direct contradiction: A and NOT A")
            self._replace_record(active, quarantined_active)
            self.active.pop(key)
            self.conflicts.add(quarantined_active, negation)
        else:
            self.conflicts.add(negation)
        return negation

    def query(self, subject: str, relation: str, *, trace_id: str | None = None) -> QueryResult:
        key = FactKey(EntityId(subject).value, RelationId(relation).value)
        if self.conflicts.has_conflict(key):
            return QueryResult(key.subject, key.relation, FactStatus.QUARANTINED, reason="fact key is quarantined by contradiction", trace_id=trace_id)
        active = self.active.get(key)
        if active is None:
            return QueryResult(key.subject, key.relation, "unknown", reason="no supported active fact", trace_id=trace_id)
        return QueryResult(
            key.subject,
            key.relation,
            FactStatus.ACTIVE,
            answer=active.object_value,
            evidence_id=active.delta.fact.evidence.value,
            record_id=active.record_id,
            reason="latest active fact",
            trace_id=trace_id,
        )

    def _weighted_record_delta(self, record: MemoryRecord, *, modulus: int = DEFAULT_MODULUS) -> tuple[int, ...]:
        """Return w_f * Δ_f mod N for a committed memory record."""
        return tuple(q_mod(int(value) * int(record.delta.weight), modulus) for value in record.delta.delta)

    def associative_toroidal_read(
        self,
        subject: str,
        relation: str,
        *,
        current_l2_state: Sequence[int],
        candidate_objects: Sequence[str] | None = None,
        trace_id: str | None = None,
        modulus: int = DEFAULT_MODULUS,
    ) -> QueryResult:
        """Read a fact through the weighted toroidal L2 state before proof export.

        The latest-state index is used only to enumerate supported candidate
        records for the requested key.  Selection is then performed by minimizing
        the integer LUT-backed toroidal loss

            D_T(h_L2(t), w_f * Δ_f)

        for each active candidate record.  Unsupported objects are therefore not
        surfaced merely because they are close in phase space: the proof/policy
        bridge still receives the selected active record id and evidence id.
        """
        key = FactKey(EntityId(subject).value, RelationId(relation).value)
        if self.conflicts.has_conflict(key):
            return QueryResult(key.subject, key.relation, FactStatus.QUARANTINED, reason="fact key is quarantined by contradiction", trace_id=trace_id)

        state = tuple(q_mod(value, modulus) for value in current_l2_state)
        if not state:
            raise MemoryError("current_l2_state must be non-empty for toroidal read")

        allowed_objects = None
        if candidate_objects is not None:
            allowed_objects = {EntityId(item).value for item in candidate_objects}

        candidates = tuple(
            record
            for record in self.active_records()
            if record.key == key
            and record.status == FactStatus.ACTIVE
            and (allowed_objects is None or record.object_value in allowed_objects)
        )
        if not candidates:
            return QueryResult(key.subject, key.relation, "unknown", reason="no supported active toroidal candidate", trace_id=trace_id)

        best_record: MemoryRecord | None = None
        best_distance: int | None = None
        for record in candidates:
            weighted_delta = self._weighted_record_delta(record, modulus=modulus)
            if len(weighted_delta) != len(state):
                raise MemoryError("candidate delta dimension differs from L2 state")
            distance = q_toroidal_loss_vector(state, weighted_delta, modulus)
            if best_distance is None or distance < best_distance or (distance == best_distance and record.record_id < (best_record.record_id if best_record else "")):
                best_distance = distance
                best_record = record

        if best_record is None or best_distance is None:
            return QueryResult(key.subject, key.relation, "unknown", reason="toroidal candidate scoring produced no winner", trace_id=trace_id)

        return QueryResult(
            key.subject,
            key.relation,
            FactStatus.ACTIVE,
            answer=best_record.object_value,
            evidence_id=best_record.delta.fact.evidence.value,
            record_id=best_record.record_id,
            reason=f"associative_toroidal_read_min_loss_{best_distance}_candidates_{len(candidates)}",
            trace_id=trace_id,
        )

    def latest(self, subject: str, relation: str) -> MemoryRecord | None:
        """Return the active same-key fact for policy/proof integration.

        This method does not resolve conflicts and does not create facts. It is
        used by runtime gates to pass the current active fact into policy before
        a same-key commit is accepted.
        """
        key = FactKey(EntityId(subject).value, RelationId(relation).value)
        return self.active.get(key)

    def export_latest_state(self) -> dict[tuple[str, str], str]:
        """Export the active latest-state index for query proof strategies.

        The output is intentionally minimal: (subject, relation) -> object. It is
        proof input only and does not bypass evidence or policy gates.
        """
        return {
            (record.key.subject.lower(), record.key.relation.lower()): record.object_value.lower()
            for record in self.active_records()
        }

    def history(self, subject: str, relation: str) -> tuple[MemoryRecord, ...]:
        key = FactKey(EntityId(subject).value, RelationId(relation).value)
        return tuple(record for record in self.records if record.key == key)

    def active_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(record for _, record in self.active.items())

    def digest(self) -> str:
        return active_state_digest({
            "active": [record.as_payload() for record in self.active_records()],
            "conflicts": [record.as_payload() for records in self.conflicts._items.values() for record in records],
            "record_count": len(self.records),
        })

    def snapshot(self) -> dict[str, object]:
        return {
            "records": [record.snapshot_payload() for record in self.records],
            "digest": self.digest(),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "FactDeltaStore":
        store = cls()
        raw_records = payload.get("records", ())
        if not isinstance(raw_records, list):
            raise MemoryError("memory snapshot records must be a list")
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise MemoryError("memory snapshot record must be a mapping")
            key = raw.get("key", {})
            if not isinstance(key, dict):
                raise MemoryError("memory snapshot record key must be a mapping")
            fact = FactFrame(
                subject=EntityId(str(key.get("subject", ""))),
                relation=RelationId(str(key.get("relation", ""))),
                object=EntityId(str(raw.get("object", ""))),
                evidence=EvidenceId(str(raw.get("evidence_id", ""))),
                confidence_bp=int(raw.get("confidence_bp", 10000)),
                metadata=dict(raw.get("fact_metadata", {})) if isinstance(raw.get("fact_metadata", {}), dict) else {},
            )
            delta = FactDelta(
                fact=fact,
                delta=tuple(int(v) for v in raw.get("delta", ())),
                weight=int(raw.get("weight", 1)),
                modulus=int(raw.get("modulus", 65536)),
            )
            status = FactStatus(str(raw.get("status", FactStatus.ACTIVE.value)))
            record = MemoryRecord(
                delta=delta,
                status=status,
                trace_id=str(raw.get("trace_id", "restored_snapshot")),
                sequence=int(raw.get("sequence", len(store.records))),
                revision=int(raw.get("revision", 1)),
                supersedes=tuple(str(v) for v in raw.get("supersedes", ())),
                negated=bool(raw.get("negated", False)),
                reason=str(raw.get("reason", "restored_snapshot")),
                record_id=str(raw.get("record_id", "")),
            )
            store.records.append(record)
            store.evidence.add(record)
            store.episodes.append(record)
            if record.status == FactStatus.ACTIVE:
                store.active.set(record)
            elif record.status == FactStatus.SUPERSEDED:
                store.superseded.add(record)
            elif record.status == FactStatus.QUARANTINED:
                store.conflicts.add(record)
        expected_digest = payload.get("digest")
        if isinstance(expected_digest, str) and expected_digest and expected_digest != store.digest():
            raise MemoryError("memory snapshot digest mismatch")
        return store

    def _append_record(self, record: MemoryRecord) -> None:
        self.records.append(record)
        self.evidence.add(record)
        self.episodes.append(record)

    def _replace_record(self, old: MemoryRecord, new: MemoryRecord) -> None:
        for idx, record in enumerate(self.records):
            if record.record_id == old.record_id:
                self.records[idx] = new
                return
        raise MemoryError("record to replace was not found")
