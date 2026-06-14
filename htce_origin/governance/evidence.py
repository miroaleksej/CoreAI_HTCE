"""Protected evidence, canonical JSON and hash-chain trace layer.

governance boundary implements the evidence/trace boundary only.  It does not commit
facts to L2/L3 and it does not authorize external actions.  All trace payloads
are canonicalized with deterministic JSON and protected by a hash chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

GENESIS_HASH = "GENESIS"


class EvidenceError(ValueError):
    """Raised for invalid evidence, canonical JSON or trace-chain state."""


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    REVOKED = "revoked"
    QUARANTINED = "quarantined"


class EvidenceRelation(str, Enum):
    """Evidence-only relation between an anchor and a claim."""

    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SourceManifest:
    """Bounded provenance/source-quality record for evidence-only anchoring.

    The manifest is metadata about a source, not a fact claim.  It can downweight
    weak, contradicted or retracted material and can reward primary/replicated
    material, but it cannot itself settle truth.
    """

    source_id: str
    uri: str
    title: str
    source_type: str
    base_quality_bp: int
    primary_source: int = 0
    independent_replication_count: int = 0
    weak_source: int = 0
    retracted: int = 0
    correction_or_expression_of_concern: int = 0
    contradiction_markers: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise EvidenceError("source_id must be non-empty")
        if not self.uri or not self.uri.strip():
            raise EvidenceError("source uri must be non-empty")
        if not self.title or not self.title.strip():
            raise EvidenceError("source title must be non-empty")
        if not self.source_type or not self.source_type.strip():
            raise EvidenceError("source_type must be non-empty")
        for name in ("base_quality_bp", "independent_replication_count", "contradiction_markers"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise EvidenceError(f"{name} must be a non-negative integer")
        for name in ("primary_source", "weak_source", "retracted", "correction_or_expression_of_concern"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise EvidenceError(f"{name} must be 0 or 1")
        if self.base_quality_bp > 10000:
            raise EvidenceError("base_quality_bp must be in [0, 10000]")
        _json_safe(dict(self.metadata))

    @property
    def source_weight_bp(self) -> int:
        """Integer source weight.

        w(source) = base_quality + replication_bonus + primary_source_bonus
                    - weak_source_penalty - retraction_penalty - contradiction_penalty
        """

        replication_bonus = min(1500, self.independent_replication_count * 250)
        primary_source_bonus = 1000 if self.primary_source else 0
        weak_source_penalty = 2000 if self.weak_source or self.source_type.lower() in {"blog", "forum", "social", "web_noise"} else 0
        retraction_penalty = 10000 if self.retracted else (2500 if self.correction_or_expression_of_concern else 0)
        contradiction_penalty = min(4000, self.contradiction_markers * 1000)
        return max(0, min(10000, self.base_quality_bp + replication_bonus + primary_source_bonus - weak_source_penalty - retraction_penalty - contradiction_penalty))

    def as_payload(self) -> dict[str, Any]:
        return {
            "base_quality_bp": self.base_quality_bp,
            "contradiction_markers": self.contradiction_markers,
            "correction_or_expression_of_concern": self.correction_or_expression_of_concern,
            "independent_replication_count": self.independent_replication_count,
            "metadata": dict(self.metadata),
            "primary_source": self.primary_source,
            "retracted": self.retracted,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_weight_bp": self.source_weight_bp,
            "title": self.title,
            "uri": self.uri,
            "weak_source": self.weak_source,
        }


@dataclass(frozen=True)
class EvidenceAnchor:
    """Evidence-only anchor connecting a source manifest to a claim.

    A web/pdf anchor is never a settled fact.  It is a weighted source reference
    that must still pass claim/evidence/policy gates before any answer can be
    surfaced.  The constructor rejects direct fact-commit usage.
    """

    anchor_id: str
    claim_id: str
    source: SourceManifest
    relation: EvidenceRelation
    quote_digest: str
    locator: str = ""
    evidence_only_boundary: int = 1
    settled_fact_commit: int = 0

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.anchor_id.strip():
            raise EvidenceError("anchor_id must be non-empty")
        if not self.claim_id or not self.claim_id.strip():
            raise EvidenceError("claim_id must be non-empty")
        if not self.quote_digest or not self.quote_digest.strip():
            raise EvidenceError("quote_digest must be non-empty")
        if self.evidence_only_boundary != 1:
            raise EvidenceError("evidence anchor must preserve evidence_only_boundary = 1")
        if self.settled_fact_commit != 0:
            raise EvidenceError("web/pdf evidence anchor cannot be a settled fact commit")

    @property
    def weight_bp(self) -> int:
        return self.source.source_weight_bp

    def as_payload(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "claim_id": self.claim_id,
            "evidence_only_boundary": self.evidence_only_boundary,
            "locator": self.locator,
            "quote_digest": self.quote_digest,
            "relation": self.relation.value,
            "settled_fact_commit": self.settled_fact_commit,
            "source": self.source.as_payload(),
            "weight_bp": self.weight_bp,
        }


@dataclass(frozen=True)
class ClaimSupportReport:
    claim_id: str
    support_bp: int
    contradiction_bp: int
    net_support_bp: int
    source_retracted: int
    claim_allowed: int
    support_threshold_bp: int
    contradiction_threshold_bp: int
    anchors: tuple[EvidenceAnchor, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "anchors": [anchor.as_payload() for anchor in self.anchors],
            "claim_allowed": self.claim_allowed,
            "claim_id": self.claim_id,
            "contradiction_bp": self.contradiction_bp,
            "contradiction_threshold_bp": self.contradiction_threshold_bp,
            "net_support_bp": self.net_support_bp,
            "source_retracted": self.source_retracted,
            "support_bp": self.support_bp,
            "support_threshold_bp": self.support_threshold_bp,
            "web_anchor_equals_settled_fact": 0,
        }



@dataclass(frozen=True)
class SourceThresholdCalibrationReport:
    """Calibration report for evidence source thresholds.

    This is a provenance/evidence calibration object, not a fact object.  It
    estimates conservative integer thresholds from replay/source fixtures so that
    primary+replicated support passes, weak single-source support fails, and
    retracted/contradictory sources block claims.
    """

    support_threshold_bp: int
    contradiction_threshold_bp: int
    strong_primary_support_bp: int
    replicated_support_bp: int
    weak_blog_support_bp: int
    retracted_support_bp: int
    contradicted_support_bp: int
    primary_replicated_support_passes: int
    single_weak_source_fails: int
    retracted_source_blocks: int
    contradiction_source_blocks: int
    web_anchor_settled_fact_count: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.primary_replicated_support_passes == 1
            and self.single_weak_source_fails == 1
            and self.retracted_source_blocks == 1
            and self.contradiction_source_blocks == 1
            and self.web_anchor_settled_fact_count == 0
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "contradicted_support_bp": self.contradicted_support_bp,
            "contradiction_source_blocks": self.contradiction_source_blocks,
            "contradiction_threshold_bp": self.contradiction_threshold_bp,
            "passed": self.passed,
            "primary_replicated_support_passes": self.primary_replicated_support_passes,
            "replicated_support_bp": self.replicated_support_bp,
            "retracted_source_blocks": self.retracted_source_blocks,
            "retracted_support_bp": self.retracted_support_bp,
            "single_weak_source_fails": self.single_weak_source_fails,
            "strong_primary_support_bp": self.strong_primary_support_bp,
            "support_threshold_bp": self.support_threshold_bp,
            "weak_blog_support_bp": self.weak_blog_support_bp,
            "web_anchor_settled_fact_count": self.web_anchor_settled_fact_count,
        }

class EvidenceWeigher:
    """Integer source weighting and claim-support gate."""

    def __init__(self, *, support_threshold_bp: int = 6000, contradiction_threshold_bp: int = 4000) -> None:
        if not isinstance(support_threshold_bp, int) or not 0 <= support_threshold_bp <= 10000:
            raise EvidenceError("support_threshold_bp must be an integer in [0, 10000]")
        if not isinstance(contradiction_threshold_bp, int) or not 0 <= contradiction_threshold_bp <= 10000:
            raise EvidenceError("contradiction_threshold_bp must be an integer in [0, 10000]")
        self.support_threshold_bp = support_threshold_bp
        self.contradiction_threshold_bp = contradiction_threshold_bp

    def weight_source(self, source: SourceManifest) -> int:
        return source.source_weight_bp

    @classmethod
    def from_calibration(cls, report: SourceThresholdCalibrationReport) -> "EvidenceWeigher":
        """Create a claim gate from a source-threshold calibration report."""

        return cls(
            support_threshold_bp=report.support_threshold_bp,
            contradiction_threshold_bp=report.contradiction_threshold_bp,
        )

    def score_claim(self, claim_id: str, anchors: Sequence[EvidenceAnchor]) -> ClaimSupportReport:
        selected = tuple(anchor for anchor in anchors if anchor.claim_id == claim_id)
        support_sum = sum(anchor.weight_bp for anchor in selected if anchor.relation == EvidenceRelation.SUPPORT)
        contradiction_sum = sum(anchor.weight_bp for anchor in selected if anchor.relation == EvidenceRelation.CONTRADICT)
        # Clamp aggregated support to basis points while still subtracting contradiction.
        support_bp = max(0, min(10000, support_sum))
        contradiction_bp = max(0, min(10000, contradiction_sum))
        net_support_bp = max(0, min(10000, support_bp - contradiction_bp))
        source_retracted = int(any(anchor.source.retracted for anchor in selected))
        claim_allowed = int(
            net_support_bp >= self.support_threshold_bp
            and contradiction_bp < self.contradiction_threshold_bp
            and source_retracted == 0
        )
        return ClaimSupportReport(
            claim_id=claim_id,
            support_bp=support_bp,
            contradiction_bp=contradiction_bp,
            net_support_bp=net_support_bp,
            source_retracted=source_retracted,
            claim_allowed=claim_allowed,
            support_threshold_bp=self.support_threshold_bp,
            contradiction_threshold_bp=self.contradiction_threshold_bp,
            anchors=selected,
        )



def _as_source_manifest(item: SourceManifest | EvidenceAnchor) -> SourceManifest:
    if isinstance(item, EvidenceAnchor):
        return item.source
    if isinstance(item, SourceManifest):
        return item
    raise EvidenceError("calibration inputs must be SourceManifest or EvidenceAnchor instances")


def _sum_source_weights(items: Sequence[SourceManifest | EvidenceAnchor]) -> int:
    return max(0, min(10000, sum(_as_source_manifest(item).source_weight_bp for item in items)))


def calibrate_source_thresholds(
    *,
    strong_primary_support: Sequence[SourceManifest | EvidenceAnchor],
    replicated_support: Sequence[SourceManifest | EvidenceAnchor] = (),
    weak_blog_support: Sequence[SourceManifest | EvidenceAnchor] = (),
    retracted_support: Sequence[SourceManifest | EvidenceAnchor] = (),
    contradicted_support: Sequence[SourceManifest | EvidenceAnchor] = (),
) -> SourceThresholdCalibrationReport:
    """Calibrate evidence thresholds from bounded source-quality replay fixtures.

    support_threshold is placed above single weak support and at or below the
    combined primary+replicated support. contradiction_threshold is set so that
    contradicted support blocks claims.  Retracted sources remain a hard block in
    ClaimAllowed regardless of score.
    """

    if not strong_primary_support:
        raise EvidenceError("strong_primary_support calibration set must be non-empty")
    strong_primary_bp = _sum_source_weights(strong_primary_support)
    replicated_bp = _sum_source_weights(replicated_support)
    combined_strong_bp = max(strong_primary_bp, max(0, min(10000, strong_primary_bp + replicated_bp)))
    weak_blog_bp = _sum_source_weights(weak_blog_support)
    retracted_bp = _sum_source_weights(retracted_support)
    contradicted_bp = _sum_source_weights(contradicted_support)

    # Conservative integer threshold: weak single-source support fails, primary
    # plus replication passes.  If a pathological calibration makes that
    # impossible, fail closed by raising EvidenceError instead of silently
    # weakening the gate.
    support_threshold = max(1, weak_blog_bp + 1, 6000)
    if support_threshold > combined_strong_bp:
        support_threshold = combined_strong_bp
    if support_threshold <= weak_blog_bp:
        raise EvidenceError("cannot calibrate support threshold above weak support while preserving strong support")

    contradiction_threshold = 4000
    if contradicted_bp > 0 and contradiction_threshold > contradicted_bp:
        contradiction_threshold = max(1, contradicted_bp)

    primary_pass = int(combined_strong_bp >= support_threshold)
    weak_fails = int(weak_blog_bp < support_threshold)
    retracted_blocks = int(any(_as_source_manifest(item).retracted for item in retracted_support))
    contradiction_blocks = int(contradicted_bp >= contradiction_threshold if contradicted_bp > 0 else 1)

    return SourceThresholdCalibrationReport(
        support_threshold_bp=support_threshold,
        contradiction_threshold_bp=contradiction_threshold,
        strong_primary_support_bp=strong_primary_bp,
        replicated_support_bp=replicated_bp,
        weak_blog_support_bp=weak_blog_bp,
        retracted_support_bp=retracted_bp,
        contradicted_support_bp=contradicted_bp,
        primary_replicated_support_passes=primary_pass,
        single_weak_source_fails=weak_fails,
        retracted_source_blocks=retracted_blocks,
        contradiction_source_blocks=contradiction_blocks,
    )


def _json_safe(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-safe value and reject runtime-float payloads.

    Protected runtime traces are integer/string/bool/null/list/dict only.  Floats
    are intentionally rejected because the Q16 kernel Q16 runtime contract forbids
    float values in the protected execution path.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise EvidenceError(f"float value is forbidden in protected trace payload at {path}")
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"canonical JSON keys must be strings at {path}")
            safe[key] = _json_safe(item, path=f"{path}.{key}")
        return safe
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    raise EvidenceError(f"unsupported value type in protected trace payload at {path}: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic canonical JSON used for trace hashing."""

    safe = _json_safe(value)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(value: bytes | str | Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = canonical_bytes(value)
    return hashlib.sha256(data).hexdigest()




FORBIDDEN_PUBLIC_CARD_KEYS = frozenset({
    "answer_key",
    "expected_answer",
    "template_answer",
    "hidden_criteria",
    "hidden_rubric",
    "rubric",
    "private_rubric",
})


def _contains_forbidden_public_keys(value: Any, *, path: str = "$") -> tuple[str, ...]:
    """Return forbidden key paths found in a public engine-input card."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_PUBLIC_CARD_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_contains_forbidden_public_keys(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            found.extend(_contains_forbidden_public_keys(item, path=f"{path}[{idx}]"))
    return tuple(found)


@dataclass(frozen=True)
class PreExecutionCommitment:
    """Hash commitment proving hidden criteria were fixed before execution.

    The payload contains only hashes and invariant flags.  The hidden rubric text,
    answer key and template answer are not serialized into the trace event.
    """

    scenario_id: str
    public_card_hash: str
    hidden_criteria_hash: str
    experience_hash: str
    public_private_boundary_hash: str
    trace_fragment_hash: str
    run_commitment_hash: str
    previous_trace_hash: str
    created_at: str
    answer_key_visible_to_engine: int = 0
    template_answer_visible_to_engine: int = 0
    hidden_rubric_visible_before_execution: int = 0
    hidden_hash_committed_before_run: int = 1

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise EvidenceError("scenario_id must be non-empty")
        for name in (
            "public_card_hash",
            "hidden_criteria_hash",
            "experience_hash",
            "public_private_boundary_hash",
            "trace_fragment_hash",
            "run_commitment_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise EvidenceError(f"{name} must be a SHA-256 hex digest")
        if self.previous_trace_hash != GENESIS_HASH and len(self.previous_trace_hash) != 64:
            raise EvidenceError("previous_trace_hash must be GENESIS or a SHA-256 hex digest")
        if self.no_answer_leakage_pass != 1:
            raise EvidenceError("pre-execution commitment violates no-answer-leakage contract")

    @property
    def no_answer_leakage_pass(self) -> int:
        return int(
            self.answer_key_visible_to_engine == 0
            and self.template_answer_visible_to_engine == 0
            and self.hidden_rubric_visible_before_execution == 0
            and self.hidden_hash_committed_before_run == 1
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "commitment_type": "pre_execution_hidden_criteria",
            "scenario_id": self.scenario_id,
            "public_card_hash": self.public_card_hash,
            "hidden_criteria_hash": self.hidden_criteria_hash,
            "experience_hash": self.experience_hash,
            "public_private_boundary_hash": self.public_private_boundary_hash,
            "trace_fragment_hash": self.trace_fragment_hash,
            "run_commitment_hash": self.run_commitment_hash,
            "previous_trace_hash": self.previous_trace_hash,
            "created_at": self.created_at,
            "no_answer_leakage_contract": {
                "answer_key_visible_to_engine": self.answer_key_visible_to_engine,
                "template_answer_visible_to_engine": self.template_answer_visible_to_engine,
                "hidden_rubric_visible_before_execution": self.hidden_rubric_visible_before_execution,
                "hidden_hash_committed_before_run": self.hidden_hash_committed_before_run,
                "no_answer_leakage_pass": self.no_answer_leakage_pass,
            },
        }


def create_pre_execution_commitment(
    *,
    scenario_id: str,
    public_card: Mapping[str, Any],
    hidden_criteria: Mapping[str, Any] | str,
    previous_trace_hash: str,
    experience_payload: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> PreExecutionCommitment:
    """Create C_hidden, C_input and C_run before scenario execution.

    C_hidden_i = H(hidden_criteria_i)
    C_input_i  = H(public_card_i)
    C_run_i    = H(C_input_i, C_hidden_i, t_i, trace_prev)

    The public card is checked for obvious answer/rubric leakage keys.  The returned
    commitment intentionally stores only digests, never hidden criteria text.
    """

    _json_safe(dict(public_card))
    if forbidden := _contains_forbidden_public_keys(public_card):
        raise EvidenceError(f"public card contains forbidden hidden/answer fields: {', '.join(forbidden)}")
    if created_at is None:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    experience_payload = dict(experience_payload or {})
    public_card_hash = sha256_hex(public_card)
    hidden_criteria_hash = sha256_hex(hidden_criteria)
    experience_hash = sha256_hex(experience_payload)
    public_private_boundary_hash = sha256_hex({
        "boundary": "public_input_hidden_criteria_separation",
        "public_card_hash": public_card_hash,
        "hidden_criteria_hash": hidden_criteria_hash,
        "experience_hash": experience_hash,
        "answer_key_visible_to_engine": 0,
        "template_answer_visible_to_engine": 0,
        "hidden_rubric_visible_before_execution": 0,
    })
    trace_fragment_hash = sha256_hex({
        "event_type": "pre_execution_commitment",
        "previous_trace_hash": previous_trace_hash,
        "public_card_hash": public_card_hash,
        "hidden_criteria_hash": hidden_criteria_hash,
        "experience_hash": experience_hash,
        "public_private_boundary_hash": public_private_boundary_hash,
    })
    run_commitment_hash = sha256_hex({
        "C_input_i": public_card_hash,
        "C_hidden_i": hidden_criteria_hash,
        "created_at": created_at,
        "trace_prev": previous_trace_hash,
    })
    return PreExecutionCommitment(
        scenario_id=scenario_id,
        public_card_hash=public_card_hash,
        hidden_criteria_hash=hidden_criteria_hash,
        experience_hash=experience_hash,
        public_private_boundary_hash=public_private_boundary_hash,
        trace_fragment_hash=trace_fragment_hash,
        run_commitment_hash=run_commitment_hash,
        previous_trace_hash=previous_trace_hash,
        created_at=created_at,
    )


def verify_pre_execution_commitment(
    commitment: PreExecutionCommitment,
    *,
    public_card: Mapping[str, Any],
    hidden_criteria: Mapping[str, Any] | str,
    experience_payload: Mapping[str, Any] | None = None,
) -> bool:
    """Recompute hashes and verify a pre-execution commitment."""

    try:
        expected = create_pre_execution_commitment(
            scenario_id=commitment.scenario_id,
            public_card=public_card,
            hidden_criteria=hidden_criteria,
            experience_payload=experience_payload,
            previous_trace_hash=commitment.previous_trace_hash,
            created_at=commitment.created_at,
        )
    except EvidenceError:
        return False
    return expected == commitment


def commitment_from_payload(payload: Mapping[str, Any]) -> PreExecutionCommitment:
    """Rehydrate a PreExecutionCommitment from a trace payload."""

    contract = dict(payload.get("no_answer_leakage_contract", {}))
    return PreExecutionCommitment(
        scenario_id=str(payload["scenario_id"]),
        public_card_hash=str(payload["public_card_hash"]),
        hidden_criteria_hash=str(payload["hidden_criteria_hash"]),
        experience_hash=str(payload["experience_hash"]),
        public_private_boundary_hash=str(payload["public_private_boundary_hash"]),
        trace_fragment_hash=str(payload["trace_fragment_hash"]),
        run_commitment_hash=str(payload["run_commitment_hash"]),
        previous_trace_hash=str(payload["previous_trace_hash"]),
        created_at=str(payload["created_at"]),
        answer_key_visible_to_engine=int(contract.get("answer_key_visible_to_engine", 1)),
        template_answer_visible_to_engine=int(contract.get("template_answer_visible_to_engine", 1)),
        hidden_rubric_visible_before_execution=int(contract.get("hidden_rubric_visible_before_execution", 1)),
        hidden_hash_committed_before_run=int(contract.get("hidden_hash_committed_before_run", 0)),
    )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: str
    digest: str
    status: EvidenceStatus = EvidenceStatus.SUPPORTED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.evidence_id.strip():
            raise EvidenceError("evidence_id must be non-empty")
        if not self.source or not self.source.strip():
            raise EvidenceError("evidence source must be non-empty")
        if not self.digest or not self.digest.strip():
            raise EvidenceError("evidence digest must be non-empty")
        _json_safe(dict(self.metadata))

    @property
    def supported(self) -> bool:
        return self.status == EvidenceStatus.SUPPORTED

    def as_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "digest": self.digest,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    payload: Mapping[str, Any]
    previous_hash: str = GENESIS_HASH
    sequence: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    def __post_init__(self) -> None:
        if not self.event_type or not self.event_type.strip():
            raise EvidenceError("trace event_type must be non-empty")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise EvidenceError("trace sequence must be a non-negative integer")
        _json_safe(dict(self.payload))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
            "sequence": self.sequence,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    def event_hash(self) -> str:
        return sha256_hex(self.canonical_bytes())


@dataclass(frozen=True)
class TraceSnapshot:
    events: tuple[TraceEvent, ...]
    head: str
    count: int
    snapshot_hash: str


class HashChain:
    """Append-only trace chain with deterministic verification."""

    def __init__(self, events: Sequence[TraceEvent] | None = None) -> None:
        self.events: list[TraceEvent] = []
        if events:
            for event in events:
                self._append_existing(event)

    @property
    def head(self) -> str:
        return self.events[-1].event_hash() if self.events else GENESIS_HASH

    @property
    def count(self) -> int:
        return len(self.events)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            payload=dict(payload),
            previous_hash=self.head,
            sequence=len(self.events),
        )
        self.events.append(event)
        return event

    def append_pre_execution_commitment(
        self,
        *,
        scenario_id: str,
        public_card: Mapping[str, Any],
        hidden_criteria: Mapping[str, Any] | str,
        experience_payload: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> tuple[TraceEvent, PreExecutionCommitment]:
        commitment = create_pre_execution_commitment(
            scenario_id=scenario_id,
            public_card=public_card,
            hidden_criteria=hidden_criteria,
            experience_payload=experience_payload,
            previous_trace_hash=self.head,
            created_at=created_at,
        )
        event = self.append("pre_execution_commitment", commitment.as_payload())
        return event, commitment

    def append_scenario_execution(
        self,
        *,
        run_commitment_hash: str,
        scenario_id: str,
        public_output_hash: str,
        decision: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        return self.append(
            "scenario_execution",
            {
                "run_commitment_hash": run_commitment_hash,
                "scenario_id": scenario_id,
                "public_output_hash": public_output_hash,
                "decision": decision,
                "metrics": dict(metrics or {}),
            },
        )


    def append_claim_support_report(self, report: ClaimSupportReport) -> TraceEvent:
        """Append an evidence-only claim-support report to the trace."""

        return self.append("claim_support_report", report.as_payload())

    def append_source_threshold_calibration(self, report: SourceThresholdCalibrationReport) -> TraceEvent:
        """Append an evidence threshold calibration report to the trace."""

        return self.append("source_threshold_calibration", report.as_payload())

    def _append_existing(self, event: TraceEvent) -> None:
        expected_previous = self.head
        expected_sequence = len(self.events)
        if event.previous_hash != expected_previous:
            raise EvidenceError("trace previous_hash mismatch")
        if event.sequence != expected_sequence:
            raise EvidenceError("trace sequence mismatch")
        self.events.append(event)

    def snapshot(self) -> TraceSnapshot:
        payload = {
            "count": self.count,
            "event_hashes": [event.event_hash() for event in self.events],
            "head": self.head,
        }
        return TraceSnapshot(
            events=tuple(self.events),
            head=self.head,
            count=self.count,
            snapshot_hash=sha256_hex(payload),
        )

    def verify(self) -> bool:
        return TraceVerifier.verify(self.events)


# Backward-compatible name from final clean release/2.  It now uses the governance boundary hash chain.
class TraceLog(HashChain):
    pass


class TraceVerifier:
    """Streaming verifier for governance boundary protected traces."""

    @staticmethod
    def verify(events: Sequence[TraceEvent]) -> bool:
        previous = GENESIS_HASH
        for expected_sequence, event in enumerate(events):
            if event.sequence != expected_sequence:
                return False
            if event.previous_hash != previous:
                return False
            try:
                event.event_hash()
            except EvidenceError:
                return False
            previous = event.event_hash()
        return True

    @staticmethod
    def verify_snapshot(snapshot: TraceSnapshot) -> bool:
        if snapshot.count != len(snapshot.events):
            return False
        if snapshot.head != (snapshot.events[-1].event_hash() if snapshot.events else GENESIS_HASH):
            return False
        expected = sha256_hex({
            "count": snapshot.count,
            "event_hashes": [event.event_hash() for event in snapshot.events],
            "head": snapshot.head,
        })
        return snapshot.snapshot_hash == expected and TraceVerifier.verify(snapshot.events)

    @staticmethod
    def verify_hidden_commitments(events: Sequence[TraceEvent]) -> bool:
        """Verify hidden-criteria commitments precede scenario execution events."""

        if not TraceVerifier.verify(events):
            return False
        committed: set[str] = set()
        for event in events:
            if event.event_type == "pre_execution_commitment":
                try:
                    commitment = commitment_from_payload(event.payload)
                except (EvidenceError, KeyError, TypeError, ValueError):
                    return False
                if commitment.previous_trace_hash != event.previous_hash:
                    return False
                committed.add(commitment.run_commitment_hash)
                continue
            if event.event_type in {"scenario_execution", "after_run_audit", "scenario_result"}:
                run_hash = event.payload.get("run_commitment_hash")
                if not isinstance(run_hash, str) or run_hash not in committed:
                    return False
        return True


def tamper_event(event: TraceEvent, payload_update: Mapping[str, Any]) -> TraceEvent:
    """Testing utility: return an event with changed payload and unchanged chain metadata."""

    payload = dict(event.payload)
    payload.update(dict(payload_update))
    return replace(event, payload=payload)
