"""P16 L3 rule promotion boundary for HTCE-Origin.

L3 may propose semantic rule candidates, but it must never become an
answer/fact/action authority.  This module implements a closed promotion gate:

    L3 candidate rule
    -> theorem-layer validation
    -> evidence support check
    -> contradiction check
    -> policy gate
    -> provisional semantic rule only

No mutation of L1/L2/L3, no fact commit, no answer authorization, and no real
actuator authority are performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Iterable, Mapping, Sequence

from htce_origin.governance.proof import Judgment, ProofObject, QueryProofResult, Statement, normalize_statement


class L3RulePromotionError(ValueError):
    """Raised when an L3 rule-promotion boundary contract is violated."""


class L3RulePromotionStatus(str, Enum):
    PROVISIONAL = "provisional_semantic_rule"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class L3RuleCandidate:
    """Non-authoritative semantic rule candidate emitted by L3.

    The candidate is not a fact, not an answer, and not an action.  It can only
    enter the runtime as a provisional hypothesis-grade semantic rule after all
    P16 gates pass.
    """

    statement: Statement | str
    support_count_raw: int
    trace_ids: tuple[str, ...]
    source_rule_id: str = ""
    l3_state_digest: str = ""
    candidate_id: str = ""
    proposed_by: str = "l3_semantic_cortex"
    provisional_only: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", normalize_statement(self.statement))
        object.__setattr__(self, "trace_ids", tuple(str(item) for item in self.trace_ids))
        if int(self.support_count_raw) < 0:
            raise L3RulePromotionError("support_count_raw must be non-negative")
        object.__setattr__(self, "support_count_raw", int(self.support_count_raw))
        if self.provisional_only != 1:
            raise L3RulePromotionError("L3 candidates must preserve provisional_only = 1")
        if not self.proposed_by:
            raise L3RulePromotionError("proposed_by must be non-empty")
        if not self.candidate_id:
            digest = hashlib.sha256()
            digest.update(self.statement.canonical().encode("utf-8"))
            digest.update(str(self.support_count_raw).encode("ascii"))
            digest.update(self.source_rule_id.encode("utf-8"))
            digest.update(self.l3_state_digest.encode("utf-8"))
            for trace_id in self.trace_ids:
                digest.update(b"|")
                digest.update(trace_id.encode("utf-8"))
            object.__setattr__(self, "candidate_id", digest.hexdigest())

    @classmethod
    def from_rule_candidate(cls, candidate: object, *, l3_state_digest: str = "") -> "L3RuleCandidate":
        """Adapt a sleep-consolidation RuleCandidate without granting authority."""

        statement = getattr(candidate, "statement")
        trace_ids = tuple(str(item) for item in getattr(candidate, "trace_ids", ()))
        support_count = int(getattr(candidate, "support_count", len(trace_ids)))
        return cls(
            statement=statement,
            support_count_raw=support_count,
            trace_ids=trace_ids,
            source_rule_id=str(getattr(candidate, "rule_id", "")),
            l3_state_digest=str(l3_state_digest),
        )

    @property
    def claim_id(self) -> str:
        return "l3_rule:" + self.statement.canonical()

    def as_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "claim_id": self.claim_id,
            "l3_state_digest": self.l3_state_digest,
            "proposed_by": self.proposed_by,
            "provisional_only": self.provisional_only,
            "source_rule_id": self.source_rule_id,
            "statement": self.statement.canonical(),
            "support_count_raw": self.support_count_raw,
            "trace_ids": list(self.trace_ids),
        }


@dataclass(frozen=True)
class L3RuleSupportReport:
    """Theorem/evidence support report for a candidate.

    Counts are raw integers.  Basis-point scores are intentionally not used as
    decision inputs in the P13/P16 path.
    """

    candidate_id: str
    theorem_validated: bool
    proof_valid: bool
    proof_quarantined: bool
    hypothesis_allowed: bool
    proof_id: str
    evidence_supported: bool
    support_count_raw: int
    required_support_raw: int
    trace_ids: tuple[str, ...]
    reason: str

    @property
    def passed(self) -> bool:
        return self.theorem_validated and self.evidence_supported and not self.proof_quarantined

    def as_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_supported": self.evidence_supported,
            "hypothesis_allowed": self.hypothesis_allowed,
            "passed": self.passed,
            "proof_id": self.proof_id,
            "proof_quarantined": self.proof_quarantined,
            "proof_valid": self.proof_valid,
            "reason": self.reason,
            "required_support_raw": self.required_support_raw,
            "support_count_raw": self.support_count_raw,
            "theorem_validated": self.theorem_validated,
            "trace_ids": list(self.trace_ids),
        }


@dataclass(frozen=True)
class L3RuleConflictReport:
    """Contradiction report for a candidate."""

    candidate_id: str
    contradiction_found: bool
    contradiction_count_raw: int
    contradiction_statements: tuple[str, ...]
    reason: str

    @property
    def passed(self) -> bool:
        return not self.contradiction_found

    def as_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "contradiction_count_raw": self.contradiction_count_raw,
            "contradiction_found": self.contradiction_found,
            "contradiction_statements": list(self.contradiction_statements),
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class L3RulePromotionDecision:
    """Final P16 decision.

    The hard authority flags are always false for L3 promotion.  A promoted
    rule is a provisional semantic rule only.
    """

    candidate: L3RuleCandidate
    status: L3RulePromotionStatus
    support_report: L3RuleSupportReport
    conflict_report: L3RuleConflictReport
    policy_allowed: bool
    policy_trace_id: str | None
    trace_id: str | None = None
    reason: str = ""
    may_answer: bool = False
    may_commit_l2_fact: bool = False
    may_execute_real_action: bool = False

    def __post_init__(self) -> None:
        if self.may_answer or self.may_commit_l2_fact or self.may_execute_real_action:
            raise L3RulePromotionError("L3 rule promotion cannot grant answer/fact/action authority")
        if self.status == L3RulePromotionStatus.PROVISIONAL:
            if not self.support_report.passed or not self.conflict_report.passed or not self.policy_allowed:
                raise L3RulePromotionError("provisional promotion requires all P16 gates to pass")

    @property
    def provisional_promoted(self) -> bool:
        return self.status == L3RulePromotionStatus.PROVISIONAL

    def as_payload(self) -> dict[str, object]:
        return {
            "authority_boundary": {
                "may_answer": self.may_answer,
                "may_commit_l2_fact": self.may_commit_l2_fact,
                "may_execute_real_action": self.may_execute_real_action,
            },
            "candidate": self.candidate.as_payload(),
            "conflict_report": self.conflict_report.as_payload(),
            "policy_allowed": self.policy_allowed,
            "policy_trace_id": self.policy_trace_id,
            "provisional_promoted": self.provisional_promoted,
            "reason": self.reason,
            "status": self.status.value,
            "support_report": self.support_report.as_payload(),
            "trace_id": self.trace_id,
        }


def build_l3_rule_support_report(
    candidate: L3RuleCandidate,
    authorization: QueryProofResult,
    *,
    required_support_raw: int,
) -> L3RuleSupportReport:
    """Build a support report from theorem authorization plus raw trace support."""

    evidence_supported = candidate.support_count_raw >= int(required_support_raw) and bool(candidate.trace_ids)
    theorem_validated = bool((authorization.answer_allowed or authorization.hypothesis_allowed) and not authorization.proof.quarantined)
    reasons: list[str] = []
    if theorem_validated:
        reasons.append("theorem layer accepted candidate as proof/hypothesis boundary")
    else:
        reasons.append(authorization.reason or "theorem layer did not validate candidate")
    if evidence_supported:
        reasons.append("raw trace support threshold satisfied")
    else:
        reasons.append("raw trace support threshold failed")
    return L3RuleSupportReport(
        candidate_id=candidate.candidate_id,
        theorem_validated=theorem_validated,
        proof_valid=authorization.proof.valid,
        proof_quarantined=authorization.proof.quarantined,
        hypothesis_allowed=authorization.hypothesis_allowed,
        proof_id=authorization.proof.proof_id,
        evidence_supported=evidence_supported,
        support_count_raw=candidate.support_count_raw,
        required_support_raw=int(required_support_raw),
        trace_ids=candidate.trace_ids,
        reason="; ".join(reasons),
    )


def build_l3_rule_conflict_report(
    candidate: L3RuleCandidate,
    judgments: Iterable[Judgment],
    *,
    latest_state: Mapping[tuple[str, str], str] | None = None,
) -> L3RuleConflictReport:
    """Detect direct theorem/memory contradictions without mutating state."""

    contradictions: list[str] = []
    statement = candidate.statement
    if statement.negated:
        contradictions.append("negated L3 candidate cannot be promoted as provisional semantic rule")
    opposite = statement.negate()
    for judgment in judgments:
        if judgment.authoritative and judgment.statement == opposite:
            contradictions.append("authoritative opposite judgment: " + judgment.statement.canonical())
    if latest_state is not None and not statement.negated and len(statement.args) == 2:
        subject, obj = statement.args
        active_obj = latest_state.get((subject.lower(), statement.predicate.lower()))
        if active_obj is not None and active_obj != obj.lower():
            contradictions.append(f"latest-state conflict: {statement.predicate}({subject},{active_obj})")
    return L3RuleConflictReport(
        candidate_id=candidate.candidate_id,
        contradiction_found=bool(contradictions),
        contradiction_count_raw=len(contradictions),
        contradiction_statements=tuple(contradictions),
        reason="; ".join(contradictions) if contradictions else "no theorem/latest-state contradiction detected",
    )


__all__ = [
    "L3RuleCandidate",
    "L3RuleConflictReport",
    "L3RulePromotionDecision",
    "L3RulePromotionError",
    "L3RulePromotionStatus",
    "L3RuleSupportReport",
    "build_l3_rule_conflict_report",
    "build_l3_rule_support_report",
]
