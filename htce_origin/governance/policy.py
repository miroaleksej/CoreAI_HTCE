"""governance boundary policy, evidence and claim-boundary gates.

This module evaluates AIR/VM candidate events without committing to L2/L3.  It
returns traceable decisions only: answer, hypothesis, refusal, simulated action
candidate, or blocked real action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from htce_origin.governance.evidence import EvidenceRecord, HashChain


class PolicyError(ValueError):
    """Raised when a policy request is structurally invalid."""


class DecisionKind(str, Enum):
    ANSWER = "answer"
    HYPOTHESIS = "hypothesis"
    ASK_CLARIFICATION = "ask_clarification"
    REFUSE = "refuse"
    ACT_SIMULATED = "act_simulated"
    BLOCK_REAL_ACTION = "block_real_action"
    SLEEP_REQUIRED = "sleep_required"


class RequestKind(str, Enum):
    ANSWER = "answer"
    COMMIT = "commit"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    QUERY = "query"
    SIMULATED_ACTION = "simulated_action"
    REAL_ACTION = "real_action"


class ImmuneAction(str, Enum):
    ALLOW = "allow"
    SUPERSEDE = "supersede"
    QUARANTINE = "quarantine"
    BLOCK_REAL_ACTION = "block_real_action"


@dataclass(frozen=True)
class FactCandidate:
    """Bounded fact candidate inspected by the policy immune gate.

    The immune layer does not commit or mutate memory. It only classifies a
    candidate as allowed, superseding, quarantined, or unsafe.
    """

    subject: str
    relation: str
    object: str
    evidence_id: str | None = None
    confidence_bp: int = 10000
    revision: int = 0
    wants_real_action: bool = False

    def __post_init__(self) -> None:
        if not self.subject or not self.relation or not self.object:
            raise PolicyError("fact candidate requires subject, relation and object")
        if not 0 <= int(self.confidence_bp) <= 10000:
            raise PolicyError("confidence_bp must be in [0, 10000]")
        object.__setattr__(self, "confidence_bp", int(self.confidence_bp))
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject.casefold(), self.relation.casefold())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], evidence_id: str | None = None, *, wants_real_action: bool = False) -> "FactCandidate | None":
        subject = payload.get("subject")
        relation = payload.get("relation")
        obj = payload.get("object")
        if subject is None or relation is None or obj is None:
            return None
        return cls(
            str(subject),
            str(relation),
            str(obj),
            evidence_id=evidence_id,
            confidence_bp=int(payload.get("confidence_bp", 10000)),
            revision=int(payload.get("revision", 0)),
            wants_real_action=bool(payload.get("wants_real_action", wants_real_action)),
        )

    @classmethod
    def from_record(cls, record: Any, *, confidence_bp: int = 10000) -> "FactCandidate":
        """Build an immune-gate candidate from a memory record.

        Memory records are semantic state records, not policy decisions. This
        adapter exposes only the bounded fact tuple required by the immune gate:
        (subject, relation, object, evidence_id, confidence_bp, revision).
        """
        fact = record.delta.fact
        return cls(
            fact.subject.value,
            fact.relation.value,
            fact.object.value,
            evidence_id=fact.evidence.value,
            confidence_bp=int(getattr(record, "confidence_bp", getattr(fact, "confidence_bp", confidence_bp))),
            revision=int(getattr(record, "revision", 0)),
            wants_real_action=False,
        )


@dataclass(frozen=True)
class ImmuneDecision:
    action: ImmuneAction
    passed: bool
    reason: str
    code: str
    candidate_key: tuple[str, str] | None = None
    active_object: str | None = None
    candidate_object: str | None = None

    @property
    def quarantined(self) -> bool:
        return self.action == ImmuneAction.QUARANTINE

    @property
    def blocks_real_action(self) -> bool:
        return self.action == ImmuneAction.BLOCK_REAL_ACTION

    def as_payload(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "passed": self.passed,
            "reason": self.reason,
            "code": self.code,
            "candidate_key": list(self.candidate_key) if self.candidate_key else None,
            "active_object": self.active_object,
            "candidate_object": self.candidate_object,
        }


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str = ""
    code: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason, "code": self.code}


@dataclass(frozen=True)
class PolicyRequest:
    kind: RequestKind | str
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None
    supported: bool = False
    is_hypothesis: bool = False
    wants_real_action: bool = False
    source: str = "air_vm_candidate"

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", RequestKind(self.kind))
        if not self.source:
            raise PolicyError("policy request source must be non-empty")


@dataclass(frozen=True)
class PolicyDecision:
    kind: DecisionKind
    gates: tuple[GateResult, ...]
    reason: str = ""
    trace_id: str | None = None
    is_hypothesis: bool = False

    @property
    def allowed(self) -> bool:
        return self.kind in {DecisionKind.ANSWER, DecisionKind.HYPOTHESIS, DecisionKind.ACT_SIMULATED}

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision": self.kind.value,
            "gates": [gate.as_payload() for gate in self.gates],
            "is_hypothesis": self.is_hypothesis,
            "reason": self.reason,
            "trace_id": self.trace_id,
        }


class CognitiveImmuneSystem:
    """Policy-level cognitive immune system.

    Mathematically, for a candidate fact f=(s,r,o,evidence,confidence,revision),
    the immune gate implements:

        ImmuneGate(f) = confidence(f) >= tau_conf
            and not forbidden_relation(f)
            and not conflict_without_evidence(f)
            and not unsafe_action(f)

    It never mutates L1/L2/L3 and never commits facts. Supersession and
    quarantine decisions are returned as policy evidence for downstream memory.
    """

    def __init__(
        self,
        *,
        min_confidence_bp: int = 7000,
        forbidden_relations: tuple[str, ...] | None = None,
        unsafe_action_relations: tuple[str, ...] | None = None,
    ) -> None:
        if not 0 <= int(min_confidence_bp) <= 10000:
            raise PolicyError("min_confidence_bp must be in [0, 10000]")
        self.min_confidence_bp = int(min_confidence_bp)
        self.forbidden_relations = frozenset(
            rel.casefold()
            for rel in (forbidden_relations or (
                "forbidden_relation",
                "bypass_claim_boundary",
                "unverified_external_commit",
                "secret_exfiltration",
            ))
        )
        self.unsafe_action_relations = frozenset(
            rel.casefold()
            for rel in (unsafe_action_relations or (
                "real_world_actuator",
                "execute_real_action",
                "move_real_actuator",
                "unsafe_action",
                "weapon_action",
            ))
        )

    def evaluate_fact(self, candidate: FactCandidate, active_fact: FactCandidate | Mapping[str, Any] | None = None) -> ImmuneDecision:
        relation = candidate.relation.casefold()
        if candidate.wants_real_action or relation in self.unsafe_action_relations:
            return ImmuneDecision(
                ImmuneAction.BLOCK_REAL_ACTION,
                False,
                "unsafe real-action candidate blocked by immune gate",
                "IMMUNE_UNSAFE_REAL_ACTION",
                candidate.key,
                candidate_object=candidate.object,
            )
        if relation in self.forbidden_relations:
            return ImmuneDecision(
                ImmuneAction.QUARANTINE,
                False,
                "forbidden relation quarantined by immune gate",
                "IMMUNE_FORBIDDEN_RELATION",
                candidate.key,
                candidate_object=candidate.object,
            )
        if candidate.confidence_bp < self.min_confidence_bp:
            return ImmuneDecision(
                ImmuneAction.QUARANTINE,
                False,
                "low confidence fact quarantined by immune gate",
                "IMMUNE_LOW_CONFIDENCE",
                candidate.key,
                candidate_object=candidate.object,
            )
        if active_fact is not None:
            active = self._coerce_active_fact(active_fact)
            if active and active.key == candidate.key and active.object != candidate.object:
                if candidate.evidence_id and candidate.confidence_bp >= active.confidence_bp:
                    return ImmuneDecision(
                        ImmuneAction.SUPERSEDE,
                        True,
                        "newer supported fact supersedes active fact",
                        "IMMUNE_SUPERSEDE",
                        candidate.key,
                        active_object=active.object,
                        candidate_object=candidate.object,
                    )
                return ImmuneDecision(
                    ImmuneAction.QUARANTINE,
                    False,
                    "conflict without stronger evidence quarantined by immune gate",
                    "IMMUNE_CONFLICT_WITHOUT_EVIDENCE",
                    candidate.key,
                    active_object=active.object,
                    candidate_object=candidate.object,
                )
        return ImmuneDecision(
            ImmuneAction.ALLOW,
            True,
            "immune gate accepted candidate",
            "IMMUNE_OK",
            candidate.key,
            candidate_object=candidate.object,
        )

    def evaluate_request(self, request: PolicyRequest, active_fact: FactCandidate | Mapping[str, Any] | None = None) -> ImmuneDecision:
        if request.kind == RequestKind.REAL_ACTION or request.wants_real_action:
            return ImmuneDecision(
                ImmuneAction.BLOCK_REAL_ACTION,
                False,
                "real action blocked by immune gate",
                "IMMUNE_UNSAFE_REAL_ACTION",
            )
        candidate = FactCandidate.from_payload(request.payload, request.evidence_id, wants_real_action=request.wants_real_action)
        if candidate is None:
            return ImmuneDecision(ImmuneAction.ALLOW, True, "immune gate not required for non-fact payload", "IMMUNE_NOT_REQUIRED")
        return self.evaluate_fact(candidate, active_fact=active_fact)

    @staticmethod
    def _coerce_active_fact(active_fact: FactCandidate | Mapping[str, Any]) -> FactCandidate | None:
        if isinstance(active_fact, FactCandidate):
            return active_fact
        return FactCandidate.from_payload(active_fact, active_fact.get("evidence_id"))


class ImmuneGate:
    """Quarantine low-confidence, forbidden, conflicting or unsafe candidates."""

    def __init__(self, immune_system: CognitiveImmuneSystem | None = None) -> None:
        self.immune_system = immune_system or CognitiveImmuneSystem()

    def evaluate(self, request: PolicyRequest, active_fact: FactCandidate | Mapping[str, Any] | None = None) -> GateResult:
        decision = self.immune_system.evaluate_request(request, active_fact=active_fact)
        return GateResult("immune", decision.passed, decision.reason, decision.code)


class TypeGate:
    """Reject structurally invalid policy requests."""

    def evaluate(self, request: PolicyRequest) -> GateResult:
        if request.kind in {RequestKind.ANSWER, RequestKind.CLAIM, RequestKind.COMMIT} and not request.payload:
            return GateResult("type", False, "payload is required", "TYPE_PAYLOAD_REQUIRED")
        if request.kind == RequestKind.REAL_ACTION or request.wants_real_action:
            return GateResult("type", True, "real action shape detected for boundary gate", "TYPE_OK_REAL_ACTION")
        return GateResult("type", True, "type accepted", "TYPE_OK")


class EvidenceGate:
    """Require supported evidence for facts, answers and claims."""

    def __init__(self, records: Mapping[str, EvidenceRecord] | None = None) -> None:
        self.records = dict(records or {})

    def evaluate(self, request: PolicyRequest) -> GateResult:
        if request.kind in {RequestKind.HYPOTHESIS, RequestKind.QUERY}:
            return GateResult("evidence", True, "evidence not required for hypothesis/query boundary", "EVIDENCE_NOT_REQUIRED")
        if not request.evidence_id:
            return GateResult("evidence", False, "missing evidence", "EVIDENCE_MISSING")
        record = self.records.get(request.evidence_id)
        if record is not None and not record.supported:
            return GateResult("evidence", False, "evidence is not supported", "EVIDENCE_UNSUPPORTED")
        if record is None and not request.supported:
            return GateResult("evidence", False, "evidence id has no supported record", "EVIDENCE_RECORD_MISSING")
        return GateResult("evidence", True, "supported evidence accepted", "EVIDENCE_OK")


class ClaimGate:
    """Enforce claim boundary: unsupported claims refuse, hypotheses stay labelled."""

    def evaluate(self, request: PolicyRequest) -> GateResult:
        if request.kind == RequestKind.HYPOTHESIS or request.is_hypothesis:
            return GateResult("claim", True, "hypothesis must remain explicitly marked", "CLAIM_HYPOTHESIS_ONLY")
        if request.kind in {RequestKind.ANSWER, RequestKind.CLAIM, RequestKind.COMMIT} and not request.supported:
            return GateResult("claim", False, "unsupported claim refused", "CLAIM_UNSUPPORTED")
        return GateResult("claim", True, "claim boundary accepted", "CLAIM_OK")


class PolicyGate:
    """Block real-world actions and allow only simulation-first action candidates."""

    def evaluate(self, request: PolicyRequest) -> GateResult:
        if request.kind == RequestKind.REAL_ACTION or request.wants_real_action:
            return GateResult("policy", False, "real action blocked by simulation-first policy", "POLICY_REAL_ACTION_BLOCKED")
        return GateResult("policy", True, "policy accepted", "POLICY_OK")


class TraceGate:
    """Append every decision to a protected hash chain."""

    def __init__(self, chain: HashChain | None = None) -> None:
        self.chain = chain or HashChain()

    def append_decision(self, request: PolicyRequest, decision: DecisionKind, gates: tuple[GateResult, ...], reason: str, *, is_hypothesis: bool) -> str:
        event = self.chain.append("policy_decision", {
            "decision": decision.value,
            "gates": [gate.as_payload() for gate in gates],
            "is_hypothesis": is_hypothesis,
            "payload": dict(request.payload),
            "reason": reason,
            "request_kind": request.kind.value,
            "source": request.source,
        })
        return event.event_hash()


class PolicyEngine:
    """governance boundary gate bundle.

    The engine is intentionally side-effect-light: its only side effect is trace
    append.  It does not mutate memory, L1/L2/L3, world model, topology or AIR.
    """

    def __init__(
        self,
        evidence_records: Mapping[str, EvidenceRecord] | None = None,
        trace: HashChain | None = None,
        immune_system: CognitiveImmuneSystem | None = None,
    ) -> None:
        self.type_gate = TypeGate()
        self.evidence_gate = EvidenceGate(evidence_records)
        self.claim_gate = ClaimGate()
        self.policy_gate = PolicyGate()
        self.immune_gate = ImmuneGate(immune_system)
        self.trace_gate = TraceGate(trace)

    @property
    def trace(self) -> HashChain:
        return self.trace_gate.chain

    def evaluate(self, request: PolicyRequest, *, active_fact: FactCandidate | Mapping[str, Any] | None = None) -> PolicyDecision:
        gates = (
            self.type_gate.evaluate(request),
            self.evidence_gate.evaluate(request),
            self.claim_gate.evaluate(request),
            self.policy_gate.evaluate(request),
            self.immune_gate.evaluate(request, active_fact=active_fact),
        )
        failed = tuple(gate for gate in gates if not gate.passed)
        if failed:
            if any(gate.code in {"POLICY_REAL_ACTION_BLOCKED", "IMMUNE_UNSAFE_REAL_ACTION"} for gate in failed):
                decision_kind = DecisionKind.BLOCK_REAL_ACTION
            else:
                decision_kind = DecisionKind.REFUSE
            reason = "; ".join(gate.reason for gate in failed)
            is_hypothesis = False
        elif request.kind == RequestKind.HYPOTHESIS or request.is_hypothesis:
            decision_kind = DecisionKind.HYPOTHESIS
            reason = "hypothesis marked; not committed as fact"
            is_hypothesis = True
        elif request.kind == RequestKind.SIMULATED_ACTION:
            decision_kind = DecisionKind.ACT_SIMULATED
            reason = "simulated action candidate accepted"
            is_hypothesis = False
        else:
            decision_kind = DecisionKind.ANSWER
            reason = "supported answer accepted"
            is_hypothesis = False

        trace_id = self.trace_gate.append_decision(request, decision_kind, gates, reason, is_hypothesis=is_hypothesis)
        return PolicyDecision(decision_kind, gates, reason, trace_id, is_hypothesis)


# Backward-compatible alias used by early skeleton tests/imports.
GateBundle = PolicyEngine
