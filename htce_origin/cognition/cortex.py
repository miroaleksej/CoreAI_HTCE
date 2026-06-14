"""associative cortex associative cortex for HTCE-Origin Clean Body.

The associative cortex is deliberately a candidate generator, not an answer
authority.  It retrieves content-addressable candidate memories, maintains a
weighted temporal/causal/analogy long-memory graph, and marks candidates as
hypotheses unless a proof/evidence layer authorizes them later.

associative cortex boundaries:
- no L1/L2/L3 mutation;
- no runtime commit;
- no real actions;
- no claim that association equals fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from htce_origin.kernel.core import DEFAULT_TORUS_DIMENSION, hash_to_phase
from htce_origin.governance.proof import Judgment, ProofObject, Statement, TheoremLayer, normalize_statement
from htce_origin.kernel.q16 import DEFAULT_MODULUS, q_distance_vector

MAX_SEQUENCE_COUNTER = 2**31 - 1


class CortexError(ValueError):
    """Raised when associative cortex input is malformed."""


class CandidateStatus(str, Enum):
    """Authorization status of a retrieved memory candidate."""

    HYPOTHESIS = "hypothesis"
    PROVEN = "proven"
    BLOCKED = "blocked"


class EdgeKind(str, Enum):
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SUPPORT = "support"
    ANALOGY = "analogy"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class AssociationScoringWeights:
    """Integer weights for graph-aware hypothesis retrieval.

    The score is intentionally a hypothesis score, not an answer authority:

    score(q, v_i) = alpha * sim_phase(q, v_i)
                  + beta  * temporal_relevance(q, v_i)
                  + gamma * causal_support(q, v_i)
                  - delta * contradiction_penalty(q, v_i)

    All coefficients are integer basis-point weights.  Their default sum is
    100 so the output remains in [0, 10000] after clamping.
    """

    phase_weight: int = 40
    temporal_weight: int = 20
    causal_weight: int = 25
    contradiction_weight: int = 15

    def __post_init__(self) -> None:
        values = (self.phase_weight, self.temporal_weight, self.causal_weight, self.contradiction_weight)
        if any(int(v) < 0 for v in values):
            raise CortexError("association scoring weights must be non-negative")
        if self.denominator <= 0:
            raise CortexError("association scoring weight denominator must be positive")

    @property
    def denominator(self) -> int:
        return int(self.phase_weight + self.temporal_weight + self.causal_weight + self.contradiction_weight)


@dataclass(frozen=True)
class MemoryNode:
    """One content-addressable memory item.

    ``statement`` is the symbolic surface.  ``phase`` is the deterministic
    toroidal content address.  ``source='association'`` means non-authoritative
    by design; such nodes can seed hypotheses but cannot authorize answers.
    """

    node_id: str
    statement: Statement
    phase: tuple[int, ...]
    evidence_ids: tuple[str, ...] = ()
    timestamp: int = 0
    tags: tuple[str, ...] = ()
    source: str = "memory"
    supported: bool = True

    def __post_init__(self) -> None:
        if not self.node_id:
            raise CortexError("node_id must be non-empty")
        if self.source not in {"memory", "association", "derived", "external"}:
            raise CortexError(f"unknown memory node source: {self.source}")
        object.__setattr__(self, "phase", tuple(int(v) % DEFAULT_MODULUS for v in self.phase))
        object.__setattr__(self, "evidence_ids", tuple(str(v) for v in self.evidence_ids))
        object.__setattr__(self, "tags", tuple(str(v).lower() for v in self.tags))

    @property
    def authoritative_hint(self) -> bool:
        return self.supported and self.source != "association" and bool(self.evidence_ids)


@dataclass(frozen=True)
class MemoryEdge:
    """A typed weighted long-memory graph edge.

    ``temporal_delta`` is an integer relative distance between events.  It can
    be zero when the edge is non-temporal.  It is never used to authorize a
    factual answer; it only affects candidate ranking.
    """

    source_id: str
    target_id: str
    kind: EdgeKind
    weight_bp: int = 10000
    evidence_id: str | None = None
    temporal_delta: int = 0

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise CortexError("edge endpoints must be non-empty")
        if not 0 <= int(self.weight_bp) <= 10000:
            raise CortexError("edge weight_bp must be in [0, 10000]")
        object.__setattr__(self, "temporal_delta", int(self.temporal_delta))


@dataclass(frozen=True)
class CandidateHypothesis:
    """A retrieval candidate emitted by the cortex.

    The central safety invariant is ``answer_authorized``.  It is false unless
    an attached proof object is valid and non-quarantined.  associative cortex does not
    commit the candidate to memory or expose it as an answer.
    """

    node_id: str
    statement: Statement
    score_bp: int
    provenance: tuple[str, ...]
    distance: int = 0
    status: CandidateStatus = CandidateStatus.HYPOTHESIS
    proof_id: str | None = None
    proof_valid: bool = False
    blocked: bool = False
    reason: str = "candidate requires proof/evidence gate"
    graph_context: tuple[str, ...] = ()
    phase_similarity_bp: int = 0
    temporal_relevance_bp: int = 0
    causal_support_bp: int = 0
    contradiction_penalty_bp: int = 0
    answer_authorized: bool = False

    @property
    def label(self) -> str:
        return self.statement.canonical()




@dataclass(frozen=True)
class CandidateTheory:
    """A bounded reflective theory made of hypothesis-grade candidates.

    ``CandidateTheory`` is not a fact store and not an answer authority.  It is
    a compact reflective bundle:

    T = (H_1, ..., H_k)

    TheoryScore(T) = (4*proof_score + 4*evidence_score + simplicity + falsifiability) // 10

    The output remains hypothesis-grade unless a separate proof/evidence/policy
    path later authorizes an answer.
    """

    hypotheses: tuple[CandidateHypothesis, ...]
    proof_score_bp: int = 0
    evidence_score_bp: int = 0
    simplicity_bp: int = 10000
    falsifiability_bp: int = 0
    theory_id: str = ""
    status: CandidateStatus = CandidateStatus.HYPOTHESIS
    answer_authorized: bool = False
    reason: str = "candidate theory is hypothesis-grade only"

    def __post_init__(self) -> None:
        hypotheses = tuple(self.hypotheses)
        if not hypotheses:
            raise CortexError("candidate theory requires at least one hypothesis")
        for field_name in ("proof_score_bp", "evidence_score_bp", "simplicity_bp", "falsifiability_bp"):
            value = int(getattr(self, field_name))
            if not 0 <= value <= 10000:
                raise CortexError(f"{field_name} must be in [0, 10000]")
            object.__setattr__(self, field_name, value)
        if self.answer_authorized:
            raise CortexError("candidate theory cannot authorize an answer")
        object.__setattr__(self, "hypotheses", hypotheses)
        if not self.theory_id:
            import hashlib

            digest = hashlib.sha256()
            for candidate in hypotheses:
                digest.update(candidate.label.encode("utf-8"))
                digest.update(b"|")
                digest.update(str(candidate.score_bp).encode("ascii"))
            digest.update(str(self.score_bp).encode("ascii"))
            object.__setattr__(self, "theory_id", "theory_" + digest.hexdigest()[:24])

    @property
    def score_bp(self) -> int:
        score = (
            4 * int(self.proof_score_bp)
            + 4 * int(self.evidence_score_bp)
            + int(self.simplicity_bp)
            + int(self.falsifiability_bp)
        ) // 10
        return max(0, min(10000, score))


@dataclass(frozen=True)
class CounterfactualQuery:
    """A reflective what-if probe that can only request a hypothesis.

    Counterfactuals are intentionally kept outside L1/L2/L3 commit logic:
    ``given`` and ``ask`` seed candidate retrieval, but the result cannot
    become a fact without proof/evidence/policy gates.
    """

    given: Statement
    ask: Statement
    query_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "given", normalize_statement(self.given))
        object.__setattr__(self, "ask", normalize_statement(self.ask))
        if not self.query_id:
            import hashlib

            digest = hashlib.sha256((self.given.canonical() + "=>" + self.ask.canonical()).encode("utf-8")).hexdigest()
            object.__setattr__(self, "query_id", "cf_" + digest[:24])


@dataclass(frozen=True)
class AnalogyTransferResult:
    """Result of analogy transfer, always hypothesis-grade.

    The result may carry a candidate and a score, but it is never a proof and
    never an L2/L3 commit.
    """

    source_node_id: str
    target_statement: Statement
    candidate: CandidateHypothesis
    transfer_score_bp: int
    status: CandidateStatus = CandidateStatus.HYPOTHESIS
    answer_authorized: bool = False
    reason: str = "analogy transfer is hypothesis-grade only"

    def __post_init__(self) -> None:
        if not self.source_node_id:
            raise CortexError("source_node_id must be non-empty")
        object.__setattr__(self, "target_statement", normalize_statement(self.target_statement))
        score = int(self.transfer_score_bp)
        if not 0 <= score <= 10000:
            raise CortexError("transfer_score_bp must be in [0, 10000]")
        object.__setattr__(self, "transfer_score_bp", score)
        if self.answer_authorized:
            raise CortexError("analogy transfer cannot authorize an answer")


@dataclass(frozen=True)
class CognitiveEvaluationReport:
    """Small reflective report over candidate theories.

    This report is diagnostic only: it can compare hypothesis bundles but cannot
    convert them into facts or answers.
    """

    theories: tuple[CandidateTheory, ...]
    best_theory_id: str | None = None
    best_score_bp: int = 0
    answer_authorized: bool = False
    reason: str = "reflective evaluation is diagnostic only"

    def __post_init__(self) -> None:
        theories = tuple(self.theories)
        object.__setattr__(self, "theories", theories)
        if self.answer_authorized:
            raise CortexError("cognitive evaluation report cannot authorize an answer")
        if theories:
            best = max(theories, key=lambda item: (item.score_bp, item.theory_id))
            object.__setattr__(self, "best_theory_id", best.theory_id)
            object.__setattr__(self, "best_score_bp", best.score_bp)
        else:
            object.__setattr__(self, "best_theory_id", None)
            object.__setattr__(self, "best_score_bp", 0)


@dataclass
class LongMemoryGraph:
    """Minimal temporal/causal graph used by the associative cortex."""

    edges: list[MemoryEdge] = field(default_factory=list)

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        self.edges.append(edge)
        return edge

    def outgoing(self, node_id: str, *, kind: EdgeKind | None = None) -> tuple[MemoryEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source_id == node_id and (kind is None or edge.kind == kind))

    def incoming(self, node_id: str, *, kind: EdgeKind | None = None) -> tuple[MemoryEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target_id == node_id and (kind is None or edge.kind == kind))

    def context_for(self, node_id: str) -> tuple[str, ...]:
        parts: list[str] = []
        for edge in self.incoming(node_id):
            parts.append(
                f"{edge.kind.value}:in:{edge.source_id}->{edge.target_id}:w={edge.weight_bp}:dt={edge.temporal_delta}"
            )
        for edge in self.outgoing(node_id):
            parts.append(
                f"{edge.kind.value}:out:{edge.source_id}->{edge.target_id}:w={edge.weight_bp}:dt={edge.temporal_delta}"
            )
        return tuple(parts)


class LongMemoryAssociationGraph(LongMemoryGraph):
    """Named alias for the weighted association graph used by the cortex."""


class AssociativeCortex:
    """Content-addressable candidate retrieval plus non-authoritative graph memory."""

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_TORUS_DIMENSION,
        modulus: int = DEFAULT_MODULUS,
        scoring_weights: AssociationScoringWeights | None = None,
    ) -> None:
        if dimension <= 0:
            raise CortexError("dimension must be positive")
        self.dimension = int(dimension)
        self.modulus = int(modulus)
        self.scoring_weights = scoring_weights or AssociationScoringWeights()
        self._nodes: dict[str, MemoryNode] = {}
        self.graph = LongMemoryAssociationGraph()
        self._sequence_counter = 0

    @property
    def nodes(self) -> tuple[MemoryNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[MemoryEdge, ...]:
        return tuple(self.graph.edges)

    def _phase_for(self, statement: Statement) -> tuple[int, ...]:
        return hash_to_phase(
            statement.canonical(),
            dimension=self.dimension,
            modulus=self.modulus,
            namespace="cortex.statement",
        )

    def _node_id_for(self, statement: Statement, evidence_ids: Sequence[str], source: str) -> str:
        evidence_part = ",".join(sorted(str(v) for v in evidence_ids))
        raw = f"{statement.canonical()}|{source}|{evidence_part}"
        phase = hash_to_phase(raw, dimension=4, modulus=self.modulus, namespace="cortex.node")
        return "cx_" + "_".join(f"{v:04x}" for v in phase)

    def remember(
        self,
        statement: Statement | str,
        *,
        evidence_id: str | None = None,
        evidence_ids: Iterable[str] = (),
        timestamp: int | None = None,
        tags: Iterable[str] = (),
        source: str = "memory",
        supported: bool = True,
    ) -> MemoryNode:
        """Insert or replace a memory candidate node.

        This is a cortex-local memory graph operation only.  It does not mutate
        L2/L3 toroidal runtime state.
        """

        st = normalize_statement(statement)
        evidences = tuple([evidence_id] if evidence_id else ()) + tuple(str(v) for v in evidence_ids)
        if timestamp is None:
            self._sequence_counter = (self._sequence_counter + 1) % MAX_SEQUENCE_COUNTER
            timestamp = self._sequence_counter
        node_id = self._node_id_for(st, evidences, source)
        node = MemoryNode(
            node_id=node_id,
            statement=st,
            phase=self._phase_for(st),
            evidence_ids=evidences,
            timestamp=int(timestamp),
            tags=tuple(tags),
            source=source,
            supported=bool(supported),
        )
        self._nodes[node_id] = node
        return node

    def remember_association(self, statement: Statement | str, *, tags: Iterable[str] = ()) -> MemoryNode:
        """Store a non-authoritative association candidate."""

        return self.remember(statement, tags=tags, source="association", supported=True)

    def link_temporal(
        self,
        earlier: MemoryNode | str,
        later: MemoryNode | str,
        *,
        weight_bp: int = 10000,
        evidence_id: str | None = None,
        temporal_delta: int = 1,
    ) -> MemoryEdge:
        return self._link(
            earlier,
            later,
            EdgeKind.TEMPORAL,
            weight_bp=weight_bp,
            evidence_id=evidence_id,
            temporal_delta=temporal_delta,
        )

    def link_causal(self, cause: MemoryNode | str, effect: MemoryNode | str, *, weight_bp: int = 10000, evidence_id: str | None = None) -> MemoryEdge:
        return self._link(cause, effect, EdgeKind.CAUSAL, weight_bp=weight_bp, evidence_id=evidence_id, temporal_delta=0)

    def link_support(self, source: MemoryNode | str, target: MemoryNode | str, *, weight_bp: int = 10000, evidence_id: str | None = None) -> MemoryEdge:
        return self._link(source, target, EdgeKind.SUPPORT, weight_bp=weight_bp, evidence_id=evidence_id, temporal_delta=0)

    def link_analogy(self, source: MemoryNode | str, target: MemoryNode | str, *, weight_bp: int = 7000, evidence_id: str | None = None) -> MemoryEdge:
        return self._link(source, target, EdgeKind.ANALOGY, weight_bp=weight_bp, evidence_id=evidence_id, temporal_delta=0)

    def link_contradiction(self, source: MemoryNode | str, target: MemoryNode | str, *, weight_bp: int = 10000, evidence_id: str | None = None) -> MemoryEdge:
        return self._link(source, target, EdgeKind.CONTRADICTION, weight_bp=weight_bp, evidence_id=evidence_id, temporal_delta=0)

    def _link(
        self,
        source: MemoryNode | str,
        target: MemoryNode | str,
        kind: EdgeKind,
        *,
        weight_bp: int,
        evidence_id: str | None,
        temporal_delta: int,
    ) -> MemoryEdge:
        source_id = source.node_id if isinstance(source, MemoryNode) else str(source)
        target_id = target.node_id if isinstance(target, MemoryNode) else str(target)
        if source_id not in self._nodes or target_id not in self._nodes:
            raise CortexError("cannot link unknown memory nodes")
        return self.graph.add_edge(
            MemoryEdge(
                source_id,
                target_id,
                kind,
                weight_bp=weight_bp,
                evidence_id=evidence_id,
                temporal_delta=temporal_delta,
            )
        )

    def retrieve(
        self,
        query: Statement | str | object,
        *,
        proof_layer: TheoremLayer | None = None,
        top_k: int = 5,
        min_score_bp: int = 0,
    ) -> tuple[CandidateHypothesis, ...]:
        """Retrieve candidate hypotheses for a query statement.

        Without a valid proof object, returned candidates are hypotheses.  A
        candidate contradicted by the supplied proof layer is blocked and
        down-ranked to zero.
        """

        if top_k <= 0:
            return ()
        query_statement = normalize_statement(query) if isinstance(query, (Statement, str)) else normalize_statement(str(query))
        query_phase = self._phase_for(query_statement)
        candidates: list[CandidateHypothesis] = []
        for node in self._nodes.values():
            raw_distance = q_distance_vector(query_phase, node.phase, self.modulus)
            phase_score = self._phase_score_bp(raw_distance)
            token_score = self._token_score_bp(query_statement, node.statement)
            sim_phase = (phase_score * 60 + token_score * 40) // 100
            temporal_relevance = self._temporal_relevance_bp(node.node_id, query_statement)
            causal_support = self._causal_support_bp(node.node_id, query_statement)
            contradiction_penalty = self._contradiction_penalty_bp(node.node_id)
            base_score = self._candidate_score_bp(
                sim_phase,
                temporal_relevance,
                causal_support,
                contradiction_penalty,
            )
            status = CandidateStatus.HYPOTHESIS
            proof_id: str | None = None
            proof_valid = False
            blocked = False
            answer_authorized = False
            reason = "candidate without proof is hypothesis"

            if node.source == "association":
                # Associations are explicitly not facts.  Give them a small
                # retrieval penalty so supported memories win ties.
                base_score = (base_score * 85) // 100
                reason = "association alone cannot prove an answer"

            if contradiction_penalty > 0:
                reason = "candidate has graph contradiction penalty and remains hypothesis"

            if proof_layer is not None:
                proof = proof_layer.prove(node.statement)
                proof_id = proof.proof_id
                proof_valid = proof.valid and not proof.quarantined
                if proof.quarantined:
                    status = CandidateStatus.BLOCKED
                    blocked = True
                    answer_authorized = False
                    base_score = 0
                    reason = proof.reason or "contradicted candidate blocked"
                elif proof_valid:
                    status = CandidateStatus.PROVEN
                    answer_authorized = True
                    base_score = min(10000, base_score + 1500)
                    reason = "candidate authorized by proof layer"
                else:
                    status = CandidateStatus.HYPOTHESIS
                    answer_authorized = False
                    reason = proof.reason or reason

            if node.authoritative_hint and proof_layer is None:
                # Evidence in cortex-local storage is useful for ranking, but
                # still not enough to authorize an answer at associative cortex.
                base_score = min(10000, base_score + 500)

            if base_score < min_score_bp and not blocked:
                continue
            provenance = tuple(filter(None, (
                f"node:{node.node_id}",
                f"source:{node.source}",
                *(f"evidence:{eid}" for eid in node.evidence_ids),
            )))
            candidates.append(
                CandidateHypothesis(
                    node_id=node.node_id,
                    statement=node.statement,
                    score_bp=max(0, min(10000, base_score)),
                    distance=raw_distance,
                    provenance=provenance,
                    status=status,
                    proof_id=proof_id,
                    proof_valid=proof_valid,
                    blocked=blocked,
                    reason=reason,
                    graph_context=self.graph.context_for(node.node_id),
                    phase_similarity_bp=sim_phase,
                    temporal_relevance_bp=temporal_relevance,
                    causal_support_bp=causal_support,
                    contradiction_penalty_bp=contradiction_penalty,
                    answer_authorized=answer_authorized,
                )
            )
        candidates.sort(key=lambda item: (item.blocked, -item.score_bp, item.distance, item.label))
        return tuple(candidates[:top_k])

    def build_candidate_theory(
        self,
        candidates: Iterable[CandidateHypothesis],
        *,
        proof_score_bp: int = 0,
        evidence_score_bp: int | None = None,
        simplicity_bp: int | None = None,
        falsifiability_bp: int = 0,
    ) -> CandidateTheory:
        """Bundle candidates into a reflective hypothesis theory.

        The method deliberately returns a hypothesis-grade object.  It does not
        call proof/evidence gates and cannot authorize an answer.
        """

        items = tuple(candidates)
        if not items:
            raise CortexError("cannot build a candidate theory without candidates")
        if evidence_score_bp is None:
            evidence_score_bp = sum(item.score_bp for item in items) // len(items)
        if simplicity_bp is None:
            # Simpler theories receive a higher score; every extra hypothesis
            # beyond the first costs 1000 basis points.
            simplicity_bp = max(0, 10000 - (len(items) - 1) * 1000)
        return CandidateTheory(
            hypotheses=items,
            proof_score_bp=int(proof_score_bp),
            evidence_score_bp=int(evidence_score_bp),
            simplicity_bp=int(simplicity_bp),
            falsifiability_bp=int(falsifiability_bp),
        )

    def evaluate_candidate_theories(self, theories: Iterable[CandidateTheory]) -> CognitiveEvaluationReport:
        """Compare candidate theories without turning any of them into facts."""

        return CognitiveEvaluationReport(tuple(theories))

    def counterfactual_hypotheses(
        self,
        query: CounterfactualQuery,
        *,
        top_k: int = 3,
    ) -> tuple[CandidateHypothesis, ...]:
        """Retrieve hypothesis-grade candidates for a counterfactual query.

        The ``given`` part is used only as an additional symbolic cue.  No state
        transition or memory commit occurs.
        """

        combined = Statement.atom(
            "counterfactual",
            query.given.canonical(),
            query.ask.canonical(),
        )
        direct = self.retrieve(query.ask, top_k=top_k)
        if direct:
            return direct
        return self.retrieve(combined, top_k=top_k)

    def transfer_by_analogy(
        self,
        source: MemoryNode | str,
        target_statement: Statement | str,
        *,
        weight_bp: int = 7000,
        evidence_id: str | None = None,
    ) -> AnalogyTransferResult:
        """Create an analogy candidate and link it to a source node.

        The target is stored as ``source='association'`` and returned as a
        hypothesis-grade transfer result only.
        """

        source_id = source.node_id if isinstance(source, MemoryNode) else str(source)
        if source_id not in self._nodes:
            raise CortexError("cannot transfer analogy from unknown source node")
        target = self.remember_association(target_statement, tags=("analogy_transfer",))
        self.link_analogy(source_id, target.node_id, weight_bp=weight_bp, evidence_id=evidence_id)
        candidate = next(item for item in self.retrieve(target.statement, top_k=10) if item.node_id == target.node_id)
        transfer_score = min(10000, (int(weight_bp) + candidate.score_bp) // 2)
        return AnalogyTransferResult(
            source_node_id=source_id,
            target_statement=target.statement,
            candidate=candidate,
            transfer_score_bp=transfer_score,
        )

    def _phase_score_bp(self, distance: int) -> int:
        max_distance = self.dimension * ((self.modulus // 2) ** 2)
        if max_distance <= 0:
            return 0
        score = 10000 - (int(distance) * 10000 // max_distance)
        return max(0, min(10000, score))

    @staticmethod
    def _tokens(statement: Statement) -> set[str]:
        return {statement.predicate.lower(), *(arg.lower() for arg in statement.args)}

    @classmethod
    def _token_score_bp(cls, query: Statement, candidate: Statement) -> int:
        q_tokens = cls._tokens(query)
        c_tokens = cls._tokens(candidate)
        if not q_tokens or not c_tokens:
            return 0
        union = q_tokens | c_tokens
        intersection = q_tokens & c_tokens
        return (len(intersection) * 10000) // len(union)

    def _edge_relevance_bp(self, node_id: str, edge: MemoryEdge, query: Statement) -> int:
        source = self._nodes.get(edge.source_id)
        target = self._nodes.get(edge.target_id)
        related = target if edge.source_id == node_id else source
        # Keep the relevance symbolic and integer-only: edge weight times token overlap.
        overlap = 10000
        if related is not None:
            overlap = self._token_score_bp(query, related.statement)
            if overlap == 0:
                overlap = 5000
        score = int(edge.weight_bp) * int(overlap) // 10000
        if edge.kind == EdgeKind.TEMPORAL:
            decay = min(7000, abs(int(edge.temporal_delta)) * 500)
            score = max(0, score - decay)
        return max(0, min(10000, score))

    def _temporal_relevance_bp(self, node_id: str, query: Statement) -> int:
        scores = [
            self._edge_relevance_bp(node_id, edge, query)
            for edge in (*self.graph.incoming(node_id, kind=EdgeKind.TEMPORAL), *self.graph.outgoing(node_id, kind=EdgeKind.TEMPORAL))
        ]
        return max(scores) if scores else 0

    def _causal_support_bp(self, node_id: str, query: Statement) -> int:
        support_kinds = (EdgeKind.CAUSAL, EdgeKind.SUPPORT, EdgeKind.ANALOGY)
        scores: list[int] = []
        for kind in support_kinds:
            scores.extend(self._edge_relevance_bp(node_id, edge, query) for edge in self.graph.incoming(node_id, kind=kind))
            scores.extend(self._edge_relevance_bp(node_id, edge, query) for edge in self.graph.outgoing(node_id, kind=kind))
        return max(scores) if scores else 0

    def _contradiction_penalty_bp(self, node_id: str) -> int:
        edges = (*self.graph.incoming(node_id, kind=EdgeKind.CONTRADICTION), *self.graph.outgoing(node_id, kind=EdgeKind.CONTRADICTION))
        return max((int(edge.weight_bp) for edge in edges), default=0)

    def _candidate_score_bp(self, sim_phase: int, temporal_relevance: int, causal_support: int, contradiction_penalty: int) -> int:
        weights = self.scoring_weights
        numerator = (
            int(weights.phase_weight) * int(sim_phase)
            + int(weights.temporal_weight) * int(temporal_relevance)
            + int(weights.causal_weight) * int(causal_support)
            - int(weights.contradiction_weight) * int(contradiction_penalty)
        )
        return max(0, min(10000, numerator // weights.denominator))

    def _graph_score_bp(self, node_id: str) -> int:
        # Backward-compatible connectivity score retained for external callers.
        degree = len(self.graph.incoming(node_id)) + len(self.graph.outgoing(node_id))
        return min(10000, degree * 1500)


__all__ = [
    "AnalogyTransferResult",
    "AssociationScoringWeights",
    "AssociativeCortex",
    "CandidateHypothesis",
    "CandidateStatus",
    "CandidateTheory",
    "CognitiveEvaluationReport",
    "CounterfactualQuery",
    "CortexError",
    "EdgeKind",
    "LongMemoryAssociationGraph",
    "LongMemoryGraph",
    "MemoryEdge",
    "MemoryNode",
]
