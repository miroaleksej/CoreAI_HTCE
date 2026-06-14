"""proof layer proof calculus for HTCE-Origin Clean Body.

The proof layer is deliberately bounded.  It creates typed proof objects and
quarantine decisions, but it does not mutate L1/L2/L3 memory and it does not
commit answers.  Downstream policy/evidence gates decide whether a proven
candidate may be surfaced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ProofError(ValueError):
    """Raised when proof-layer input is malformed."""


class RuleKind(str, Enum):
    ASSERTED = "ASSERTED"
    ASSOCIATION = "ASSOCIATION"
    AND = "AND"
    OR = "OR"
    IMPLIES = "IMPLIES"
    NOT = "NOT"
    CONTRADICTION = "CONTRADICTION"
    TRANSITIVE_LOCATION = "TRANSITIVE_LOCATION"
    CLASS_INHERITANCE = "CLASS_INHERITANCE"
    ENSURES = "ENSURES"
    SKILL = "SKILL"
    LATEST_STATE = "LATEST_STATE"
    CLASS_RULE_DEDUCTION = "CLASS_RULE_DEDUCTION"
    SAME_CLASS_PROPERTY_INDUCTION = "SAME_CLASS_PROPERTY_INDUCTION"
    QUERY_STRATEGY = "QUERY_STRATEGY"
    PROOF_PATH_SCORING = "PROOF_PATH_SCORING"
    API_CALL_READY = "API_CALL_READY"


@dataclass(frozen=True, order=True)
class Statement:
    """A small typed logical atom.

    Examples:
    - ``Statement.atom("A")``
    - ``Statement.atom("located_in", "Mary", "office")``
    - ``Statement.atom("located_in", "Mary", "office").negate()``
    """

    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.predicate or not isinstance(self.predicate, str):
            raise ProofError("statement predicate must be a non-empty string")
        if any(not isinstance(arg, str) or not arg for arg in self.args):
            raise ProofError("statement args must be non-empty strings")
        object.__setattr__(self, "args", tuple(self.args))

    @classmethod
    def atom(cls, predicate: str, *args: str) -> "Statement":
        return cls(predicate=predicate, args=tuple(args), negated=False)

    @classmethod
    def from_text(cls, text: str) -> "Statement":
        """Parse a tiny human-readable atom form.

        Supported forms:
        - ``A``
        - ``NOT A``
        - ``located_in(Mary,office)``
        - ``not located_in(Mary,office)``
        """
        raw = text.strip()
        if not raw:
            raise ProofError("statement text is empty")
        negated = False
        lowered = raw.lower()
        if lowered.startswith("not "):
            negated = True
            raw = raw[4:].strip()
        elif raw.startswith("¬"):
            negated = True
            raw = raw[1:].strip()
        if "(" in raw:
            if not raw.endswith(")"):
                raise ProofError(f"invalid statement syntax: {text!r}")
            predicate, arg_blob = raw[:-1].split("(", 1)
            args = tuple(arg.strip() for arg in arg_blob.split(",") if arg.strip())
            return cls(predicate.strip(), args, negated)
        return cls(raw, (), negated)

    def negate(self) -> "Statement":
        return Statement(self.predicate, self.args, not self.negated)

    def canonical(self) -> str:
        args = ",".join(self.args)
        atom = f"{self.predicate}({args})" if self.args else self.predicate
        return f"NOT {atom}" if self.negated else atom

    def __str__(self) -> str:
        return self.canonical()


def normalize_statement(value: Statement | str) -> Statement:
    if isinstance(value, Statement):
        return value
    if isinstance(value, str):
        return Statement.from_text(value)
    raise ProofError("expected Statement or string")


@dataclass(frozen=True)
class Judgment:
    """A statement plus provenance classification.

    ``source='association'`` is explicitly non-authoritative.  It can seed a
    hypothesis but cannot prove an answer by itself.
    """

    statement: Statement | str
    evidence_id: str | None = None
    source: str = "asserted"
    supported: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "statement", normalize_statement(self.statement))
        if self.source not in {"asserted", "proof", "association", "skill", "ensures", "derived"}:
            raise ProofError(f"unknown judgment source: {self.source}")

    @property
    def authoritative(self) -> bool:
        return self.supported and self.source != "association"

    def canonical(self) -> str:
        return self.statement.canonical()


@dataclass(frozen=True)
class ImplicationRule:
    antecedent: Statement | str
    consequent: Statement | str
    rule_id: str = "implies"

    def __post_init__(self) -> None:
        object.__setattr__(self, "antecedent", normalize_statement(self.antecedent))
        object.__setattr__(self, "consequent", normalize_statement(self.consequent))
        if not self.rule_id:
            raise ProofError("implication rule_id must be non-empty")


@dataclass(frozen=True)
class EnsuresObligation:
    """A verified-skill obligation: procedure execution is valid only if
    the listed postcondition can be proven before the skill is exposed.
    """

    skill_name: str
    postcondition: Statement | str

    def __post_init__(self) -> None:
        if not self.skill_name:
            raise ProofError("skill_name must be non-empty")
        object.__setattr__(self, "postcondition", normalize_statement(self.postcondition))


@dataclass(frozen=True)
class ProofObject:
    conclusion: Judgment
    premises: tuple[Judgment, ...] = ()
    rules: tuple[RuleKind, ...] = ()
    valid: bool = False
    quarantined: bool = False
    reason: str = ""
    proof_id: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "premises", tuple(self.premises))
        object.__setattr__(self, "rules", tuple(self.rules))
        if not self.proof_id:
            digest = hashlib.sha256()
            digest.update(self.conclusion.canonical().encode("utf-8"))
            for premise in self.premises:
                digest.update(b"|")
                digest.update(premise.canonical().encode("utf-8"))
                digest.update(b":")
                digest.update(premise.source.encode("utf-8"))
            for rule in self.rules:
                digest.update(b"#")
                digest.update(rule.value.encode("utf-8"))
            digest.update(b"valid" if self.valid else b"invalid")
            digest.update(b"quarantine" if self.quarantined else b"open")
            object.__setattr__(self, "proof_id", digest.hexdigest())


@dataclass(frozen=True)
class QuarantineRecord:
    positive: Judgment
    negative: Judgment
    reason: str = "contradiction detected"

    def as_proof(self) -> ProofObject:
        conclusion = Judgment(self.positive.statement, source="proof", supported=False)
        return ProofObject(
            conclusion=conclusion,
            premises=(self.positive, self.negative),
            rules=(RuleKind.NOT, RuleKind.CONTRADICTION),
            valid=False,
            quarantined=True,
            reason=self.reason,
        )




@dataclass(frozen=True)
class AssociationReport:
    """Non-authoritative associative signal.

    The proof layer may receive associative candidates from cortex, but an
    association report is never a proof.  It can only authorize a hypothesis
    when the claim boundary allows hypothesis-grade output.
    """

    candidates: tuple[Judgment, ...] = ()
    report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.report_id:
            digest = hashlib.sha256()
            for candidate in self.candidates:
                digest.update(candidate.canonical().encode("utf-8"))
                digest.update(b"|")
                digest.update(candidate.source.encode("utf-8"))
            object.__setattr__(self, "report_id", digest.hexdigest())

    @property
    def has_candidates(self) -> bool:
        return bool(self.candidates)


@dataclass(frozen=True)
class QueryProofResult:
    """Authorization result for query-level proof strategies.

    ``answer_allowed`` is true only for valid, non-quarantined proofs.
    ``hypothesis_allowed`` is true only when association/induction provides a
    bounded hypothesis while proof remains invalid.
    """

    proof: ProofObject
    association_report: AssociationReport | None = None
    answer_allowed: bool = False
    hypothesis_allowed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.answer_allowed and self.hypothesis_allowed:
            raise ProofError("query proof result cannot be both answer and hypothesis")


@dataclass(frozen=True)
class ProofPathScore:
    """Diagnostic score for a proof path.

    This score is not a proof rule and does not authorize an answer by itself.
    It supports reflective comparison of candidate theories:

    TheoryScore(T) = (4*proof_score + 4*evidence_score + simplicity + falsifiability) // 10
    """

    proof_id: str
    proof_score_bp: int
    evidence_score_bp: int
    simplicity_bp: int
    falsifiability_bp: int
    theory_score_bp: int
    answer_allowed: bool = False
    reason: str = "proof path score is diagnostic only"

    def __post_init__(self) -> None:
        if not self.proof_id:
            raise ProofError("proof_id must be non-empty")
        for field_name in ("proof_score_bp", "evidence_score_bp", "simplicity_bp", "falsifiability_bp", "theory_score_bp"):
            value = int(getattr(self, field_name))
            if not 0 <= value <= 10000:
                raise ProofError(f"{field_name} must be in [0, 10000]")
            object.__setattr__(self, field_name, value)


class ContradictionDetector:
    """Detect direct A / NOT A conflicts among authoritative judgments."""

    def find(self, judgments: Iterable[Judgment]) -> tuple[QuarantineRecord, ...]:
        records: list[QuarantineRecord] = []
        by_statement: dict[Statement, Judgment] = {}
        for judgment in judgments:
            if not judgment.authoritative:
                continue
            opposite = judgment.statement.negate()
            if opposite in by_statement:
                earlier = by_statement[opposite]
                if judgment.statement.negated:
                    records.append(QuarantineRecord(positive=earlier, negative=judgment))
                else:
                    records.append(QuarantineRecord(positive=judgment, negative=earlier))
            by_statement[judgment.statement] = judgment
        return tuple(records)


class TheoremLayer:
    """Bounded proof calculus for proof layer.

    Supported inference families:
    - asserted authoritative fact proves itself;
    - A and A -> B proves B;
    - A and NOT A quarantines A;
    - located_in(x,y) + located_in(y,z) / inside(y,z) proves located_in(x,z);
    - is_a(x,c) + has_property(c,p) proves has_property(x,p);
    - skill execution requires ENSURES postcondition proof;
    - where(x) is proven from a latest-state index, not from association;
    - class-rule deduction proves predicates such as afraid_of(x,z) from
      is_a(x,C) and afraid_of(C,z);
    - same-class property transfer can create a hypothesis, never an answer.
    """

    def __init__(self) -> None:
        self._judgments: list[Judgment] = []
        self._implications: list[ImplicationRule] = []
        self._ensures: dict[str, EnsuresObligation] = {}

    @property
    def judgments(self) -> tuple[Judgment, ...]:
        return tuple(self._judgments)

    @property
    def implications(self) -> tuple[ImplicationRule, ...]:
        return tuple(self._implications)

    def add_judgment(self, judgment: Judgment | Statement | str, *, evidence_id: str | None = None, source: str = "asserted", supported: bool = True) -> Judgment:
        if isinstance(judgment, Judgment):
            item = judgment
        else:
            item = Judgment(judgment, evidence_id=evidence_id, source=source, supported=supported)
        self._judgments.append(item)
        return item

    def add_association(self, statement: Statement | str) -> Judgment:
        return self.add_judgment(statement, source="association", supported=True)

    def add_implication(self, antecedent: Statement | str, consequent: Statement | str, *, rule_id: str = "implies") -> ImplicationRule:
        rule = ImplicationRule(antecedent, consequent, rule_id=rule_id)
        self._implications.append(rule)
        return rule

    def add_ensures(self, skill_name: str, postcondition: Statement | str) -> EnsuresObligation:
        obligation = EnsuresObligation(skill_name, postcondition)
        self._ensures[skill_name] = obligation
        return obligation


    def export_state(self) -> dict[str, object]:
        """Export bounded proof-layer state for runtime snapshot/restore."""

        return {
            "judgments": [
                {
                    "evidence_id": judgment.evidence_id,
                    "source": judgment.source,
                    "statement": judgment.statement.canonical(),
                    "supported": judgment.supported,
                }
                for judgment in self._judgments
            ],
            "implications": [
                {
                    "antecedent": rule.antecedent.canonical(),
                    "consequent": rule.consequent.canonical(),
                    "rule_id": rule.rule_id,
                }
                for rule in self._implications
            ],
            "ensures": [
                {
                    "postcondition": obligation.postcondition.canonical(),
                    "skill_name": obligation.skill_name,
                }
                for obligation in self._ensures.values()
            ],
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, object]) -> "TheoremLayer":
        """Restore proof-layer judgments/rules from a runtime snapshot."""

        layer = cls()
        for raw in payload.get("judgments", ()):  # type: ignore[union-attr]
            if not isinstance(raw, Mapping):
                raise ProofError("proof snapshot judgment must be a mapping")
            layer.add_judgment(
                str(raw.get("statement", "")),
                evidence_id=raw.get("evidence_id") if raw.get("evidence_id") is None else str(raw.get("evidence_id")),
                source=str(raw.get("source", "asserted")),
                supported=bool(raw.get("supported", True)),
            )
        for raw in payload.get("implications", ()):  # type: ignore[union-attr]
            if not isinstance(raw, Mapping):
                raise ProofError("proof snapshot implication must be a mapping")
            layer.add_implication(
                str(raw.get("antecedent", "")),
                str(raw.get("consequent", "")),
                rule_id=str(raw.get("rule_id", "implies")),
            )
        for raw in payload.get("ensures", ()):  # type: ignore[union-attr]
            if not isinstance(raw, Mapping):
                raise ProofError("proof snapshot ensures must be a mapping")
            layer.add_ensures(str(raw.get("skill_name", "")), str(raw.get("postcondition", "")))
        return layer

    def contradiction_records(self) -> tuple[QuarantineRecord, ...]:
        return ContradictionDetector().find(self._judgments)

    def quarantine_if_contradicted(self, statement: Statement | str) -> ProofObject | None:
        target = normalize_statement(statement)
        for record in self.contradiction_records():
            if record.positive.statement == target or record.negative.statement == target:
                return record.as_proof()
        return None

    def _authoritative_matches(self, statement: Statement) -> tuple[Judgment, ...]:
        return tuple(j for j in self._judgments if j.statement == statement and j.authoritative)

    def _association_matches(self, statement: Statement) -> tuple[Judgment, ...]:
        return tuple(j for j in self._judgments if j.statement == statement and j.source == "association")

    def prove(self, goal: Judgment | Statement | str) -> ProofObject:
        goal_judgment = goal if isinstance(goal, Judgment) else Judgment(goal, source="proof")
        goal_statement = goal_judgment.statement

        quarantine = self.quarantine_if_contradicted(goal_statement)
        if quarantine is not None:
            return quarantine

        direct = self._authoritative_matches(goal_statement)
        if direct:
            return ProofObject(
                conclusion=Judgment(goal_statement, source="proof", supported=True),
                premises=(direct[0],),
                rules=(RuleKind.ASSERTED,),
                valid=True,
                reason="authoritative fact proves itself",
            )

        associations = self._association_matches(goal_statement)
        if associations:
            return ProofObject(
                conclusion=Judgment(goal_statement, source="proof", supported=False),
                premises=associations,
                rules=(RuleKind.ASSOCIATION,),
                valid=False,
                reason="association alone cannot prove an answer",
            )

        implied = self._prove_by_implication(goal_statement)
        if implied is not None:
            return implied

        transitive = self._prove_transitive_location(goal_statement)
        if transitive is not None:
            return transitive

        inherited = self._prove_class_inheritance(goal_statement)
        if inherited is not None:
            return inherited

        class_rule = self._prove_class_rule_deduction(goal_statement)
        if class_rule is not None:
            return class_rule

        return ProofObject(
            conclusion=Judgment(goal_statement, source="proof", supported=False),
            premises=(),
            rules=(),
            valid=False,
            reason="no proof found",
        )

    def _prove_by_implication(self, goal: Statement) -> ProofObject | None:
        for rule in self._implications:
            if rule.consequent != goal:
                continue
            antecedents = self._authoritative_matches(rule.antecedent)
            if antecedents:
                return ProofObject(
                    conclusion=Judgment(goal, source="proof", supported=True),
                    premises=(antecedents[0],),
                    rules=(RuleKind.IMPLIES,),
                    valid=True,
                    reason=f"{rule.antecedent.canonical()} implies {goal.canonical()}",
                )
        return None

    def _prove_transitive_location(self, goal: Statement) -> ProofObject | None:
        if goal.negated or goal.predicate != "located_in" or len(goal.args) != 2:
            return None
        subject, destination = goal.args
        authoritative = [j for j in self._judgments if j.authoritative and not j.statement.negated]
        # P22: object-location chaining is proof-only.  A carried object is not
        # directly re-written as a location fact by the NLU bridge; instead the
        # theorem layer proves located_in(object, place) from
        # carried_by(object, actor) + located_in(actor, place).  Existing
        # located_in/inside chains remain supported.
        first_hops = [
            j for j in authoritative
            if j.statement.predicate in {"located_in", "carried_by", "held_by"}
            and len(j.statement.args) == 2
            and j.statement.args[0] == subject
        ]
        for first in first_hops:
            middle = first.statement.args[1]
            for second in authoritative:
                st = second.statement
                if len(st.args) != 2:
                    continue
                if st.predicate in {"located_in", "inside"} and st.args == (middle, destination):
                    return ProofObject(
                        conclusion=Judgment(goal, source="proof", supported=True),
                        premises=(first, second),
                        rules=(RuleKind.TRANSITIVE_LOCATION,),
                        valid=True,
                        reason="transitive/carried location proof",
                    )
        return None

    def _prove_class_inheritance(self, goal: Statement) -> ProofObject | None:
        if goal.negated or goal.predicate != "has_property" or len(goal.args) != 2:
            return None
        entity, property_name = goal.args
        authoritative = [j for j in self._judgments if j.authoritative and not j.statement.negated]
        class_facts = [j for j in authoritative if j.statement.predicate == "is_a" and len(j.statement.args) == 2 and j.statement.args[0] == entity]
        for class_fact in class_facts:
            class_name = class_fact.statement.args[1]
            for property_fact in authoritative:
                st = property_fact.statement
                if st.predicate == "has_property" and len(st.args) == 2 and st.args == (class_name, property_name):
                    return ProofObject(
                        conclusion=Judgment(goal, source="proof", supported=True),
                        premises=(class_fact, property_fact),
                        rules=(RuleKind.CLASS_INHERITANCE,),
                        valid=True,
                        reason="class inheritance proof",
                    )
        return None

    def prove_where(self, entity: str, latest_state_index: Mapping[str, str] | Mapping[tuple[str, str], str], *, evidence_id: str = "latest_state_index") -> ProofObject:
        """Prove ``located_in(entity, y)`` from a latest-state index.

        The index is accepted in either of two bounded forms:
        - ``{"mary": "office"}``
        - ``{("mary", "located_in"): "office"}``

        The method creates a proof object only.  It does not mutate memory and
        it does not commit the fact to L2.
        """
        if not entity or not isinstance(entity, str):
            raise ProofError("entity must be a non-empty string")
        normalized_entity = entity.strip().lower()
        location: str | None = None
        for key, value in latest_state_index.items():
            if isinstance(key, tuple):
                if len(key) != 2:
                    continue
                subject, relation = str(key[0]).lower(), str(key[1]).lower()
                if subject == normalized_entity and relation in {"located_in", "location"}:
                    location = str(value).strip().lower()
                    break
            elif str(key).lower() == normalized_entity:
                location = str(value).strip().lower()
                break
        if not location:
            return ProofObject(
                conclusion=Judgment(Statement.atom("located_in", normalized_entity, "unknown"), source="proof", supported=False),
                premises=(),
                rules=(RuleKind.LATEST_STATE, RuleKind.QUERY_STRATEGY),
                valid=False,
                reason="no latest-state fact available for where-query",
            )
        conclusion = Statement.atom("located_in", normalized_entity, location)
        premise = Judgment(conclusion, evidence_id=evidence_id, source="derived", supported=True)
        return ProofObject(
            conclusion=Judgment(conclusion, source="proof", supported=True),
            premises=(premise,),
            rules=(RuleKind.LATEST_STATE, RuleKind.QUERY_STRATEGY),
            valid=True,
            reason="where-query proven from latest-state index",
        )

    def _prove_class_rule_deduction(self, goal: Statement) -> ProofObject | None:
        """Generic class-rule deduction.

        Example:
        ``is_a(sparrow,bird)`` and ``afraid_of(bird,cat)`` prove
        ``afraid_of(sparrow,cat)``.  The rule is intentionally bounded to
        binary predicates and does not handle negation or probabilistic lift.
        """
        if goal.negated or len(goal.args) != 2 or goal.predicate in {"is_a", "located_in", "inside", "has_property"}:
            return None
        entity, target = goal.args
        authoritative = [j for j in self._judgments if j.authoritative and not j.statement.negated]
        class_facts = [j for j in authoritative if j.statement.predicate == "is_a" and len(j.statement.args) == 2 and j.statement.args[0] == entity]
        for class_fact in class_facts:
            class_name = class_fact.statement.args[1]
            for rule_fact in authoritative:
                st = rule_fact.statement
                if st.predicate == goal.predicate and len(st.args) == 2 and st.args == (class_name, target):
                    return ProofObject(
                        conclusion=Judgment(goal, source="proof", supported=True),
                        premises=(class_fact, rule_fact),
                        rules=(RuleKind.CLASS_RULE_DEDUCTION,),
                        valid=True,
                        reason="class-rule deduction proof",
                    )
        return None

    def infer_same_class_property_hypothesis(self, goal: Statement | str, *, min_examples: int = 1) -> QueryProofResult:
        """Infer a same-class property transfer only as a hypothesis.

        Example: if two known cats have ``color_of(..., black)`` and the goal
        entity is also a cat, the result is a hypothesis, not an answer.  This
        preserves the invariant ``association_report != proof``.
        """
        goal_statement = normalize_statement(goal)
        if goal_statement.negated or len(goal_statement.args) != 2:
            proof = ProofObject(
                conclusion=Judgment(goal_statement, source="proof", supported=False),
                premises=(),
                rules=(RuleKind.SAME_CLASS_PROPERTY_INDUCTION,),
                valid=False,
                reason="same-class induction requires a positive binary predicate",
            )
            return QueryProofResult(proof=proof, answer_allowed=False, hypothesis_allowed=False, reason=proof.reason)
        entity, target_value = goal_statement.args
        authoritative = [j for j in self._judgments if j.authoritative and not j.statement.negated]
        entity_classes = [j.statement.args[1] for j in authoritative if j.statement.predicate == "is_a" and len(j.statement.args) == 2 and j.statement.args[0] == entity]
        candidate_premises: list[Judgment] = []
        for class_name in entity_classes:
            siblings = {j.statement.args[0] for j in authoritative if j.statement.predicate == "is_a" and len(j.statement.args) == 2 and j.statement.args[1] == class_name and j.statement.args[0] != entity}
            for property_fact in authoritative:
                st = property_fact.statement
                if st.predicate == goal_statement.predicate and len(st.args) == 2 and st.args[1] == target_value and st.args[0] in siblings:
                    candidate_premises.append(property_fact)
        # Deterministic de-duplication while preserving order.
        deduped: list[Judgment] = []
        seen: set[str] = set()
        for premise in candidate_premises:
            key = premise.canonical() + "|" + str(premise.evidence_id)
            if key not in seen:
                seen.add(key)
                deduped.append(premise)
        if len(deduped) >= int(min_examples):
            report = AssociationReport(tuple(Judgment(goal_statement, source="association", supported=True) for _ in (0,)))
            proof = ProofObject(
                conclusion=Judgment(goal_statement, source="proof", supported=False),
                premises=tuple(deduped[: int(min_examples)]),
                rules=(RuleKind.SAME_CLASS_PROPERTY_INDUCTION,),
                valid=False,
                reason="same-class property transfer is hypothesis-grade only",
            )
            return QueryProofResult(
                proof=proof,
                association_report=report,
                answer_allowed=False,
                hypothesis_allowed=True,
                reason="hypothesis allowed because induction is not proof",
            )
        proof = ProofObject(
            conclusion=Judgment(goal_statement, source="proof", supported=False),
            premises=tuple(deduped),
            rules=(RuleKind.SAME_CLASS_PROPERTY_INDUCTION,),
            valid=False,
            reason="not enough same-class examples for hypothesis",
        )
        return QueryProofResult(proof=proof, answer_allowed=False, hypothesis_allowed=False, reason=proof.reason)


    def prove_api_call_ready(
        self,
        required_slots: Mapping[str, str],
        latest_state_index: Mapping[str, str] | Mapping[tuple[str, str], str],
        *,
        context_subject: str = "current_dialog",
        quarantined_slots: Sequence[str] = (),
    ) -> ProofObject:
        """Prove that a simulated API call has all active dialog slots.

        P23 reuses L2 latest-state facts as slot memory.  A filled slot is
        represented as ``(current_dialog, has_slot_value_<slot>) -> value``.
        This method does not read gold answers and does not mutate memory; the
        runtime passes a latest-state snapshot and any quarantined slot names.
        """

        normalized_required = {str(slot).strip().lower(): str(expected).strip().lower() for slot, expected in required_slots.items()}
        missing: list[str] = []
        mismatched: list[str] = []
        quarantined = {str(slot).strip().lower() for slot in quarantined_slots}
        premises: list[Judgment] = []

        def lookup(slot: str) -> str | None:
            relation = f"has_slot_value_{slot}"
            tuple_key = (context_subject.lower(), relation.lower())
            if tuple_key in latest_state_index:  # type: ignore[operator]
                return str(latest_state_index[tuple_key]).lower()  # type: ignore[index]
            text_key = f"{tuple_key[0]}::{tuple_key[1]}"
            if text_key in latest_state_index:  # type: ignore[operator]
                return str(latest_state_index[text_key]).lower()  # type: ignore[index]
            return None

        for slot, expected in normalized_required.items():
            if slot in quarantined:
                continue
            value = lookup(slot)
            if value is None:
                missing.append(slot)
                continue
            if expected != "*" and value != expected:
                mismatched.append(f"{slot} expected {expected} got {value}")
                continue
            premises.append(Judgment(Statement.atom("slot_filled", context_subject, slot, value), source="proof", supported=True))

        conclusion = Judgment(Statement.atom("api_call_ready", context_subject, "_".join(sorted(normalized_required))), source="proof", supported=not missing and not mismatched and not quarantined)
        if quarantined:
            return ProofObject(
                conclusion=conclusion,
                premises=tuple(premises),
                rules=(RuleKind.API_CALL_READY,),
                valid=False,
                quarantined=True,
                reason="dialog slot is quarantined by contradiction: " + ", ".join(sorted(quarantined)),
            )
        if missing:
            return ProofObject(
                conclusion=conclusion,
                premises=tuple(premises),
                rules=(RuleKind.API_CALL_READY,),
                valid=False,
                reason="missing required dialog slots: " + ", ".join(sorted(missing)),
            )
        if mismatched:
            return ProofObject(
                conclusion=conclusion,
                premises=tuple(premises),
                rules=(RuleKind.API_CALL_READY,),
                valid=False,
                reason="dialog slot mismatch: " + "; ".join(sorted(mismatched)),
            )
        return ProofObject(
            conclusion=conclusion,
            premises=tuple(premises),
            rules=(RuleKind.API_CALL_READY,),
            valid=True,
            reason="all required dialog slots are active and non-quarantined",
        )

    def authorize_query(
        self,
        goal: Statement | str | ProofObject,
        *,
        association_report: AssociationReport | None = None,
        association_candidates: Iterable[Judgment | Statement | str] = (),
        allow_hypothesis: bool = True,
    ) -> QueryProofResult:
        """Authorize answer vs hypothesis for a query.

        Formula implemented:
        ``AnswerAllowed(q) = Proof(q).valid``
        ``HypothesisAllowed(q) = Assoc(q) != empty and Proof(q).valid = 0``
        when ``allow_hypothesis`` is true.

        ``goal`` may be either a statement to prove internally or a precomputed
        ProofObject from a query strategy such as ``prove_where``. This lets the
        runtime prove latest-state answers using RuleKind.LATEST_STATE rather
        than falling back to association or generic direct proof.
        """
        if isinstance(goal, ProofObject):
            proof = goal
            goal_statement = proof.conclusion.statement
        else:
            goal_statement = normalize_statement(goal)
            proof = self.prove(goal_statement)
        if proof.valid and not proof.quarantined:
            return QueryProofResult(proof=proof, association_report=association_report, answer_allowed=True, hypothesis_allowed=False, reason="valid proof authorizes answer")
        candidates: list[Judgment] = []
        if association_report is not None:
            candidates.extend(association_report.candidates)
        for candidate in association_candidates:
            if isinstance(candidate, Judgment):
                item = candidate
            else:
                item = Judgment(candidate, source="association", supported=True)
            if item.statement == goal_statement:
                candidates.append(item)
        if not candidates:
            candidates.extend(self._association_matches(goal_statement))
        report = AssociationReport(tuple(candidates)) if candidates else association_report
        hypothesis_allowed = bool(allow_hypothesis and report and report.has_candidates and not proof.valid and not proof.quarantined)
        return QueryProofResult(
            proof=proof,
            association_report=report,
            answer_allowed=False,
            hypothesis_allowed=hypothesis_allowed,
            reason="association can authorize only hypothesis" if hypothesis_allowed else proof.reason,
        )

    def score_proof_path(
        self,
        proof: ProofObject,
        *,
        evidence_score_bp: int | None = None,
        falsifiability_bp: int = 0,
    ) -> ProofPathScore:
        """Return a bounded diagnostic score for a proof path.

        Valid proof contributes proof_score=10000; invalid/quarantined proof
        contributes proof_score=0.  Evidence score defaults to the fraction of
        premises with evidence identifiers.  Simplicity decreases with proof
        path length.
        """

        if evidence_score_bp is None:
            if proof.premises:
                with_evidence = sum(1 for premise in proof.premises if premise.evidence_id)
                evidence_score_bp = with_evidence * 10000 // len(proof.premises)
            else:
                evidence_score_bp = 0
        evidence_score_bp = max(0, min(10000, int(evidence_score_bp)))
        falsifiability_bp = max(0, min(10000, int(falsifiability_bp)))
        proof_score_bp = 10000 if proof.valid and not proof.quarantined else 0
        path_length = len(proof.premises) + len(proof.rules)
        simplicity_bp = max(0, 10000 - max(0, path_length - 1) * 1000)
        if proof.quarantined:
            simplicity_bp = 0
        theory_score_bp = (4 * proof_score_bp + 4 * evidence_score_bp + simplicity_bp + falsifiability_bp) // 10
        return ProofPathScore(
            proof_id=proof.proof_id,
            proof_score_bp=proof_score_bp,
            evidence_score_bp=evidence_score_bp,
            simplicity_bp=simplicity_bp,
            falsifiability_bp=falsifiability_bp,
            theory_score_bp=max(0, min(10000, theory_score_bp)),
            answer_allowed=proof.valid and not proof.quarantined,
            reason="valid proof path scored" if proof.valid and not proof.quarantined else proof.reason or "invalid proof path scored",
        )

    def verify_skill(self, skill_name: str) -> ProofObject:
        obligation = self._ensures.get(skill_name)
        if obligation is None:
            return ProofObject(
                conclusion=Judgment(Statement.atom("skill_verified", skill_name), source="proof", supported=False),
                premises=(),
                rules=(RuleKind.SKILL, RuleKind.ENSURES),
                valid=False,
                reason="skill has no ENSURES obligation",
            )
        proof = self.prove(obligation.postcondition)
        if proof.valid and not proof.quarantined:
            return ProofObject(
                conclusion=Judgment(Statement.atom("skill_verified", skill_name), source="proof", supported=True),
                premises=proof.premises + (Judgment(obligation.postcondition, source="ensures", supported=True),),
                rules=proof.rules + (RuleKind.ENSURES, RuleKind.SKILL),
                valid=True,
                reason="skill ENSURES obligation proven",
            )
        return ProofObject(
            conclusion=Judgment(Statement.atom("skill_verified", skill_name), source="proof", supported=False),
            premises=proof.premises,
            rules=proof.rules + (RuleKind.ENSURES, RuleKind.SKILL),
            valid=False,
            quarantined=proof.quarantined,
            reason="skill ENSURES obligation is not proven",
        )


__all__ = [
    "AssociationReport",
    "ContradictionDetector",
    "EnsuresObligation",
    "ImplicationRule",
    "Judgment",
    "ProofError",
    "ProofObject",
    "ProofPathScore",
    "QueryProofResult",
    "QuarantineRecord",
    "RuleKind",
    "Statement",
    "TheoremLayer",
    "normalize_statement",
]
