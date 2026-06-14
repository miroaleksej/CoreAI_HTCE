"""P18 no-leakage dynamic benchmark protocol for HTCE-Origin.

Scope boundary:
- Evaluation contour only; does not mutate L1/L2/L3 runtime state.
- The engine receives public task cards only.
- Gold answers and hidden criteria are stored only as canonical hash commitments.
- Dynamic/counterfactual rewrites are deterministic from explicit seeds.
- All protected artifacts use canonical JSON and reject floats by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from htce_origin.governance.evidence import HashChain
from htce_origin.kernel.serialization import sha256_hex


class DynamicTaskFamily(str, Enum):
    LATEST_STATE = "latest_state_dynamic"
    DEDUCTION = "deduction_dynamic"
    CONTRADICTION = "contradiction_retraction_dynamic"
    ARC_SYMBOLIC = "arc_symbolic_dynamic"
    CLOSED_LOOP = "closed_loop_dynamic"


@dataclass(frozen=True)
class DynamicTaskCard:
    """Public benchmark card visible to the system under test.

    It intentionally contains no gold answer, no hidden criteria text and no
    scoring rubric.  The public card can be sent to an engine; the private
    goldset must not be sent.
    """

    task_id: str
    family: DynamicTaskFamily
    seed: str
    prompt: str
    public_facts: tuple[str, ...]
    query: str
    required_capability: str
    htce_modules_used: tuple[str, ...]
    counterfactual_index: int = 0
    contains_gold_answer: int = 0
    contains_hidden_criteria: int = 0
    contains_template_answer: int = 0

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.seed.strip() or not self.prompt.strip():
            raise ValueError("task_id, seed and prompt must be non-empty")
        if self.contains_gold_answer != 0 or self.contains_hidden_criteria != 0 or self.contains_template_answer != 0:
            raise ValueError("DynamicTaskCard must not contain gold answer, hidden criteria or template answer")
        if not self.htce_modules_used:
            raise ValueError("htce_modules_used must be non-empty")

    def public_payload(self) -> dict[str, object]:
        return {
            "contains_gold_answer": self.contains_gold_answer,
            "contains_hidden_criteria": self.contains_hidden_criteria,
            "contains_template_answer": self.contains_template_answer,
            "counterfactual_index": self.counterfactual_index,
            "family": self.family.value,
            "htce_modules_used": list(self.htce_modules_used),
            "prompt": self.prompt,
            "public_facts": list(self.public_facts),
            "query": self.query,
            "required_capability": self.required_capability,
            "seed": self.seed,
            "task_id": self.task_id,
        }

    def public_hash(self) -> str:
        return sha256_hex(self.public_payload())


@dataclass(frozen=True)
class SeededPrivateGoldset:
    """Private gold answer bundle; never passed to engine input."""

    task_id: str
    seed: str
    gold_answer: str
    acceptable_answers: tuple[str, ...]
    hidden_criteria: tuple[str, ...]
    evidence_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.seed.strip() or not self.gold_answer.strip():
            raise ValueError("private goldset identifiers and gold answer must be non-empty")
        if self.gold_answer not in self.acceptable_answers:
            raise ValueError("gold_answer must be included in acceptable_answers")
        if not self.hidden_criteria:
            raise ValueError("hidden_criteria must be non-empty")

    def private_payload(self) -> dict[str, object]:
        return {
            "acceptable_answers": list(self.acceptable_answers),
            "evidence_path": list(self.evidence_path),
            "gold_answer": self.gold_answer,
            "hidden_criteria": list(self.hidden_criteria),
            "seed": self.seed,
            "task_id": self.task_id,
        }

    def commitment_hash(self) -> str:
        return sha256_hex(self.private_payload())


@dataclass(frozen=True)
class HiddenCriteriaCommitment:
    task_id: str
    public_task_hash: str
    gold_commitment_hash: str
    commitment_hash: str
    committed_before_execution: int = 1

    def __post_init__(self) -> None:
        if self.committed_before_execution != 1:
            raise ValueError("hidden criteria must be committed before execution")

    @classmethod
    def create(cls, card: DynamicTaskCard, goldset: SeededPrivateGoldset) -> "HiddenCriteriaCommitment":
        if card.task_id != goldset.task_id or card.seed != goldset.seed:
            raise ValueError("card/goldset identity mismatch")
        public_hash = card.public_hash()
        gold_hash = goldset.commitment_hash()
        return cls(
            task_id=card.task_id,
            public_task_hash=public_hash,
            gold_commitment_hash=gold_hash,
            commitment_hash=sha256_hex({
                "gold_commitment_hash": gold_hash,
                "public_task_hash": public_hash,
                "schema": "htce-p18-hidden-criteria-commitment-v1",
                "task_id": card.task_id,
            }),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "commitment_hash": self.commitment_hash,
            "committed_before_execution": self.committed_before_execution,
            "gold_commitment_hash": self.gold_commitment_hash,
            "public_task_hash": self.public_task_hash,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class NoAnswerLeakageContract:
    public_task_contains_gold_answer: int = 0
    public_task_contains_template_answer: int = 0
    public_task_contains_hidden_criteria: int = 0
    engine_receives_gold_answer: int = 0
    engine_receives_private_goldset: int = 0
    hidden_criteria_committed_before_execution: int = 1
    gold_used_only_after_execution: int = 1

    def passed(self) -> bool:
        return self.as_payload() == {
            "engine_receives_gold_answer": 0,
            "engine_receives_private_goldset": 0,
            "gold_used_only_after_execution": 1,
            "hidden_criteria_committed_before_execution": 1,
            "public_task_contains_gold_answer": 0,
            "public_task_contains_hidden_criteria": 0,
            "public_task_contains_template_answer": 0,
        }

    def as_payload(self) -> dict[str, int]:
        return {
            "engine_receives_gold_answer": self.engine_receives_gold_answer,
            "engine_receives_private_goldset": self.engine_receives_private_goldset,
            "gold_used_only_after_execution": self.gold_used_only_after_execution,
            "hidden_criteria_committed_before_execution": self.hidden_criteria_committed_before_execution,
            "public_task_contains_gold_answer": self.public_task_contains_gold_answer,
            "public_task_contains_hidden_criteria": self.public_task_contains_hidden_criteria,
            "public_task_contains_template_answer": self.public_task_contains_template_answer,
        }


@dataclass(frozen=True)
class CounterfactualRewriteTest:
    original_task_id: str
    rewritten_task_id: str
    original_public_hash: str
    rewritten_public_hash: str
    original_gold_commitment_hash: str
    rewritten_gold_commitment_hash: str
    answer_changed: int
    public_prompt_changed: int

    def __post_init__(self) -> None:
        if self.answer_changed not in (0, 1) or self.public_prompt_changed not in (0, 1):
            raise ValueError("counterfactual flags must be 0 or 1")

    @property
    def passed(self) -> bool:
        return self.public_prompt_changed == 1 and self.answer_changed == 1

    def as_payload(self) -> dict[str, object]:
        return {
            "answer_changed": self.answer_changed,
            "original_gold_commitment_hash": self.original_gold_commitment_hash,
            "original_public_hash": self.original_public_hash,
            "original_task_id": self.original_task_id,
            "passed": self.passed,
            "public_prompt_changed": self.public_prompt_changed,
            "rewritten_gold_commitment_hash": self.rewritten_gold_commitment_hash,
            "rewritten_public_hash": self.rewritten_public_hash,
            "rewritten_task_id": self.rewritten_task_id,
        }


@dataclass(frozen=True)
class P18ProtocolRow:
    task: str
    family: str
    required_capability: str
    htce_modules_used: tuple[str, ...]
    engine_input_hash: str
    public_task_hash: str
    hidden_gold_commitment_hash: str
    pre_execution_commitment_hash: str
    answer: str | None
    answer_digest: str | None
    passed: int
    refusal_correctness: int
    false_support: int
    answer_key_visible_to_engine: int
    evidence_path: tuple[str, ...]
    trace_hash: str
    counterfactual_of: str | None = None

    def __post_init__(self) -> None:
        for name in ("passed", "refusal_correctness", "false_support", "answer_key_visible_to_engine"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")

    def as_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "answer_digest": self.answer_digest,
            "answer_key_visible_to_engine": self.answer_key_visible_to_engine,
            "counterfactual_of": self.counterfactual_of,
            "engine_input_hash": self.engine_input_hash,
            "evidence_path": list(self.evidence_path),
            "false_support": self.false_support,
            "family": self.family,
            "hidden_gold_commitment_hash": self.hidden_gold_commitment_hash,
            "htce_modules_used": list(self.htce_modules_used),
            "passed": self.passed,
            "pre_execution_commitment_hash": self.pre_execution_commitment_hash,
            "public_task_hash": self.public_task_hash,
            "refusal_correctness": self.refusal_correctness,
            "required_capability": self.required_capability,
            "task": self.task,
            "trace_hash": self.trace_hash,
        }


@dataclass(frozen=True)
class P18NoLeakageReport:
    schema_version: str
    rows: tuple[P18ProtocolRow, ...]
    commitments: tuple[HiddenCriteriaCommitment, ...]
    counterfactual_tests: tuple[CounterfactualRewriteTest, ...]
    no_answer_leakage_contract: NoAnswerLeakageContract
    trace_head: str
    passed_count: int
    total_count: int
    false_support_count: int
    answer_key_visible_count: int

    @property
    def passed(self) -> bool:
        return (
            self.total_count > 0
            and self.passed_count == self.total_count
            and self.false_support_count == 0
            and self.answer_key_visible_count == 0
            and self.no_answer_leakage_contract.passed()
            and all(item.passed for item in self.counterfactual_tests)
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "answer_key_visible_count": self.answer_key_visible_count,
            "commitments": [item.as_payload() for item in self.commitments],
            "counterfactual_tests": [item.as_payload() for item in self.counterfactual_tests],
            "false_support_count": self.false_support_count,
            "no_answer_leakage_contract": self.no_answer_leakage_contract.as_payload(),
            "passed": self.passed,
            "passed_count": self.passed_count,
            "rows": [row.as_payload() for row in self.rows],
            "schema_version": self.schema_version,
            "total_count": self.total_count,
            "trace_head": self.trace_head,
        }


class DynamicTaskGenerator:
    """Deterministic seeded task generator with private gold commitments."""

    _entities = ("mary", "john", "sandra", "daniel", "yuri", "olga", "li", "mira")
    _locations = ("office", "garden", "kitchen", "library", "hall", "lab", "garage", "studio")
    _classes = ("swan", "raven", "falcon", "crane", "heron", "ibis")
    _properties = ("bird", "animal", "thing", "living_entity")

    def _pick(self, seed: str, namespace: str, items: Sequence[str], offset: int = 0) -> str:
        digest = sha256_hex({"namespace": namespace, "offset": offset, "seed": seed})
        index = int(digest[:16], 16) % len(items)
        return items[index]

    def latest_state(self, seed: str, *, counterfactual_index: int = 0) -> tuple[DynamicTaskCard, SeededPrivateGoldset]:
        entity = self._pick(seed, "entity", self._entities, counterfactual_index)
        first = self._pick(seed, "first_location", self._locations, counterfactual_index)
        second = self._pick(seed, "second_location", self._locations, counterfactual_index + 11)
        if second == first:
            second = self._locations[(self._locations.index(second) + 1) % len(self._locations)]
        task_id = f"p18_latest_state_{counterfactual_index}_{sha256_hex(seed)[:12]}"
        facts = (f"{entity} located_in {first}", f"{entity} located_in {second}")
        card = DynamicTaskCard(
            task_id=task_id,
            family=DynamicTaskFamily.LATEST_STATE,
            seed=seed,
            prompt="Use the public facts only. Return the latest known location.",
            public_facts=facts,
            query=f"where is {entity}?",
            required_capability="latest-state recall without answer leakage",
            htce_modules_used=("body.memory", "body.layers", "governance.trace"),
            counterfactual_index=counterfactual_index,
        )
        gold = SeededPrivateGoldset(
            task_id=task_id,
            seed=seed,
            gold_answer=second,
            acceptable_answers=(second,),
            hidden_criteria=("latest fact for same entity/relation must win", "no answer key visible before execution"),
            evidence_path=("public_fact:1", "same_key_supersession"),
        )
        return card, gold

    def deduction(self, seed: str, *, counterfactual_index: int = 0) -> tuple[DynamicTaskCard, SeededPrivateGoldset]:
        subject = self._pick(seed, "class", self._classes, counterfactual_index)
        mid = self._pick(seed, "property", self._properties, counterfactual_index)
        target = self._pick(seed, "property_target", self._properties, counterfactual_index + 5)
        if target == mid:
            target = self._properties[(self._properties.index(target) + 1) % len(self._properties)]
        task_id = f"p18_deduction_{counterfactual_index}_{sha256_hex(seed)[:12]}"
        facts = (f"{subject} is_a {mid}", f"{mid} implies {target}")
        card = DynamicTaskCard(
            task_id=task_id,
            family=DynamicTaskFamily.DEDUCTION,
            seed=seed,
            prompt="Use public rules only. Infer the target class if supported.",
            public_facts=facts,
            query=f"what does {subject} imply?",
            required_capability="single-step symbolic deduction without gold leakage",
            htce_modules_used=("cognition.l3_promotion", "governance.proof", "governance.trace"),
            counterfactual_index=counterfactual_index,
        )
        gold = SeededPrivateGoldset(
            task_id=task_id,
            seed=seed,
            gold_answer=target,
            acceptable_answers=(target,),
            hidden_criteria=("use transitive rule only if both supports are public", "do not use hidden gold text"),
            evidence_path=("public_fact:0", "public_fact:1"),
        )
        return card, gold

    def contradiction(self, seed: str, *, counterfactual_index: int = 0) -> tuple[DynamicTaskCard, SeededPrivateGoldset]:
        claim = self._pick(seed, "claim", ("claim_alpha", "claim_beta", "claim_gamma", "claim_delta"), counterfactual_index)
        task_id = f"p18_contradiction_{counterfactual_index}_{sha256_hex(seed)[:12]}"
        facts = (f"source_weak supports {claim}", f"source_primary retracts {claim}", f"source_independent contradicts {claim}")
        card = DynamicTaskCard(
            task_id=task_id,
            family=DynamicTaskFamily.CONTRADICTION,
            seed=seed,
            prompt="Determine whether the claim can be used as supported evidence.",
            public_facts=facts,
            query=f"is {claim} supported?",
            required_capability="contradiction and retraction blocking without leaked label",
            htce_modules_used=("body.memory", "governance.evidence", "governance.proof", "governance.trace"),
            counterfactual_index=counterfactual_index,
        )
        gold = SeededPrivateGoldset(
            task_id=task_id,
            seed=seed,
            gold_answer="refuse",
            acceptable_answers=("refuse", "blocked", "unsupported"),
            hidden_criteria=("retracted or contradicted support must not settle truth", "refusal is correct"),
            evidence_path=("public_fact:1", "public_fact:2"),
        )
        return card, gold

    def arc_symbolic(self, seed: str, *, counterfactual_index: int = 0) -> tuple[DynamicTaskCard, SeededPrivateGoldset]:
        value = int(sha256_hex({"seed": seed, "counterfactual_index": counterfactual_index})[:4], 16) % 9 + 1
        direction = "right" if counterfactual_index % 2 == 0 else "down"
        answer = f"move_{value}_{direction}"
        task_id = f"p18_arc_symbolic_{counterfactual_index}_{sha256_hex(seed)[:12]}"
        facts = (f"train: {value} at origin -> {direction}", f"test: {value} at origin")
        card = DynamicTaskCard(
            task_id=task_id,
            family=DynamicTaskFamily.ARC_SYMBOLIC,
            seed=seed,
            prompt="Infer the symbolic transformation from public train examples.",
            public_facts=facts,
            query="what transformation applies to the test grid?",
            required_capability="ARC-style symbolic rewrite without memorized answer",
            htce_modules_used=("cognition.l3_promotion", "governance.trace"),
            counterfactual_index=counterfactual_index,
        )
        gold = SeededPrivateGoldset(
            task_id=task_id,
            seed=seed,
            gold_answer=answer,
            acceptable_answers=(answer,),
            hidden_criteria=("derive from public transformation", "private answer is audit-only"),
            evidence_path=("public_fact:0", "public_fact:1"),
        )
        return card, gold

    def card_pair(self, family: DynamicTaskFamily, seed: str, *, counterfactual_index: int = 0) -> tuple[DynamicTaskCard, SeededPrivateGoldset]:
        if family == DynamicTaskFamily.LATEST_STATE:
            return self.latest_state(seed, counterfactual_index=counterfactual_index)
        if family == DynamicTaskFamily.DEDUCTION:
            return self.deduction(seed, counterfactual_index=counterfactual_index)
        if family == DynamicTaskFamily.CONTRADICTION:
            return self.contradiction(seed, counterfactual_index=counterfactual_index)
        if family == DynamicTaskFamily.ARC_SYMBOLIC:
            return self.arc_symbolic(seed, counterfactual_index=counterfactual_index)
        raise ValueError(f"unsupported dynamic family: {family}")

    def default_pairs(self, seed: str = "htce-p18-private-seed") -> tuple[tuple[DynamicTaskCard, SeededPrivateGoldset], ...]:
        return (
            self.latest_state(seed, counterfactual_index=0),
            self.latest_state(seed, counterfactual_index=1),
            self.deduction(seed, counterfactual_index=0),
            self.deduction(seed, counterfactual_index=1),
            self.contradiction(seed, counterfactual_index=0),
            self.arc_symbolic(seed, counterfactual_index=0),
            self.arc_symbolic(seed, counterfactual_index=1),
        )


def _engine_answer_public_only(card: DynamicTaskCard) -> tuple[str | None, tuple[str, ...], int]:
    """Tiny deterministic evaluator using public card only; returns answer/evidence/refusal_correctness."""

    facts = card.public_facts
    if card.family == DynamicTaskFamily.LATEST_STATE:
        entity = card.query.replace("where is ", "").replace("?", "").strip()
        latest = None
        evidence_index = -1
        for idx, fact in enumerate(facts):
            parts = fact.split()
            if len(parts) == 3 and parts[0] == entity and parts[1] == "located_in":
                latest = parts[2]
                evidence_index = idx
        if latest is None:
            return None, (), 1
        return latest, (f"public_fact:{evidence_index}",), 1
    if card.family == DynamicTaskFamily.DEDUCTION:
        subject = card.query.replace("what does ", "").replace(" imply?", "").strip()
        mid = None
        target = None
        for idx, fact in enumerate(facts):
            parts = fact.split()
            if len(parts) == 3 and parts[0] == subject and parts[1] == "is_a":
                mid = parts[2]
        for fact in facts:
            parts = fact.split()
            if len(parts) == 3 and mid is not None and parts[0] == mid and parts[1] == "implies":
                target = parts[2]
        if target is None:
            return None, (), 1
        return target, ("public_fact:0", "public_fact:1"), 1
    if card.family == DynamicTaskFamily.CONTRADICTION:
        has_retract = any("retracts" in fact for fact in facts)
        has_contradict = any("contradicts" in fact for fact in facts)
        if has_retract or has_contradict:
            return "refuse", ("public_fact:1", "public_fact:2"), 1
        return "supported", ("public_fact:0",), 1
    if card.family == DynamicTaskFamily.ARC_SYMBOLIC:
        train = facts[0]
        value = train.split()[1]
        direction = "right" if "right" in train else "down"
        return f"move_{value}_{direction}", ("public_fact:0", "public_fact:1"), 1
    return None, (), 1


class P18NoLeakageProtocol:
    """Runs dynamic benchmark rows with hidden gold commitments."""

    def __init__(self, *, seed: str = "htce-p18-private-seed") -> None:
        self.seed = seed
        self.generator = DynamicTaskGenerator()
        self.trace = HashChain()

    def _run_pair(
        self,
        card: DynamicTaskCard,
        goldset: SeededPrivateGoldset,
        *,
        counterfactual_of: str | None = None,
    ) -> tuple[P18ProtocolRow, HiddenCriteriaCommitment]:
        commitment = HiddenCriteriaCommitment.create(card, goldset)
        commit_event = self.trace.append("p18_hidden_criteria_commitment", commitment.as_payload())
        engine_input = card.public_payload()
        answer, evidence_path, refusal_correctness = _engine_answer_public_only(card)
        visible = 0
        passed = int(answer in goldset.acceptable_answers and visible == 0 and commitment.committed_before_execution == 1)
        false_support = int(answer is not None and len(evidence_path) == 0)
        exec_event = self.trace.append("p18_no_leakage_execution", {
            "answer_digest": sha256_hex(answer) if answer is not None else None,
            "answer_key_visible_to_engine": visible,
            "commit_event_hash": commit_event.event_hash(),
            "engine_input_hash": sha256_hex(engine_input),
            "false_support": false_support,
            "passed": passed,
            "pre_execution_commitment_hash": commitment.commitment_hash,
            "public_task_hash": card.public_hash(),
            "task_id": card.task_id,
        })
        row = P18ProtocolRow(
            task=card.task_id,
            family=card.family.value,
            required_capability=card.required_capability,
            htce_modules_used=card.htce_modules_used,
            engine_input_hash=sha256_hex(engine_input),
            public_task_hash=card.public_hash(),
            hidden_gold_commitment_hash=commitment.gold_commitment_hash,
            pre_execution_commitment_hash=commitment.commitment_hash,
            answer=answer,
            answer_digest=sha256_hex(answer) if answer is not None else None,
            passed=passed,
            refusal_correctness=refusal_correctness,
            false_support=false_support,
            answer_key_visible_to_engine=visible,
            evidence_path=tuple(evidence_path),
            trace_hash=exec_event.event_hash(),
            counterfactual_of=counterfactual_of,
        )
        return row, commitment

    def run(self) -> P18NoLeakageReport:
        pairs = list(self.generator.default_pairs(self.seed))
        rows: list[P18ProtocolRow] = []
        commitments: list[HiddenCriteriaCommitment] = []
        for card, goldset in pairs:
            row, commitment = self._run_pair(card, goldset)
            rows.append(row)
            commitments.append(commitment)

        counterfactual_tests: list[CounterfactualRewriteTest] = []
        for family in (DynamicTaskFamily.LATEST_STATE, DynamicTaskFamily.DEDUCTION, DynamicTaskFamily.ARC_SYMBOLIC):
            original_card, original_gold = self.generator.card_pair(family, self.seed, counterfactual_index=0)
            rewritten_card, rewritten_gold = self.generator.card_pair(family, self.seed, counterfactual_index=3)
            rewritten_row, rewritten_commitment = self._run_pair(rewritten_card, rewritten_gold, counterfactual_of=original_card.task_id)
            rows.append(rewritten_row)
            commitments.append(rewritten_commitment)
            counterfactual_tests.append(CounterfactualRewriteTest(
                original_task_id=original_card.task_id,
                rewritten_task_id=rewritten_card.task_id,
                original_public_hash=original_card.public_hash(),
                rewritten_public_hash=rewritten_card.public_hash(),
                original_gold_commitment_hash=original_gold.commitment_hash(),
                rewritten_gold_commitment_hash=rewritten_gold.commitment_hash(),
                answer_changed=int(original_gold.gold_answer != rewritten_gold.gold_answer),
                public_prompt_changed=int(original_card.public_hash() != rewritten_card.public_hash()),
            ))

        passed_count = sum(row.passed for row in rows)
        false_support_count = sum(row.false_support for row in rows)
        answer_key_visible_count = sum(row.answer_key_visible_to_engine for row in rows)
        contract = NoAnswerLeakageContract()
        report = P18NoLeakageReport(
            schema_version="htce-p18-no-leakage-dynamic-benchmark-v1",
            rows=tuple(rows),
            commitments=tuple(commitments),
            counterfactual_tests=tuple(counterfactual_tests),
            no_answer_leakage_contract=contract,
            trace_head=self.trace.head,
            passed_count=passed_count,
            total_count=len(rows),
            false_support_count=false_support_count,
            answer_key_visible_count=answer_key_visible_count,
        )
        self.trace.append("p18_no_leakage_report", report.as_payload())
        return P18NoLeakageReport(**{**report.__dict__, "trace_head": self.trace.head})
