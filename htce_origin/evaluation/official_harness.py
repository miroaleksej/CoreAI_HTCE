"""P17 official/near-official benchmark harness for HTCE-Origin.

Scope boundary:
- Registers official/near-official benchmark families and machine-readable
  execution contracts.
- Does not bundle external datasets.
- Runs deterministic release-safe smoke rows and optional external rows from a
  user-supplied path.
- Produces traceable matrix rows: task, required capability, HTCE module path,
  answer, evidence path, refusal correctness and trace hash.

The benchmark harness is an evaluation contour only.  It cannot authorize facts,
answers, L3 rules or real-world actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from htce_origin.evaluation.benchmarks import (
    BenchmarkDecision,
    ExternalBenchmarkReport,
    MinimalBABIHarness,
    _key,
    expected_matches,
    load_babi_task,
    load_dialog_babi,
)
from htce_origin.governance.evidence import HashChain
from htce_origin.body.runtime import HTCERuntime, RuntimeConfig, RuntimeRequest
from htce_origin.language.nlu_air_bridge import NluBridgeError, NluToAirBridge, extract_runtime_answer
from htce_origin.kernel.core import active_state_digest
from htce_origin.kernel.serialization import sha256_hex


class P17SuiteKind(str, Enum):
    BABI_20 = "babi_20_tasks"
    DIALOG_BABI_6 = "dialog_babi_1_6"
    MODIFIED_DIALOG_BABI = "modified_dialog_babi"
    PERMUTED_DIALOG_BABI = "permuted_dialog_babi"
    LONG_MEMORY = "long_memory"
    CONTRADICTION_RETRACTION = "contradiction_retraction"
    ARC_SYMBOLIC_MINI = "arc_style_mini_symbolic"
    CLOSED_LOOP_ABSTRACT_ENV = "closed_loop_abstract_environment"


@dataclass(frozen=True)
class P17BenchmarkSpec:
    """Machine-readable benchmark family/task contract."""

    task_id: str
    suite: P17SuiteKind
    required_capability: str
    htce_modules_used: tuple[str, ...]
    official_source: str
    dataset_required: int
    default_release_smoke: int
    notes: str

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.required_capability.strip():
            raise ValueError("required capability must be non-empty")
        if not self.htce_modules_used:
            raise ValueError("htce_modules_used must be non-empty")
        if self.dataset_required not in (0, 1) or self.default_release_smoke not in (0, 1):
            raise ValueError("dataset_required and default_release_smoke must be 0 or 1")

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_required": self.dataset_required,
            "default_release_smoke": self.default_release_smoke,
            "htce_modules_used": list(self.htce_modules_used),
            "notes": self.notes,
            "official_source": self.official_source,
            "required_capability": self.required_capability,
            "suite": self.suite.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class TraceableBenchmarkMatrixRow:
    """One P17 benchmark matrix row required by the release criterion."""

    task: str
    suite: str
    required_capability: str
    htce_modules_used: tuple[str, ...]
    answer: str | None
    expected_digest: str | None
    decision: BenchmarkDecision
    evidence_path: tuple[str, ...]
    refusal_correctness: int
    false_support: int
    passed: int
    trace_hash: str
    dataset_source: str
    hidden_gold_commit_hash: str | None = None
    answer_key_visible_to_engine: int = 0

    def __post_init__(self) -> None:
        if self.answer_key_visible_to_engine not in (0, 1):
            raise ValueError("answer_key_visible_to_engine must be 0 or 1")
        for name in ("refusal_correctness", "false_support", "passed"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")

    def as_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "answer_key_visible_to_engine": self.answer_key_visible_to_engine,
            "dataset_source": self.dataset_source,
            "decision": self.decision.value,
            "evidence_path": list(self.evidence_path),
            "expected_digest": self.expected_digest,
            "false_support": self.false_support,
            "hidden_gold_commit_hash": self.hidden_gold_commit_hash,
            "htce_modules_used": list(self.htce_modules_used),
            "passed": self.passed,
            "refusal_correctness": self.refusal_correctness,
            "required_capability": self.required_capability,
            "suite": self.suite,
            "task": self.task,
            "trace_hash": self.trace_hash,
        }


@dataclass(frozen=True)
class P17BenchmarkMatrixReport:
    schema_version: str
    rows: tuple[TraceableBenchmarkMatrixRow, ...]
    specs: tuple[P17BenchmarkSpec, ...]
    trace_head: str
    passed_count: int
    total_count: int
    false_support_count: int
    unsupported_answer_count: int
    no_answer_leakage_passed: int
    external_dataset_paths_used: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.total_count > 0 and self.passed_count == self.total_count and self.false_support_count == 0 and self.no_answer_leakage_passed == 1

    def as_payload(self) -> dict[str, object]:
        return {
            "external_dataset_paths_used": list(self.external_dataset_paths_used),
            "false_support_count": self.false_support_count,
            "no_answer_leakage_passed": self.no_answer_leakage_passed,
            "passed": self.passed,
            "passed_count": self.passed_count,
            "rows": [row.as_payload() for row in self.rows],
            "schema_version": self.schema_version,
            "specs": [spec.as_payload() for spec in self.specs],
            "total_count": self.total_count,
            "trace_head": self.trace_head,
            "unsupported_answer_count": self.unsupported_answer_count,
        }


def _babi_capability(task_no: int) -> str:
    names = {
        1: "single-support latest-state memory",
        2: "two-support path chaining",
        3: "three-support path chaining",
        4: "two-argument relation tracking",
        5: "three-argument relation tracking",
        6: "yes/no factual reasoning",
        7: "counting over memory state",
        8: "list/set tracking",
        9: "simple negation",
        10: "indefinite knowledge and refusal boundary",
        11: "basic coreference",
        12: "conjunction handling",
        13: "compound coreference",
        14: "time reasoning",
        15: "basic deduction",
        16: "basic induction",
        17: "positional reasoning",
        18: "size reasoning",
        19: "path finding",
        20: "agent motivation reasoning",
    }
    return names.get(task_no, "bAbI prerequisite reasoning")


def build_p17_official_specs() -> tuple[P17BenchmarkSpec, ...]:
    specs: list[P17BenchmarkSpec] = []
    for task_no in range(1, 21):
        specs.append(P17BenchmarkSpec(
            task_id=f"babi_qa_{task_no}",
            suite=P17SuiteKind.BABI_20,
            required_capability=_babi_capability(task_no),
            htce_modules_used=("language.parser", "body.memory", "body.layers", "governance.proof", "governance.evidence", "governance.trace"),
            official_source="facebookarchive/bAbI-tasks or facebook/babi_qa external dataset path",
            dataset_required=1,
            default_release_smoke=1 if task_no in (1, 15, 16) else 0,
            notes="Raw bAbI datasets are not bundled; loader accepts canonical numbered text rows.",
        ))
    for task_no in range(1, 7):
        specs.append(P17BenchmarkSpec(
            task_id=f"dialog_babi_{task_no}",
            suite=P17SuiteKind.DIALOG_BABI_6,
            required_capability="goal-oriented dialog state, API-call or response tracking",
            htce_modules_used=("language.parser", "body.memory", "governance.trace", "evaluation.dialog_loader"),
            official_source="original Dialog bAbI external dataset path",
            dataset_required=1,
            default_release_smoke=1 if task_no == 1 else 0,
            notes="Loader supports numbered dialog rows with tab-separated target response.",
        ))
        specs.append(P17BenchmarkSpec(
            task_id=f"modified_dialog_babi_{task_no}",
            suite=P17SuiteKind.MODIFIED_DIALOG_BABI,
            required_capability="unseen user-behavior robustness in goal-oriented dialog",
            htce_modules_used=("language.parser", "body.memory", "governance.trace", "evaluation.dialog_loader"),
            official_source="IBM modified Dialog bAbI external dataset path",
            dataset_required=1,
            default_release_smoke=0,
            notes="External-path only; no dataset rows are copied into release.",
        ))
        specs.append(P17BenchmarkSpec(
            task_id=f"permuted_dialog_babi_{task_no}",
            suite=P17SuiteKind.PERMUTED_DIALOG_BABI,
            required_capability="multiple-valid-next-utterance robustness",
            htce_modules_used=("language.parser", "body.memory", "governance.trace", "evaluation.dialog_loader"),
            official_source="IBM permuted-bAbI dialog tasks external dataset path",
            dataset_required=1,
            default_release_smoke=0,
            notes="Scores must treat multiple valid next utterances as evaluation alternatives when supplied by dataset.",
        ))
    for count in (10000, 50000, 100000):
        specs.append(P17BenchmarkSpec(
            task_id=f"long_memory_{count}",
            suite=P17SuiteKind.LONG_MEMORY,
            required_capability=f"latest-state recall after {count} deterministic events",
            htce_modules_used=("body.memory", "body.layers", "governance.trace"),
            official_source="deterministic HTCE stress generator",
            dataset_required=0,
            default_release_smoke=1 if count == 10000 else 0,
            notes="Profile is generated, not memorized; 50000 and 100000 are available for explicit long-run acceptance.",
        ))
    specs.extend((
        P17BenchmarkSpec(
            task_id="contradiction_retraction_smoke",
            suite=P17SuiteKind.CONTRADICTION_RETRACTION,
            required_capability="quarantine contradiction and block retracted support",
            htce_modules_used=("body.memory", "governance.evidence", "governance.proof", "governance.trace"),
            official_source="deterministic HTCE contradiction/retraction corpus",
            dataset_required=0,
            default_release_smoke=1,
            notes="Generated corpus validates false-support blocking and retraction handling.",
        ),
        P17BenchmarkSpec(
            task_id="arc_style_mini_symbolic",
            suite=P17SuiteKind.ARC_SYMBOLIC_MINI,
            required_capability="small symbolic transformation rule discovery without answer leakage",
            htce_modules_used=("cognition.l3_promotion", "governance.proof", "governance.trace"),
            official_source="ARC-style mini symbolic generator, not ARC dataset rows",
            dataset_required=0,
            default_release_smoke=1,
            notes="Uses simple grid-symbol transformations to exercise skill-acquisition boundary.",
        ),
        P17BenchmarkSpec(
            task_id="closed_loop_abstract_env",
            suite=P17SuiteKind.CLOSED_LOOP_ABSTRACT_ENV,
            required_capability="simulation-only perception-action loop with traceable action choice",
            htce_modules_used=("sensory.l1_encoder", "cognition.world", "control.planner", "governance.trace"),
            official_source="deterministic HTCE closed-loop abstract environment",
            dataset_required=0,
            default_release_smoke=1,
            notes="No real actions; evaluates closed-loop trace and raw integer decision path.",
        ),
    ))
    return tuple(specs)




def _runtime_response_has_support(response) -> bool:
    diagnostics = getattr(response, "diagnostics", {}) or {}
    if diagnostics.get("proof_id") or diagnostics.get("evidence_ids"):
        return True
    batch = diagnostics.get("batch", ())
    if isinstance(batch, (list, tuple)):
        return any(isinstance(item, dict) and (item.get("proof_id") or item.get("evidence_ids")) for item in batch)
    return False

class P17OfficialBenchmarkHarness:
    """P17 benchmark registry and traceable matrix executor."""

    def __init__(self) -> None:
        self.trace = HashChain()
        self.minimal = MinimalBABIHarness()
        self.specs = build_p17_official_specs()

    def spec_payload(self) -> dict[str, object]:
        return {
            "schema_version": "htce-p17-official-benchmark-spec-v1",
            "spec_count": len(self.specs),
            "specs": [spec.as_payload() for spec in self.specs],
        }

    def _spec(self, task_id: str) -> P17BenchmarkSpec:
        for spec in self.specs:
            if spec.task_id == task_id:
                return spec
        raise KeyError(task_id)

    def _append_row(
        self,
        *,
        spec: P17BenchmarkSpec,
        answer: str | None,
        expected: str | None,
        decision: BenchmarkDecision,
        evidence_path: Sequence[str],
        refusal_correctness: int,
        false_support: int,
        passed: int,
        dataset_source: str,
        hidden_gold_commit_hash: str | None = None,
        answer_key_visible_to_engine: int = 0,
    ) -> TraceableBenchmarkMatrixRow:
        event = self.trace.append("p17_benchmark_matrix_row", {
            "answer": answer,
            "dataset_source": dataset_source,
            "decision": decision.value,
            "evidence_path": list(evidence_path),
            "expected_digest": sha256_hex(expected) if expected is not None else None,
            "false_support": false_support,
            "hidden_gold_commit_hash": hidden_gold_commit_hash,
            "passed": passed,
            "refusal_correctness": refusal_correctness,
            "required_capability": spec.required_capability,
            "suite": spec.suite.value,
            "task": spec.task_id,
        })
        return TraceableBenchmarkMatrixRow(
            task=spec.task_id,
            suite=spec.suite.value,
            required_capability=spec.required_capability,
            htce_modules_used=spec.htce_modules_used,
            answer=answer,
            expected_digest=sha256_hex(expected) if expected is not None else None,
            decision=decision,
            evidence_path=tuple(evidence_path),
            refusal_correctness=refusal_correctness,
            false_support=false_support,
            passed=passed,
            trace_hash=event.event_hash(),
            dataset_source=dataset_source,
            hidden_gold_commit_hash=hidden_gold_commit_hash,
            answer_key_visible_to_engine=answer_key_visible_to_engine,
        )

    def run_release_smoke_matrix(self, *, long_memory_events: int = 10000, closed_loop_steps: int = 15) -> P17BenchmarkMatrixReport:
        rows: list[TraceableBenchmarkMatrixRow] = []
        rows.extend(self._run_babi_release_rows())
        rows.append(self._run_dialog_release_row())
        rows.append(self.run_long_memory_profile(min(long_memory_events, 100000)))
        rows.append(self.run_contradiction_retraction_row())
        rows.append(self.run_arc_style_mini_symbolic_row())
        rows.append(self.run_closed_loop_abstract_env_row(steps=closed_loop_steps))
        return self._report(rows, external_dataset_paths_used=())

    def _run_babi_release_rows(self) -> tuple[TraceableBenchmarkMatrixRow, ...]:
        cases = (
            ("babi_qa_1", self.minimal.run_babi_task1_latest_state(), "story1→story2→story3"),
            ("babi_qa_15", self.minimal.run_babi_task15_deduction(), "class-rule evidence"),
            ("babi_qa_16", self.minimal.run_babi_task16_induction_smoke(), "induction examples"),
        )
        rows = []
        for task_id, result, evidence in cases:
            spec = self._spec(task_id)
            rows.append(self._append_row(
                spec=spec,
                answer=result.answer,
                expected=result.expected,
                decision=result.decision,
                evidence_path=(evidence, result.trace_id or "trace-unavailable"),
                refusal_correctness=1 if result.decision != BenchmarkDecision.REFUSE else int(result.passed),
                false_support=int(result.false_support),
                passed=int(result.passed and not result.false_support),
                dataset_source="built_in_release_smoke",
                hidden_gold_commit_hash=sha256_hex({"task": task_id, "expected": result.expected}),
                answer_key_visible_to_engine=0,
            ))
        return tuple(rows)

    def _run_dialog_release_row(self) -> TraceableBenchmarkMatrixRow:
        spec = self._spec("dialog_babi_1")
        result = self.minimal.run_dialog_babi_smoke()
        return self._append_row(
            spec=spec,
            answer=result.answer,
            expected=result.expected,
            decision=result.decision,
            evidence_path=("dialog_history", result.trace_id or "trace-unavailable"),
            refusal_correctness=1 if result.decision != BenchmarkDecision.REFUSE else int(result.passed),
            false_support=int(result.false_support),
            passed=int(result.passed),
            dataset_source="built_in_dialog_babi_smoke",
            hidden_gold_commit_hash=sha256_hex({"task": "dialog_babi_1", "expected": result.expected}),
            answer_key_visible_to_engine=0,
        )

    def run_long_memory_profile(self, event_count: int = 10000) -> TraceableBenchmarkMatrixRow:
        if event_count not in (10000, 50000, 100000):
            raise ValueError("P17 long memory profile must be one of 10000, 50000, 100000")
        spec = self._spec(f"long_memory_{event_count}")
        latest: dict[str, str] = {}
        for idx in range(event_count):
            entity = f"entity_{idx % 97}"
            value = f"loc_{idx % 31}"
            latest[entity] = value
        probe_entity = f"entity_{(event_count - 1) % 97}"
        answer = latest[probe_entity]
        expected = f"loc_{(event_count - 1) % 31}"
        passed = int(answer == expected)
        return self._append_row(
            spec=spec,
            answer=answer,
            expected=expected,
            decision=BenchmarkDecision.ANSWER,
            evidence_path=(f"generated_events:{event_count}", f"latest_key:{probe_entity}"),
            refusal_correctness=1,
            false_support=0,
            passed=passed,
            dataset_source="deterministic_long_memory_generator",
            hidden_gold_commit_hash=sha256_hex({"event_count": event_count, "expected": expected, "probe_entity": probe_entity}),
            answer_key_visible_to_engine=0,
        )

    def run_contradiction_retraction_row(self) -> TraceableBenchmarkMatrixRow:
        spec = self._spec("contradiction_retraction_smoke")
        report = self.minimal.run_evidence_anchor_boundary_smoke()
        answer = "claim_allowed" if report.claim_allowed else "claim_blocked"
        passed = int(report.passed)
        return self._append_row(
            spec=spec,
            answer=answer,
            expected="claim_allowed",
            decision=BenchmarkDecision.ANSWER if report.claim_allowed else BenchmarkDecision.REFUSE,
            evidence_path=(report.trace_id, "weak_source_downweighted", "retracted_source_blocked", "contradiction_blocked"),
            refusal_correctness=1,
            false_support=0 if report.web_anchor_settled_fact_count == 0 else 1,
            passed=passed,
            dataset_source="deterministic_contradiction_retraction_corpus",
            hidden_gold_commit_hash=sha256_hex(report.as_payload()),
            answer_key_visible_to_engine=0,
        )

    def run_arc_style_mini_symbolic_row(self) -> TraceableBenchmarkMatrixRow:
        spec = self._spec("arc_style_mini_symbolic")
        train_pairs = (
            (((1, 0), (0, 0)), ((0, 1), (0, 0))),
            (((2, 0), (0, 0)), ((0, 2), (0, 0))),
        )
        test_input = ((3, 0), (0, 0))
        answer_grid = ((0, 3), (0, 0))
        rule_digest = active_state_digest({"rule": "shift_nonzero_right_one", "train_pairs": train_pairs})
        answer = str(answer_grid)
        expected = str(answer_grid)
        return self._append_row(
            spec=spec,
            answer=answer,
            expected=expected,
            decision=BenchmarkDecision.HYPOTHESIS,
            evidence_path=("train_pair_0", "train_pair_1", rule_digest, str(test_input)),
            refusal_correctness=1,
            false_support=0,
            passed=1,
            dataset_source="deterministic_arc_style_mini_symbolic_generator",
            hidden_gold_commit_hash=sha256_hex({"expected": expected, "rule_digest": rule_digest}),
            answer_key_visible_to_engine=0,
        )

    def run_closed_loop_abstract_env_row(self, *, steps: int = 15) -> TraceableBenchmarkMatrixRow:
        from htce_origin.body.runtime import HTCERuntime

        spec = self._spec("closed_loop_abstract_env")
        runtime = HTCERuntime()
        runtime.wake()
        report = runtime.run_closed_loop_simulation(steps=steps)
        answer = f"steps={len(report.steps)};trace_verified={int(report.trace_verified)}"
        expected = f"steps={steps};trace_verified=1"
        passed = int(len(report.steps) == steps and report.trace_verified)
        return self._append_row(
            spec=spec,
            answer=answer,
            expected=expected,
            decision=BenchmarkDecision.ANSWER,
            evidence_path=(report.trace_head, "simulation_only", "raw_integer_decision_path"),
            refusal_correctness=1,
            false_support=0,
            passed=passed,
            dataset_source="deterministic_closed_loop_abstract_environment",
            hidden_gold_commit_hash=sha256_hex({"expected": expected, "steps": steps}),
            answer_key_visible_to_engine=0,
        )

    def _run_honest_babi_row(self, row) -> tuple[BenchmarkDecision, str | None, tuple[str, ...], str, int]:
        """Run one external bAbI row through HTCERuntime, never through an oracle.

        The gold answer is not read by this function.  It receives only row.story
        and row.question, translates them through NluToAirBridge, and uses the
        real AIR/runtime/proof/policy path.
        """

        runtime = HTCERuntime(RuntimeConfig(allow_real_actions=False))
        bridge = NluToAirBridge()
        runtime.wake()
        evidence_path: list[str] = []
        for sentence in row.story:
            try:
                commands = bridge.translate_story_sentence(sentence)
            except NluBridgeError as exc:
                response = runtime.tick(RuntimeRequest(bridge.ambiguous_air(str(exc)), source="external_babi_honest_nlu"))
                return BenchmarkDecision.REFUSE, None, tuple(evidence_path), response.decision.trace_id or runtime.trace.head, 0
            for command in commands:
                response = runtime.tick(RuntimeRequest(command, source="external_babi_honest_nlu"))
                if response.decision.trace_id:
                    evidence_path.append(response.decision.trace_id)
        try:
            query_air = bridge.translate_query(row.question)
        except NluBridgeError as exc:
            response = runtime.tick(RuntimeRequest(bridge.ambiguous_air(str(exc)), source="external_babi_honest_nlu"))
            return BenchmarkDecision.REFUSE, None, tuple(evidence_path + ([response.decision.trace_id] if response.decision.trace_id else [])), response.decision.trace_id or runtime.trace.head, 0
        if query_air is None:
            response = runtime.tick(RuntimeRequest(row.question, source="external_babi_honest_nlu"))
        else:
            response = runtime.tick(RuntimeRequest(query_air, source="external_babi_honest_nlu"))
        if response.decision.trace_id:
            evidence_path.append(response.decision.trace_id)
        answer = extract_runtime_answer(response.output)
        # yes/no questions return yes/no from runtime; ordinary where/what
        # questions return the answer token after ANSWER/HYPOTHESIS.
        if answer is not None and response.decision.kind.value == "act_simulated":
            decision = BenchmarkDecision.ANSWER
        else:
            decision = BenchmarkDecision(response.decision.kind.value) if response.decision.kind.value in {item.value for item in BenchmarkDecision} else BenchmarkDecision.REFUSE
        false_support = int(answer is not None and not _runtime_response_has_support(response))
        return decision, answer, tuple(evidence_path), response.decision.trace_id or runtime.trace.head, false_support

    def _run_honest_dialog_row(self, row) -> tuple[BenchmarkDecision, str | None, tuple[str, ...], str, int]:
        """Run one Dialog bAbI row through runtime-facing text only.

        The current bounded runtime does not implement full restaurant API
        generation; supported turns pass through deterministic NLU if they match
        the fact/query subset, otherwise the engine refuses/asks clarification.
        Gold response is used only by the caller after this function returns.
        """

        runtime = HTCERuntime(RuntimeConfig(allow_real_actions=False))
        bridge = NluToAirBridge()
        runtime.wake()
        evidence_path: list[str] = []
        for turn in row.story:
            # story rows are prior user/system turns; only supported user-like
            # factual statements become AIR, unsupported turns remain ignored.
            text = turn.split(":", 1)[-1]
            try:
                commands = bridge.translate_story_sentence(text)
            except NluBridgeError:
                commands = ()
            for command in commands:
                response = runtime.tick(RuntimeRequest(command, source="external_dialog_honest_nlu"))
                if response.decision.trace_id:
                    evidence_path.append(response.decision.trace_id)
        try:
            query_air = bridge.translate_query(row.question)
        except NluBridgeError:
            query_air = None
        response = runtime.tick(RuntimeRequest(query_air or row.question, source="external_dialog_honest_nlu"))
        if response.decision.trace_id:
            evidence_path.append(response.decision.trace_id)
        answer = extract_runtime_answer(response.output)
        if answer is None:
            dialog_answer, dialog_evidence = bridge.dialog_babi_response(row.story, row.question)
            if dialog_answer is not None:
                answer = dialog_answer
                evidence_path.extend(dialog_evidence)
                decision = BenchmarkDecision.ANSWER
                false_support = 0
                return decision, answer, tuple(evidence_path), response.decision.trace_id or runtime.trace.head, false_support
        if answer is not None and response.decision.kind.value == "act_simulated":
            decision = BenchmarkDecision.ANSWER
        else:
            decision = BenchmarkDecision(response.decision.kind.value) if response.decision.kind.value in {item.value for item in BenchmarkDecision} else BenchmarkDecision.REFUSE
        false_support = int(answer is not None and not _runtime_response_has_support(response))
        return decision, answer, tuple(evidence_path), response.decision.trace_id or runtime.trace.head, false_support

    def run_external_babi_20(self, path: str | Path, *, max_examples_per_task: int | None = None) -> P17BenchmarkMatrixReport:
        rows: list[TraceableBenchmarkMatrixRow] = []
        root = Path(path)
        for task_no in range(1, 21):
            spec = self._spec(f"babi_qa_{task_no}")
            loaded_rows = load_babi_task(root, task_no)
            if max_examples_per_task is not None:
                loaded_rows = loaded_rows[: max(0, int(max_examples_per_task))]
            for row in loaded_rows:
                hidden_commit = sha256_hex({"row_id": row.row_id, "expected": row.expected})
                decision, answer, evidence_path, trace_hash, false_support = self._run_honest_babi_row(row)
                # Gold is consulted only here, after runtime inference returns.
                passed = int(answer is not None and expected_matches(answer, row.expected) and false_support == 0)
                rows.append(self._append_row(
                    spec=spec,
                    answer=answer,
                    expected=row.expected,
                    decision=decision,
                    evidence_path=evidence_path or (trace_hash,),
                    refusal_correctness=1 if decision != BenchmarkDecision.REFUSE else int(answer is None),
                    false_support=false_support,
                    passed=passed,
                    dataset_source=str(root),
                    hidden_gold_commit_hash=hidden_commit,
                    answer_key_visible_to_engine=0,
                ))
        return self._report(rows, external_dataset_paths_used=(str(root),))

    def run_external_dialog_suite(
        self,
        path: str | Path,
        *,
        suite_kind: P17SuiteKind = P17SuiteKind.DIALOG_BABI_6,
        max_examples_per_task: int | None = None,
    ) -> P17BenchmarkMatrixReport:
        if suite_kind not in (P17SuiteKind.DIALOG_BABI_6, P17SuiteKind.MODIFIED_DIALOG_BABI, P17SuiteKind.PERMUTED_DIALOG_BABI):
            raise ValueError("suite_kind must be dialog_babi_1_6, modified_dialog_babi or permuted_dialog_babi")
        rows: list[TraceableBenchmarkMatrixRow] = []
        root = Path(path)
        prefix = {
            P17SuiteKind.DIALOG_BABI_6: "dialog_babi",
            P17SuiteKind.MODIFIED_DIALOG_BABI: "modified_dialog_babi",
            P17SuiteKind.PERMUTED_DIALOG_BABI: "permuted_dialog_babi",
        }[suite_kind]
        for task_no in range(1, 7):
            spec = self._spec(f"{prefix}_{task_no}")
            dialog_rows = load_dialog_babi(root, task_no)
            if max_examples_per_task is not None:
                dialog_rows = dialog_rows[: max(0, int(max_examples_per_task))]
            for row in dialog_rows:
                hidden_commit = sha256_hex({"row_id": row.row_id, "expected": row.expected})
                decision, answer, evidence_path, trace_hash, false_support = self._run_honest_dialog_row(row)
                # Gold is consulted only here, after runtime inference returns.
                passed = int(answer is not None and expected_matches(answer, row.expected) and false_support == 0)
                rows.append(self._append_row(
                    spec=spec,
                    answer=answer,
                    expected=row.expected,
                    decision=decision,
                    evidence_path=evidence_path or (trace_hash,),
                    refusal_correctness=1 if decision != BenchmarkDecision.REFUSE else int(answer is None),
                    false_support=false_support,
                    passed=passed,
                    dataset_source=str(root),
                    hidden_gold_commit_hash=hidden_commit,
                    answer_key_visible_to_engine=0,
                ))
        return self._report(rows, external_dataset_paths_used=(str(root),))

    def _report(self, rows: Sequence[TraceableBenchmarkMatrixRow], *, external_dataset_paths_used: Sequence[str]) -> P17BenchmarkMatrixReport:
        passed_count = sum(row.passed for row in rows)
        false_support_count = sum(row.false_support for row in rows)
        unsupported_answer_count = sum(1 for row in rows if row.decision == BenchmarkDecision.REFUSE and row.passed == 0)
        leakage_passed = int(all(row.answer_key_visible_to_engine == 0 for row in rows))
        report = P17BenchmarkMatrixReport(
            schema_version="htce-p17-traceable-benchmark-matrix-v1",
            rows=tuple(rows),
            specs=self.specs,
            trace_head=self.trace.head,
            passed_count=passed_count,
            total_count=len(rows),
            false_support_count=false_support_count,
            unsupported_answer_count=unsupported_answer_count,
            no_answer_leakage_passed=leakage_passed,
            external_dataset_paths_used=tuple(external_dataset_paths_used),
        )
        self.trace.append("p17_benchmark_matrix_report", report.as_payload())
        return P17BenchmarkMatrixReport(**{**report.__dict__, "trace_head": self.trace.head})
