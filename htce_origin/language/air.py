"""AIR v1 source language boundary for HTCE-Origin Clean Body.

AIR boundary implements a deliberately small AIR language.  It is not a Python
replacement and it does not mutate HTCE state.  Its only job is to convert
trusted AIR source into typed candidate events that later gates may pass
through policy/evidence/claim/proof/topology gates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from htce_origin.kernel.core import EntityId, EvidenceId, RelationId


class AIRException(ValueError):
    """Base class for AIR language, checking and execution errors."""

    code = "AIR_ERROR"

    def __init__(self, message: str, *, span: tuple[int, int] | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span
        if code is not None:
            self.code = code


class AIRLexicalError(AIRException):
    code = "AIR_LEXICAL_ERROR"


class AIRSyntaxError(AIRException):
    code = "AIR_SYNTAX_ERROR"


class AIRSecurityError(AIRException):
    code = "AIR_SECURITY_ERROR"


class AIRTypeError(AIRException):
    code = "AIR_TYPE_ERROR"


class AIRPolicyError(AIRException):
    code = "AIR_POLICY_ERROR"


class AIRVMError(AIRException):
    code = "AIR_VM_ERROR"


class AIROp(str, Enum):
    FACT = "FACT"
    QUERY = "QUERY"
    NEGATE = "NEGATE"
    GOAL = "GOAL"
    ACTION_SIM = "ACTION_SIM"
    POLICY_FORBID = "POLICY_FORBID"
    EVIDENCE = "EVIDENCE"
    CLAIM = "CLAIM"
    PROC = "PROC"
    CALL = "CALL"
    ENSURES = "ENSURES"
    PROVE = "PROVE"


class TokenKind(str, Enum):
    IDENT = "IDENT"
    NEWLINE = "NEWLINE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    EOF = "EOF"
    FACT = "FACT"
    QUERY = "QUERY"
    NEGATE = "NEGATE"
    EVID = "EVID"
    PROC = "PROC"
    CALL = "CALL"
    ENSURES = "ENSURES"
    ENDPROC = "ENDPROC"
    POLICY = "POLICY"
    FORBID = "FORBID"
    ACTION_SIM = "ACTION_SIM"
    PROVE = "PROVE"
    CLAIM = "CLAIM"
    GOAL = "GOAL"


KEYWORDS: Mapping[str, TokenKind] = {
    "FACT": TokenKind.FACT,
    "QUERY": TokenKind.QUERY,
    "NEGATE": TokenKind.NEGATE,
    "EVID": TokenKind.EVID,
    "PROC": TokenKind.PROC,
    "CALL": TokenKind.CALL,
    "ENSURES": TokenKind.ENSURES,
    "ENDPROC": TokenKind.ENDPROC,
    "POLICY": TokenKind.POLICY,
    "FORBID": TokenKind.FORBID,
    "ACTION_SIM": TokenKind.ACTION_SIM,
    "PROVE": TokenKind.PROVE,
    "CLAIM": TokenKind.CLAIM,
    "GOAL": TokenKind.GOAL,
}

# Raw JSON/LLM payloads are intentionally not AIR.  AIR source must use the
# narrow line-oriented grammar below.
FORBIDDEN_SOURCE_CHARS = frozenset({'{', '}', '[', ']', '"', "'", '`', '$', '\x00'})
IDENTIFIER_EXTRA_CHARS = frozenset({'_', '-', ':', '.', '/'})
INVALID_EVIDENCE_IDS = frozenset({"", "none", "null", "missing", "unknown", "-"})
DIRECT_MUTATION_TERMS = frozenset({
    "commit",
    "committed",
    "mutate",
    "mutation",
    "update_state",
    "write_state",
    "write_l2",
    "write_l3",
    "set_l2",
    "set_l3",
    "l2_commit",
    "l3_commit",
    "memory_commit",
})
DIRECT_STATE_TARGETS = frozenset({"l2", "l3", "memory", "runtime_state", "state"})


@dataclass(frozen=True)
class AIRToken:
    kind: TokenKind
    value: str
    start: int
    end: int


class AIRLexer:
    """Small deterministic AIR lexer.

    The lexer rejects JSON-like or quoted payloads to make sure natural-language
    or LLM-produced blobs cannot masquerade as executable AIR.
    """

    def tokenize(self, source: str) -> tuple[AIRToken, ...]:
        if not isinstance(source, str):
            raise AIRLexicalError("AIR source must be a string", code="AIR_SOURCE_NOT_STRING")
        bad = sorted(ch for ch in set(source) if ch in FORBIDDEN_SOURCE_CHARS)
        if bad:
            raise AIRSecurityError(
                f"forbidden AIR source character(s): {' '.join(repr(ch) for ch in bad)}",
                code="AIR_RAW_PAYLOAD_BLOCKED",
            )

        tokens: list[AIRToken] = []
        i = 0
        n = len(source)
        while i < n:
            ch = source[i]
            if ch in " \t\r":
                i += 1
                continue
            if ch == "\n":
                tokens.append(AIRToken(TokenKind.NEWLINE, ch, i, i + 1))
                i += 1
                continue
            if ch == "(":
                tokens.append(AIRToken(TokenKind.LPAREN, ch, i, i + 1))
                i += 1
                continue
            if ch == ")":
                tokens.append(AIRToken(TokenKind.RPAREN, ch, i, i + 1))
                i += 1
                continue
            if ch == ",":
                tokens.append(AIRToken(TokenKind.COMMA, ch, i, i + 1))
                i += 1
                continue
            if ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < n and (source[i].isalnum() or source[i] in IDENTIFIER_EXTRA_CHARS):
                    i += 1
                value = source[start:i]
                kind = KEYWORDS.get(value, TokenKind.IDENT)
                tokens.append(AIRToken(kind, value, start, i))
                continue
            if ch.isdigit():
                start = i
                i += 1
                while i < n and (source[i].isalnum() or source[i] in IDENTIFIER_EXTRA_CHARS):
                    i += 1
                tokens.append(AIRToken(TokenKind.IDENT, source[start:i], start, i))
                continue
            raise AIRLexicalError(f"illegal AIR character {ch!r}", span=(i, i + 1), code="AIR_ILLEGAL_CHARACTER")
        tokens.append(AIRToken(TokenKind.EOF, "", n, n))
        return tuple(tokens)


@dataclass(frozen=True)
class AIRIdentifier:
    value: str

    def __post_init__(self) -> None:
        text = self.value.strip()
        if not text:
            raise AIRTypeError("identifier must be non-empty", code="AIR_EMPTY_IDENTIFIER")
        object.__setattr__(self, "value", text)

    @property
    def canonical(self) -> str:
        return self.value.lower()


@dataclass(frozen=True)
class GoalExpression:
    name: AIRIdentifier
    arguments: tuple[AIRIdentifier, ...] = ()

    def as_text(self) -> str:
        if not self.arguments:
            return self.name.value
        args = ", ".join(arg.value for arg in self.arguments)
        return f"{self.name.value}({args})"


@dataclass(frozen=True)
class FactStatement:
    subject: AIRIdentifier
    relation: AIRIdentifier
    object: AIRIdentifier
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class QueryStatement:
    subject: AIRIdentifier
    query_type: AIRIdentifier
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class NegateStatement:
    subject: AIRIdentifier
    relation: AIRIdentifier
    object: AIRIdentifier
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class PolicyForbidStatement:
    target: AIRIdentifier
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class ProcedureStatement:
    name: AIRIdentifier
    ensures: GoalExpression
    body: tuple[object, ...] = ()
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class CallStatement:
    name: AIRIdentifier
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class ActionSimStatement:
    name: AIRIdentifier
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class ProveStatement:
    claim: GoalExpression
    evidence: EvidenceId
    source_span: tuple[int, int] | None = None


AIRStatement = FactStatement | QueryStatement | NegateStatement | PolicyForbidStatement | ProcedureStatement | CallStatement | ActionSimStatement | ProveStatement


@dataclass(frozen=True)
class AIRNode:
    """Backward-compatible generic AST wrapper."""

    op: AIROp
    args: dict[str, Any] = field(default_factory=dict)
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class AIRProgram:
    statements: tuple[AIRStatement, ...] = ()
    source_hash: str = ""

    @property
    def nodes(self) -> tuple[AIRNode, ...]:
        return tuple(statement_to_node(stmt) for stmt in self.statements)


@dataclass(frozen=True)
class AIRCheckResult:
    ok: bool
    reasons: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()


class AIRParser:
    def __init__(self, lexer: AIRLexer | None = None) -> None:
        self.lexer = lexer or AIRLexer()
        self._tokens: tuple[AIRToken, ...] = ()
        self._pos = 0

    def parse(self, source: str) -> AIRProgram:
        self._tokens = self.lexer.tokenize(source)
        self._pos = 0
        statements: list[AIRStatement] = []
        self._consume_newlines()
        while not self._at(TokenKind.EOF):
            statements.append(self._parse_statement())
            self._consume_newlines()
        return AIRProgram(statements=tuple(statements), source_hash=_source_hash(source))

    def _parse_statement(self) -> AIRStatement:
        tok = self._peek()
        if tok.kind == TokenKind.FACT:
            return self._parse_fact()
        if tok.kind == TokenKind.QUERY:
            return self._parse_query()
        if tok.kind == TokenKind.NEGATE:
            return self._parse_negate()
        if tok.kind == TokenKind.POLICY:
            return self._parse_policy_forbid()
        if tok.kind == TokenKind.PROC:
            return self._parse_proc()
        if tok.kind == TokenKind.CALL:
            return self._parse_call()
        if tok.kind == TokenKind.ACTION_SIM:
            return self._parse_action_sim()
        if tok.kind == TokenKind.PROVE:
            return self._parse_prove()
        raise AIRSyntaxError(
            f"expected AIR statement, got {tok.value!r}",
            span=(tok.start, tok.end),
            code="AIR_UNKNOWN_STATEMENT",
        )

    def _parse_fact(self) -> FactStatement:
        start = self._consume(TokenKind.FACT).start
        subject = self._identifier()
        relation = self._identifier()
        obj = self._identifier()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return FactStatement(subject, relation, obj, evidence, (start, end))

    def _parse_query(self) -> QueryStatement:
        start = self._consume(TokenKind.QUERY).start
        subject = self._identifier()
        query_type = self._identifier()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return QueryStatement(subject, query_type, evidence, (start, end))

    def _parse_negate(self) -> NegateStatement:
        start = self._consume(TokenKind.NEGATE).start
        subject = self._identifier()
        relation = self._identifier()
        obj = self._identifier()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return NegateStatement(subject, relation, obj, evidence, (start, end))

    def _parse_policy_forbid(self) -> PolicyForbidStatement:
        start = self._consume(TokenKind.POLICY).start
        self._consume(TokenKind.FORBID)
        target = self._identifier()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return PolicyForbidStatement(target, evidence, (start, end))

    def _parse_proc(self) -> ProcedureStatement:
        start = self._consume(TokenKind.PROC).start
        name = self._identifier()
        self._consume(TokenKind.ENSURES)
        ensures = self._goal_expression()
        if self._at(TokenKind.ENDPROC):
            self._advance()
        end = self._previous().end
        self._require_statement_end()
        return ProcedureStatement(name=name, ensures=ensures, body=(), source_span=(start, end))

    def _parse_call(self) -> CallStatement:
        start = self._consume(TokenKind.CALL).start
        name = self._identifier()
        end = self._previous().end
        self._require_statement_end()
        return CallStatement(name, (start, end))

    def _parse_action_sim(self) -> ActionSimStatement:
        start = self._consume(TokenKind.ACTION_SIM).start
        name = self._identifier()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return ActionSimStatement(name, evidence, (start, end))

    def _parse_prove(self) -> ProveStatement:
        start = self._consume(TokenKind.PROVE).start
        claim = self._goal_expression()
        evidence = self._evidence()
        end = self._previous().end
        self._require_statement_end()
        return ProveStatement(claim, evidence, (start, end))

    def _goal_expression(self) -> GoalExpression:
        name = self._identifier()
        args: list[AIRIdentifier] = []
        if self._match(TokenKind.LPAREN):
            if not self._at(TokenKind.RPAREN):
                args.append(self._identifier())
                while self._match(TokenKind.COMMA):
                    args.append(self._identifier())
            self._consume(TokenKind.RPAREN)
        return GoalExpression(name, tuple(args))

    def _evidence(self) -> EvidenceId:
        self._consume(TokenKind.EVID)
        ident = self._identifier()
        return EvidenceId(ident.value)

    def _identifier(self) -> AIRIdentifier:
        tok = self._consume(TokenKind.IDENT)
        return AIRIdentifier(tok.value)

    def _require_statement_end(self) -> None:
        if self._at(TokenKind.NEWLINE) or self._at(TokenKind.EOF):
            return
        tok = self._peek()
        raise AIRSyntaxError(
            f"unexpected token after statement: {tok.value!r}",
            span=(tok.start, tok.end),
            code="AIR_TRAILING_TOKENS",
        )

    def _consume_newlines(self) -> None:
        while self._at(TokenKind.NEWLINE):
            self._advance()

    def _consume(self, kind: TokenKind) -> AIRToken:
        tok = self._peek()
        if tok.kind != kind:
            raise AIRSyntaxError(
                f"expected {kind.value}, got {tok.value!r}",
                span=(tok.start, tok.end),
                code="AIR_EXPECTED_TOKEN",
            )
        return self._advance()

    def _match(self, kind: TokenKind) -> bool:
        if self._at(kind):
            self._advance()
            return True
        return False

    def _at(self, kind: TokenKind) -> bool:
        return self._peek().kind == kind

    def _peek(self) -> AIRToken:
        return self._tokens[self._pos]

    def _previous(self) -> AIRToken:
        return self._tokens[self._pos - 1]

    def _advance(self) -> AIRToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok


class AIRStaticChecker:
    """Type, evidence and minimal policy checker for AIR boundary AIR."""

    def check(self, program: AIRProgram) -> AIRCheckResult:
        reasons: list[str] = []
        codes: list[str] = []
        procedures: set[str] = set()
        forbidden: set[str] = set()

        for stmt in program.statements:
            if isinstance(stmt, ProcedureStatement):
                procedures.add(stmt.name.canonical)
            elif isinstance(stmt, PolicyForbidStatement):
                forbidden.add(stmt.target.canonical)

        for stmt in program.statements:
            self._check_evidence(stmt, reasons, codes)
            self._check_direct_mutation(stmt, reasons, codes)
            if isinstance(stmt, CallStatement):
                name = stmt.name.canonical
                if name not in procedures:
                    reasons.append(f"CALL references unknown PROC: {stmt.name.value}")
                    codes.append("AIR_UNKNOWN_PROCEDURE")
                if name in forbidden:
                    reasons.append(f"CALL is forbidden by policy: {stmt.name.value}")
                    codes.append("AIR_FORBIDDEN_ACTION")
            if isinstance(stmt, ActionSimStatement) and stmt.name.canonical in forbidden:
                reasons.append(f"ACTION_SIM is forbidden by policy: {stmt.name.value}")
                codes.append("AIR_FORBIDDEN_ACTION")
            if isinstance(stmt, ProcedureStatement) and not stmt.ensures.name.value:
                reasons.append(f"PROC has no ENSURES obligation: {stmt.name.value}")
                codes.append("AIR_PROC_WITHOUT_ENSURES")

        return AIRCheckResult(ok=not reasons, reasons=tuple(reasons), error_codes=tuple(codes))

    def require_ok(self, program: AIRProgram) -> None:
        result = self.check(program)
        if not result.ok:
            code = result.error_codes[0] if result.error_codes else "AIR_CHECK_FAILED"
            raise AIRTypeError("; ".join(result.reasons), code=code)

    def _check_evidence(self, stmt: AIRStatement, reasons: list[str], codes: list[str]) -> None:
        evidence = getattr(stmt, "evidence", None)
        if evidence is None:
            return
        value = evidence.value.lower()
        if value in INVALID_EVIDENCE_IDS:
            reasons.append("statement has missing or invalid evidence")
            codes.append("AIR_MISSING_EVIDENCE")

    def _check_direct_mutation(self, stmt: AIRStatement, reasons: list[str], codes: list[str]) -> None:
        identifiers = _statement_identifiers(stmt)
        canonical = [item.canonical for item in identifiers]
        has_mutation_term = any(item in DIRECT_MUTATION_TERMS for item in canonical)
        has_state_target = any(item in DIRECT_STATE_TARGETS for item in canonical)
        if has_mutation_term or has_state_target and isinstance(stmt, (FactStatement, NegateStatement, ActionSimStatement)):
            reasons.append("AIR source attempts direct L2/L3/runtime state mutation")
            codes.append("AIR_DIRECT_STATE_MUTATION_BLOCKED")


@dataclass(frozen=True)
class BytecodeInstruction:
    op: AIROp
    args: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None
    mutates_l2_l3: bool = False


class AIRBytecodeCompiler:
    def __init__(self, checker: AIRStaticChecker | None = None) -> None:
        self.checker = checker or AIRStaticChecker()

    def compile(self, program: AIRProgram) -> tuple[BytecodeInstruction, ...]:
        self.checker.require_ok(program)
        return tuple(_compile_statement(stmt) for stmt in program.statements)


@dataclass(frozen=True)
class AIRVMEvent:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None
    mutates_l2_l3: bool = False


class AIRVM:
    """AIR boundary bytecode VM.

    It emits candidate events only.  It never writes memory, L2/L3, topology,
    evidence logs or runtime state.  Mutation is a later gated runtime step.
    """

    def __init__(self, compiler: AIRBytecodeCompiler | None = None) -> None:
        self.compiler = compiler or AIRBytecodeCompiler()

    def execute(self, program_or_bytecode: AIRProgram | Sequence[BytecodeInstruction]) -> tuple[AIRVMEvent, ...]:
        if isinstance(program_or_bytecode, AIRProgram):
            bytecode = self.compiler.compile(program_or_bytecode)
        else:
            bytecode = tuple(program_or_bytecode)
        events: list[AIRVMEvent] = []
        for instruction in bytecode:
            if instruction.mutates_l2_l3:
                raise AIRVMError("AIR VM instruction attempted L2/L3 mutation", code="AIR_VM_MUTATION_BLOCKED")
            events.append(_event_from_instruction(instruction))
        return tuple(events)


Compiler = AIRBytecodeCompiler
BytecodeVM = AIRVM


def parse_air(source: str) -> AIRProgram:
    return AIRParser().parse(source)


def check_air(source: str) -> AIRCheckResult:
    return AIRStaticChecker().check(parse_air(source))


def compile_air(source: str) -> tuple[BytecodeInstruction, ...]:
    return AIRBytecodeCompiler().compile(parse_air(source))


def execute_air(source: str) -> tuple[AIRVMEvent, ...]:
    return AIRVM().execute(parse_air(source))


def statement_to_node(stmt: AIRStatement) -> AIRNode:
    instruction = _compile_statement(stmt)
    return AIRNode(instruction.op, dict(instruction.args), getattr(stmt, "source_span", None))


def _compile_statement(stmt: AIRStatement) -> BytecodeInstruction:
    if isinstance(stmt, FactStatement):
        return BytecodeInstruction(
            AIROp.FACT,
            {"subject": stmt.subject.value, "relation": stmt.relation.value, "object": stmt.object.value},
            stmt.evidence.value,
        )
    if isinstance(stmt, QueryStatement):
        return BytecodeInstruction(
            AIROp.QUERY,
            {"subject": stmt.subject.value, "query_type": stmt.query_type.value},
            stmt.evidence.value,
        )
    if isinstance(stmt, NegateStatement):
        return BytecodeInstruction(
            AIROp.NEGATE,
            {"subject": stmt.subject.value, "relation": stmt.relation.value, "object": stmt.object.value},
            stmt.evidence.value,
        )
    if isinstance(stmt, PolicyForbidStatement):
        return BytecodeInstruction(AIROp.POLICY_FORBID, {"target": stmt.target.value}, stmt.evidence.value)
    if isinstance(stmt, ProcedureStatement):
        return BytecodeInstruction(AIROp.PROC, {"name": stmt.name.value, "ensures": stmt.ensures.as_text()}, None)
    if isinstance(stmt, CallStatement):
        return BytecodeInstruction(AIROp.CALL, {"name": stmt.name.value}, None)
    if isinstance(stmt, ActionSimStatement):
        return BytecodeInstruction(AIROp.ACTION_SIM, {"name": stmt.name.value}, stmt.evidence.value)
    if isinstance(stmt, ProveStatement):
        return BytecodeInstruction(AIROp.PROVE, {"claim": stmt.claim.as_text()}, stmt.evidence.value)
    raise AIRVMError(f"cannot compile statement type {type(stmt).__name__}", code="AIR_COMPILE_UNKNOWN_STATEMENT")


def _event_from_instruction(instruction: BytecodeInstruction) -> AIRVMEvent:
    kind_by_op = {
        AIROp.FACT: "fact_candidate",
        AIROp.QUERY: "query_candidate",
        AIROp.NEGATE: "negation_candidate",
        AIROp.POLICY_FORBID: "policy_forbid_registered",
        AIROp.PROC: "procedure_registered",
        AIROp.CALL: "procedure_call_requested",
        AIROp.ACTION_SIM: "simulated_action_candidate",
        AIROp.PROVE: "proof_candidate",
    }
    return AIRVMEvent(
        kind=kind_by_op.get(instruction.op, instruction.op.value.lower()),
        payload=dict(instruction.args),
        evidence_id=instruction.evidence_id,
        mutates_l2_l3=False,
    )


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _statement_identifiers(stmt: AIRStatement) -> tuple[AIRIdentifier, ...]:
    if isinstance(stmt, FactStatement):
        return (stmt.subject, stmt.relation, stmt.object)
    if isinstance(stmt, QueryStatement):
        return (stmt.subject, stmt.query_type)
    if isinstance(stmt, NegateStatement):
        return (stmt.subject, stmt.relation, stmt.object)
    if isinstance(stmt, PolicyForbidStatement):
        return (stmt.target,)
    if isinstance(stmt, ProcedureStatement):
        return (stmt.name, stmt.ensures.name, *stmt.ensures.arguments)
    if isinstance(stmt, CallStatement):
        return (stmt.name,)
    if isinstance(stmt, ActionSimStatement):
        return (stmt.name,)
    if isinstance(stmt, ProveStatement):
        return (stmt.claim.name, *stmt.claim.arguments)
    return ()


def to_fact_frame(stmt: FactStatement):
    """Convert an AIR FACT candidate to a core FactFrame without committing it."""
    from htce_origin.kernel.core import FactFrame

    return FactFrame(
        subject=EntityId(stmt.subject.value),
        relation=RelationId(stmt.relation.value),
        object=EntityId(stmt.object.value),
        evidence=stmt.evidence,
    )
