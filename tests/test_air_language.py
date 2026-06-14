import pytest

from htce_origin.language.air import (
    AIRLexer,
    AIRParser,
    AIRSecurityError,
    AIRStaticChecker,
    AIRSyntaxError,
    FactStatement,
    QueryStatement,
    NegateStatement,
    ProcedureStatement,
    CallStatement,
    check_air,
)


def test_lexer_tokenizes_required_fact_program():
    tokens = AIRLexer().tokenize("FACT Mary located_in office EVID event42")
    values = [tok.value for tok in tokens]
    assert values[:6] == ["FACT", "Mary", "located_in", "office", "EVID", "event42"]


def test_parser_accepts_required_minimal_programs():
    src = "\n".join([
        "FACT Mary located_in office EVID event42",
        "QUERY Mary location EVID event42",
        "NEGATE Mary located_in office EVID correction43",
        "PROC move_demo ENSURES location(robot, table)",
        "CALL move_demo",
    ])
    program = AIRParser().parse(src)
    assert len(program.statements) == 5
    assert isinstance(program.statements[0], FactStatement)
    assert isinstance(program.statements[1], QueryStatement)
    assert isinstance(program.statements[2], NegateStatement)
    assert isinstance(program.statements[3], ProcedureStatement)
    assert isinstance(program.statements[4], CallStatement)
    assert program.source_hash
    result = AIRStaticChecker().check(program)
    assert result.ok, result.reasons


def test_missing_evidence_is_blocked_by_parser_or_checker():
    with pytest.raises(AIRSyntaxError):
        AIRParser().parse("FACT Mary located_in office")
    program = AIRParser().parse("FACT Mary located_in office EVID none")
    result = AIRStaticChecker().check(program)
    assert not result.ok
    assert "AIR_MISSING_EVIDENCE" in result.error_codes


def test_malicious_json_payload_is_blocked_before_parse():
    with pytest.raises(AIRSecurityError):
        AIRParser().parse('{"op":"FACT","subject":"Mary"}')


def test_plain_llm_text_is_not_air_and_cannot_mutate_state():
    with pytest.raises(AIRSyntaxError):
        AIRParser().parse("Mary is in office")
    with pytest.raises(AIRSyntaxError):
        AIRParser().parse("COMMIT L2 Mary located_in office")


def test_direct_l2_l3_mutation_vocabulary_is_rejected_by_checker():
    program = AIRParser().parse("FACT L2 commit office EVID event42")
    result = check_air("FACT L2 commit office EVID event42")
    assert not result.ok
    assert "AIR_DIRECT_STATE_MUTATION_BLOCKED" in result.error_codes
