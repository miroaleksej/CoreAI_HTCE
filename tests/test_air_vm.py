import pytest

from htce_origin.language.air import (
    AIRBytecodeCompiler,
    AIRParser,
    AIRPolicyError,
    AIRStaticChecker,
    AIRTypeError,
    AIRVM,
    AIROp,
    BytecodeInstruction,
    execute_air,
)


def test_bytecode_compiler_emits_expected_ops_without_mutation():
    program = AIRParser().parse("\n".join([
        "FACT Mary located_in office EVID event42",
        "QUERY Mary location EVID event42",
        "NEGATE Mary located_in office EVID correction43",
        "PROC move_demo ENSURES location(robot, table)",
        "CALL move_demo",
    ]))
    bytecode = AIRBytecodeCompiler().compile(program)
    assert [item.op for item in bytecode] == [AIROp.FACT, AIROp.QUERY, AIROp.NEGATE, AIROp.PROC, AIROp.CALL]
    assert all(item.mutates_l2_l3 is False for item in bytecode)


def test_vm_emits_candidate_events_only():
    events = execute_air("\n".join([
        "FACT Mary located_in office EVID event42",
        "PROC move_demo ENSURES location(robot, table)",
        "CALL move_demo",
    ]))
    assert [event.kind for event in events] == ["fact_candidate", "procedure_registered", "procedure_call_requested"]
    assert all(event.mutates_l2_l3 is False for event in events)
    assert events[0].evidence_id == "event42"


def test_forbidden_action_is_blocked_before_vm_execution():
    program = AIRParser().parse("\n".join([
        "POLICY FORBID move_demo EVID policy_event",
        "PROC move_demo ENSURES location(robot, table)",
        "CALL move_demo",
    ]))
    result = AIRStaticChecker().check(program)
    assert not result.ok
    assert "AIR_FORBIDDEN_ACTION" in result.error_codes
    with pytest.raises(AIRTypeError):
        AIRBytecodeCompiler().compile(program)


def test_unknown_call_is_blocked():
    program = AIRParser().parse("CALL move_demo")
    result = AIRStaticChecker().check(program)
    assert not result.ok
    assert "AIR_UNKNOWN_PROCEDURE" in result.error_codes


def test_vm_refuses_mutating_bytecode_even_if_constructed_manually():
    with pytest.raises(Exception) as excinfo:
        AIRVM().execute((BytecodeInstruction(AIROp.FACT, {"subject": "Mary"}, "event42", mutates_l2_l3=True),))
    assert "mutation" in str(excinfo.value).lower()


def test_action_sim_forbidden_target_blocked():
    program = AIRParser().parse("\n".join([
        "POLICY FORBID move_arm EVID policy_event",
        "ACTION_SIM move_arm EVID event99",
    ]))
    result = AIRStaticChecker().check(program)
    assert not result.ok
    assert "AIR_FORBIDDEN_ACTION" in result.error_codes
