# AIR v1 Source Language Specification

AIR is the typed source-language boundary between an untrusted parser/LLM adapter and the HTCE runtime. AIR programs are not Python, not natural language, and not direct memory writes.

## Minimal grammar

```text
FACT <subject> <relation> <object> EVID <evidence_id>
QUERY <subject> <query_type> EVID <evidence_id>
NEGATE <subject> <relation> <object> EVID <evidence_id>
PROC <name> ENSURES <predicate>(<arg1>, <arg2>)
CALL <name>
```

## Trust boundary

- Parser/LLM output is untrusted.
- AIR parser/checker/compiler/VM emit candidate events only.
- L2/L3 mutation is allowed only through the gated runtime body.
- Missing evidence, malformed syntax, forbidden action, malicious JSON, or unsupported claim produces fail-closed rejection.

## VM boundary

The AIR VM has no direct commit/update operation. It emits typed candidate events consumed by policy/evidence/proof/runtime gates.

## Non-goals

AIR is not a general programming language, not a Python replacement, not a truth source, and not an actuator.
