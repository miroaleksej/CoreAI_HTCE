# v1.0 Proof of Capability

## Verified behaviour

HTCE-Origin v1.0 demonstrates a bounded capability set that common LLM/chatbot stacks do not provide as a native invariant:

1. **Integer-only protected decision path.** Runtime state and protected serialization reject floats and operate over Q256 integer/toroidal structures.
2. **Proof-gated answers.** Supported answers carry proof/evidence diagnostics; unsupported or quarantined facts do not become confident answers.
3. **Contradiction quarantine.** A fact and its negation trigger refusal until repaired rather than silent overwrite or hallucinated resolution.
4. **Supersession with trace.** Dialog corrections replace active slot values while preserving the historical chain.
5. **Action proof gate.** Simulated `api_call` is emitted only after required slots are active and non-quarantined.
6. **One simulation line.** Grid world, L1 sensory intake, homeostasis, dialog policy, L2 memory, L3 consolidation and protected trace run in one `HTCERuntime`.
7. **Adaptive improvement.** Repeated same-goal episodes reduce or preserve raw integer cost after sleep/L3 consolidation.
8. **Continual memory without regression.** Repeated episodes retain prior L3 hints, proof gates and dialog/bAbI probes.
9. **Multitask no-regression.** Grid, dialog slots, bAbI-style reasoning and contradiction probes are interleaved without cross-domain cost growth.
10. **No-answer-leakage external-shaped validation.** v1.0 loads Dialog-bAbI-style `USR|/SYS|` rows and checks expected answers only after runtime inference.

## What competitors usually lack natively

Typical LLM/RAG agents can emulate parts of this behaviour with prompts, tools and validators, but they do not usually expose a deterministic Q256 state kernel, proof-gated fact memory, contradiction quarantine, topology guard and cryptographic trace as one mandatory runtime contract.

## What is not proven

The release does not prove AGI, consciousness, real robotics autonomy, broad natural-language competence, official benchmark leadership or physical-board hardware measurements.
