# HTCE-Origin v1.0 — Clean Q256 Unified Cognitive Runtime

HTCE-Origin v1.0 is the clean final_math Q256 build of the toroidal cognitive runtime.  It is a single bounded simulation system, not a collection of disconnected demo shells.  The same `HTCERuntime` carries AIR/NLU input, L1/L2/L3 toroidal state, fact-as-delta memory, proof/evidence/policy gates, topology checks, active-agent grid simulation, dialog/action policy, adaptive sleep/L3 consolidation, continual memory and multitask regression probes.

## What v1.0 proves

- Q256 integer-only protected runtime path: no float literals in `htce_origin`.
- L1/L2/L3 state is updated through one runtime lifecycle.
- L2 fact memory supports supersession, latest-state answers and contradiction quarantine.
- L3 stores provisional hints learned through sleep/consolidation; it does not bypass proof gates.
- Dialog slot memory and simulated `api_call` are proof-gated; missing slots ask clarification.
- P25/P26/P27/P28 behaviours run inside one simulation line: living loop, adaptive improvement, continual memory and multitask no-regression.
- v1.0 external-shaped revalidation checks bAbI-style rows, Dialog-bAbI `USR|/SYS|` rows, contradiction rows and multitask stress without answer-key leakage.
- Protected trace and release manifest provide auditable evidence.

## Run

```bash
make acceptance
```

The bounded acceptance pipeline runs compile, version sync, invariants, organism sanity, active-agent/adaptive/multitask/v1 revalidation, benchmarks, topology, hardware-width arithmetic model, stability smoke, artifact export, trace verification and manifest verification.

## Claim boundary

This package does not claim AGI, consciousness, qualia, biological life, real-world autonomy or board-measured hardware performance.  All actions are simulation-only unless a future audited build explicitly changes `allow_real_actions` and passes a new safety case.
