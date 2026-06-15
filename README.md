<img width="1447" height="1087" alt="9a2927df-694b-4fe9-af63-c317eefe3039" src="https://github.com/user-attachments/assets/e8d3c161-6366-4b01-9548-7f063bcece04" />

<p align="center">
  ![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b.svg)
  ![Acceptance](https://img.shields.io/badge/Invariants-19/19%20PASS-brightgreen)
</p>

# HTCE-Origin v1.0

> **Q256 integer-only toroidal cognitive runtime with L1/L2/L3 state, proof-gated memory, bounded active-agent simulation, continual adaptive memory, and no-cross-domain regression probes.**

<p align="center">
  <b>HTCE-Origin v1.0 final_math Q256 clean</b><br/>
  A clean, single-runtime, simulation-safe cognitive architecture for evidence-bound reasoning and adaptive memory.
</p>

<p align="center">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.0_final_math_q256_clean-blue"/>
  <img alt="Runtime" src="https://img.shields.io/badge/runtime-single_HTCERuntime-green"/>
  <img alt="Arithmetic" src="https://img.shields.io/badge/arithmetic-Q256_integer_only-purple"/>
  <img alt="Actions" src="https://img.shields.io/badge/actions-simulation_only-orange"/>
  <img alt="Claims" src="https://img.shields.io/badge/AGI_claim-no-red"/>
</p>

---

## Table of Contents

1. [What HTCE-Origin Is](#what-htce-origin-is)
2. [What HTCE-Origin Is Not](#what-htce-origin-is-not)
3. [Why This System Exists](#why-this-system-exists)
4. [Core Design Principle](#core-design-principle)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Architecture Overview](#architecture-overview)
7. [L1/L2/L3 Cognitive Stack](#l1l2l3-cognitive-stack)
8. [AIR: The Bounded Runtime Language](#air-the-bounded-runtime-language)
9. [Fact-as-Delta Memory](#fact-as-delta-memory)
10. [Proof, Evidence, Policy, and Topology Gates](#proof-evidence-policy-and-topology-gates)
11. [Living Simulation Loop](#living-simulation-loop)
12. [Adaptive Memory and Continual Learning](#adaptive-memory-and-continual-learning)
13. [No-Cross-Domain Regression](#no-cross-domain-regression)
14. [Protected Trace and Release Integrity](#protected-trace-and-release-integrity)
15. [What the v1.0 Release Proves](#what-the-v10-release-proves)
16. [How HTCE-Origin Differs from Common AI Stacks](#how-htce-origin-differs-from-common-ai-stacks)
17. [Repository Layout](#repository-layout)
18. [Installation](#installation)
19. [Quick Start](#quick-start)
20. [Acceptance and Verification](#acceptance-and-verification)
21. [Example Runtime Interactions](#example-runtime-interactions)
22. [Generated Artifacts](#generated-artifacts)
23. [Claim Boundary](#claim-boundary)
24. [Development Philosophy](#development-philosophy)
25. [Roadmap After v1.0](#roadmap-after-v10)
26. [License and Commercial Use](#license-and-commercial-use)

---

## What HTCE-Origin Is

**HTCE-Origin** is a clean v1.0 build of a **Q256 integer-only toroidal cognitive runtime**. It is designed as a bounded, auditable, simulation-safe cognitive system where memory, reasoning, dialog state, action policy, homeostasis, topology checks, and adaptive learning all run through one runtime object:

```python
HTCERuntime
```

The v1.0 build is not a set of disconnected demos. The same runtime carries:

- natural-language-to-AIR intake for bounded statements and dialog turns;
- L1/L2/L3 toroidal state transitions;
- fact-as-delta memory;
- latest-state, supersession, and contradiction quarantine;
- proof, evidence, policy, and topology gates;
- active-agent grid-world simulation;
- dialog slot memory and simulated API-call policy;
- sleep/L3 consolidation;
- continual adaptive memory;
- multi-task no-regression probes;
- protected trace and cryptographic release manifest.

The correct short description is:

> **HTCE-Origin v1.0 is a bounded, evidence-gated, torus-native cognitive runtime for simulation-only adaptive agency and auditable memory.**

---

## What HTCE-Origin Is Not

This repository intentionally does **not** claim:

- Artificial General Intelligence;
- consciousness;
- qualia;
- biological life;
- real-world robotic autonomy;
- board-measured hardware performance;
- unrestricted open-world natural language understanding;
- replacement of large language models for broad text generation.

All real-world action is disabled in this clean release:

```text
allow_real_actions = false
simulation_only    = true
```

The system is allowed to act only inside bounded simulation paths unless a future audited release explicitly changes that boundary and passes a separate safety case.

---

## Why This System Exists

Most AI applications today are built around one of the following patterns:

1. a large language model prompt;
2. a retrieval-augmented generation pipeline;
3. a tool-using agent wrapper;
4. a neural policy model;
5. a symbolic rules engine;
6. a hybrid orchestration layer.

These systems can be useful, but they often lack **built-in runtime invariants** for:

- refusing unsupported answers;
- preserving evidence provenance;
- separating active facts from superseded facts;
- quarantining contradictions;
- preventing answer-key leakage during evaluation;
- proving that learning in one domain did not damage another domain;
- enforcing simulation-only action boundaries;
- keeping memory updates auditable through a protected trace.

HTCE-Origin addresses this by treating cognition as a bounded state transition system over an integer toroidal phase space.

Instead of treating memory as a text buffer or hidden vector alone, the runtime treats events, facts, rules, sensor packets, goals, and dialog slots as phase deltas and gated state transitions.

---

## Core Design Principle

The core design rule is:

> **No answer or simulated action is allowed to bypass the runtime gates.**

Every meaningful transition must pass through some combination of:

```text
Input / NLU / AIR
→ Policy gate
→ Evidence gate
→ Proof layer
→ L1/L2/L3 state transition
→ Topology guard
→ Protected trace
→ RuntimeResponse
```

This makes the system slower and narrower than unconstrained text generation, but it gives the runtime a very different property: it is optimized for **bounded correctness**, **traceability**, and **non-hallucinating refusal**, not unconstrained fluency.

---

## Mathematical Foundation

### Q256 Modular State Space

The runtime uses an integer modulus:

$$
N = 2^{256}
$$

A toroidal state vector is represented as:

$$
\mathbf{x} \in (\mathbb{Z}/N\mathbb{Z})^d
$$

A transition is a modular update:

$$
\mathbf{x}_{t+1} = (\mathbf{x}_t + \Delta_t) \bmod N
$$

where:

- $\mathbf{x}_t$ is the current toroidal state;
- $\Delta_t$ is an integer phase delta derived from observation, fact, rule, or simulated action;
- $N$ is the Q256 modulus;
- $d$ is the fixed state dimension.

The decision path is integer-only. Human-readable percentages or basis points may appear in reports, but they are report-layer summaries, not floating-point decision variables.

### Toroidal Distance

For one coordinate, the circular distance is:

$$
\operatorname{dist}_N(a,b) = \min\left(|a-b|,\; N - |a-b|\right)
$$

For a vector state:

$$
D_N(\mathbf{x},\mathbf{y}) = \sum_{i=1}^{d} \operatorname{dist}_N(x_i,y_i)
$$

This gives the runtime a native topology-aware notion of change: large modular jumps can be detected as anomalous even if the raw representation wraps around.

### Fact-as-Delta Memory

A fact is not treated as plain text. A supported fact produces a canonical delta:

$$
F = (s, r, o, e)
$$

where:

- $s$ is the subject;
- $r$ is the relation;
- $o$ is the object/value;
- $e$ is the evidence identifier.

The fact key is usually relation-specific:

$$
K = H(s,r)
$$

The fact payload is hashed into a bounded integer delta:

$$
\Delta_F = H(s,r,o,e) \bmod N
$$

The memory status is one of:

```text
ACTIVE
SUPERSEDED
QUARANTINED
UNKNOWN
```

This supports latest-state reasoning while keeping old history available for trace and audit.

### Supersession

When a new fact shares the same key $K = H(s,r)$ with a previous active fact, the old active fact becomes superseded:

$$
F_{old}.status \leftarrow \texttt{SUPERSEDED}
$$

and the new fact becomes active:

$$
F_{new}.status \leftarrow \texttt{ACTIVE}
$$

This is used for locations, dialog slot corrections, and other latest-state updates.

Example:

```text
FACT Mary located_in office EVID ev1
FACT Mary located_in garden EVID ev2
QUERY Mary location EVID q1
```

The runtime should answer:

```text
ANSWER: garden
```

while preserving the previous `office` state as superseded evidence.

### Contradiction Quarantine

A contradiction is not treated as a normal update. If a fact and its negation cannot be reconciled, the target key is quarantined:

$$
F.status \leftarrow \texttt{QUARANTINED}
$$

A query against a quarantined target must not produce a normal answer:

$$
\texttt{QUARANTINED} \Rightarrow \texttt{REFUSE}
$$

This is one of the core anti-hallucination principles of the runtime.

### Protected Trace Hash Chain

The trace is a hash-linked sequence of events:

$$
h_t = H(h_{t-1} \parallel \operatorname{canonical\_json}(event_t))
$$

where:

- $h_t$ is the current trace head;
- $h_{t-1}$ is the previous trace head;
- $event_t$ is the canonical event payload;
- $H$ is SHA-256 in the release artifacts.

Any change to the event sequence changes the final trace head.

### Active Inference Surrogate

The living simulation uses an integer expected-free-energy-like cost. In this release it is a bounded surrogate, not a claim of biological free energy minimization.

A simplified form is:

$$
G(a) = C_{goal}(a) + C_{risk}(a) + C_{uncertainty}(a) + C_{energy}(a)
$$

The selected simulated action is:

$$
a^* = \arg\min_{a \in \mathcal{A}_{sim}} G(a)
$$

where all costs are integer raw values. No real action is authorized by this process.

### No-Regression Condition

For continual adaptive memory, an episode cost can be represented as:

$$
C(E_i) = S(E_i) + Q(E_i) + R(E_i)
$$

where:

- $S(E_i)$ is the number of simulated steps;
- $Q(E_i)$ is the number of clarification turns;
- $R(E_i)$ is the number of recovery or risk events.

For a repeated target, improvement is verified when:

$$
C(E_{i+1}) \leq C(E_i)
$$

or, for a strict improvement claim:

$$
C(E_{i+1}) < C(E_i)
$$

For multi-domain learning, each domain $d$ has a probe cost:

$$
P_d(t) \in \mathbb{Z}_{\geq 0}
$$

No cross-domain regression means:

$$
\forall d,\; P_d(t+1) \leq \max_{\tau \leq t} P_d(\tau)
$$

In the v1.0 clean revalidation artifact, the domain cost histories remain bounded at zero for the built-in P28 domains.

---

## Architecture Overview

At a high level:

```text
User / Benchmark / Simulation Input
        │
        ▼
NLU → AIR bridge / AIR parser
        │
        ▼
Policy gate ── Evidence gate ── Proof layer
        │
        ▼
HTCERuntime
        │
        ├── L1 sensory torus
        ├── L2 episodic/fact memory
        ├── L3 semantic/provisional rule cortex
        ├── Q256 world model
        ├── active-inference surrogate planner
        ├── homeostasis state
        ├── topology guard
        ├── snapshot/export path
        └── protected trace
        │
        ▼
RuntimeResponse: ANSWER / ASK_CLARIFICATION / REFUSE / ACT_SIMULATED / BLOCK
```

The most important architectural rule is that the system should not have multiple independent behavioral shells. Dialog, grid-world, active-agent behavior, proof-gated action, and continual adaptation are all routed through the same runtime line.

---

## L1/L2/L3 Cognitive Stack

### L1 — Sensory / Active State Intake

L1 is the fast toroidal sensory layer. In v1.0, it is used primarily for bounded simulation input and deterministic sensor packets.

L1 handles:

- grid-world observations;
- local agent position;
- heartbeat updates;
- deterministic integer sensor encoding;
- active-agent loop state.

It does not claim full real-world vision/audio/proprioception. Those require future audited encoders.

### L2 — Episodic Fact-as-Delta Memory

L2 stores active facts, superseded facts, and quarantined contradictions.

It supports:

- latest-state query;
- object carried-by relation;
- location chaining;
- dialog slot memory;
- correction through supersession;
- contradiction quarantine;
- evidence-linked fact records.

Examples:

```text
FACT current_dialog_restaurant_1 has_slot_value cuisine=italian EVID d1
FACT current_dialog_restaurant_1 has_slot_value price=cheap EVID d2
QUERY current_dialog_restaurant_1 api_call_ready EVID q1
```

If location is missing, the runtime must ask for clarification rather than fabricate a location.

### L3 — Semantic Cortex and Provisional Rules

L3 stores provisional hints and rules produced by sleep/consolidation.

It is intentionally restricted:

- L3 can propose a hypothesis;
- L3 can help select a path;
- L3 can store adaptive hints;
- L3 cannot bypass proof gates;
- L3 cannot directly authorize unsupported factual answers.

A valid runtime response must still pass through proof/evidence/policy checks.

---

## AIR: The Bounded Runtime Language

AIR is the controlled interface language used by the runtime. It prevents arbitrary text from directly becoming authority.

Typical forms:

```text
FACT <subject> <relation> <object> EVID <evidence_id>
QUERY <subject> <query_type> EVID <evidence_id>
NEGATE <subject> <relation> <object> EVID <evidence_id>
PROC <procedure_name> ENSURES <condition> EVID <evidence_id>
```

Examples:

```text
FACT mary located_in office EVID ev1
QUERY mary location EVID q1
```

Expected response:

```text
ANSWER: office
```

Natural language can be bridged into AIR for supported bounded forms:

```text
Mary went to the office.
Where is Mary?
```

Internally this becomes a controlled fact/query sequence.

Unsupported or ambiguous input should lead to:

```text
ASK_CLARIFICATION
```

not hallucinated completion.

---

## Fact-as-Delta Memory

The memory layer is one of the most important parts of HTCE-Origin.

### Latest-State Example

Input:

```text
Mary went to the office.
Mary went to the garden.
Where is Mary?
```

Expected output:

```text
ANSWER: garden
```

The old state is not erased. It becomes superseded.

### Object-Carrying Example

Input:

```text
Mary picked up the football.
Mary went to the garden.
Where is the football?
```

Expected output:

```text
ANSWER: garden
```

The answer is derived from a proof chain:

```text
carried_by(football, mary)
located_in(mary, garden)
--------------------------------
located_in(football, garden)
```

This is different from a parser shortcut. The runtime must preserve the chain.

### Dialog Slot Example

Input:

```text
I want Italian food in Rome.
Actually, make it Chinese.
Make it cheap.
Book a table.
```

Expected simulated action:

```text
api_call domain=restaurant cuisine=chinese location=rome price=cheap
```

The correction is represented as supersession:

```text
cuisine=italian  → SUPERSEDED
cuisine=chinese  → ACTIVE
```

---

## Proof, Evidence, Policy, and Topology Gates

HTCE-Origin uses multiple gates because no single check is sufficient.

### Evidence Gate

Evidence is attached to accepted facts and proof paths. A fact without evidence cannot become an authoritative answer.

### Proof Layer

The proof layer decides whether a statement can be supported from active memory, rules, and allowed derivation patterns.

For a statement:

$$
\sigma = r(s,o)
$$

proof authorization can be described as:

$$
\operatorname{Authorize}(\sigma) =
\begin{cases}
\texttt{ANSWER}, & \text{if proof valid and policy allows} \\
\texttt{ASK\_CLARIFICATION}, & \text{if required data missing} \\
\texttt{REFUSE}, & \text{if contradictory or unsafe}
\end{cases}
$$

### Policy Gate

Policy separates four cases:

```text
ANSWER              supported response
ASK_CLARIFICATION   missing information
REFUSE              contradiction / unsupported claim / policy block
ACT_SIMULATED       proof-gated simulation-only action
```

Missing information is not treated as a hard refusal. It is a normal dialog step.

### Topology Guard

The topology guard checks whether state transitions remain within expected structural boundaries.

It tracks integer topological diagnostics such as Betti-style indicators and anomaly scores. A topology anomaly can block or quarantine a path.

### Real-Action Boundary

Even if a simulated action is authorized, real action remains blocked in v1.0:

```text
ACT_SIMULATED != ACT_REAL
```

---

## Living Simulation Loop

HTCE-Origin v1.0 contains a bounded active-agent simulation. It is not a claim of biological life or consciousness. It is an engineering loop that gives the runtime an internal perception-action cycle.

The loop is:

```text
heartbeat
→ L1 observation
→ Q256 world-model prediction
→ integer action scoring
→ simulated action
→ prediction error
→ homeostasis update
→ trace append
```

The agent can move in a small simulated grid-world, update homeostatic variables, and select actions through integer cost minimization.

Homeostasis tracks variables such as:

```text
energy_bp
risk_bp
uncertainty_bp
novelty_bp
sleep_pressure_bp
integrity_bp
```

These are not subjective feelings. They are integer control variables used for simulation and safety.

---

## Adaptive Memory and Continual Learning

The system includes a bounded sleep/consolidation mechanism.

The high-level process is:

```text
episode
→ trace and L2 facts
→ sleep/consolidation
→ L3 provisional hints
→ next episode
→ improvement / no-regression probes
```

In P26/P27 development, the system demonstrated that repeated episodes can reduce cost or hold optimal cost while preserving previous knowledge.

A simplified improvement condition is:

$$
C(E_2) < C(E_1)
$$

or, after convergence:

$$
C(E_{i+1}) = C(E_i)
$$

without regression.

---

## No-Cross-Domain Regression

The v1.0 line includes multi-domain probes over:

```text
grid_nav
dialog_slots
babi_reasoning
contradiction
```

The purpose is to test catastrophic interference. The runtime should not improve or update one domain while damaging another.

A probe matrix is run after consolidation cycles:

```text
train/update grid_nav
→ probe grid_nav/dialog_slots/babi_reasoning/contradiction

train/update dialog_slots
→ probe grid_nav/dialog_slots/babi_reasoning/contradiction

train/update babi_reasoning
→ probe grid_nav/dialog_slots/babi_reasoning/contradiction

train/update contradiction
→ probe grid_nav/dialog_slots/babi_reasoning/contradiction
```

The release artifact reports zero-cost histories for the built-in P28 stress domains.

---

## Protected Trace and Release Integrity

### Protected Trace

Every meaningful runtime event is appended to a hash-chain trace. The trace can be verified after execution.

Typical trace events include:

- runtime wake;
- fact commit;
- query;
- proof authorization;
- policy decision;
- simulated action;
- topology check;
- sleep/consolidation;
- artifact export;
- manifest verification.

### Release Manifest

The release contains:

```text
RELEASE_MANIFEST.json
HASHES.txt
```

These files provide SHA-256 hashes for source and release-critical files.

The intended rule is simple:

> If a critical file changes, the manifest must change, and verification must be rerun.

---

## What the v1.0 Release Proves

The clean v1.0 package reports the following verified properties in its artifacts and acceptance stages:

```text
compileall: PASS
float_literals_in_htce_origin: 0
version_sync: PASS
invariants: PASS 16/16
release_manifest: PASS
trace_verify: PASS
topology_acceptance: PASS
hardware_width_verification: PASS
long_run_stability_smoke: PASS
benchmark: PASS 7/7
official_harness_smoke: PASS
no_leakage: PASS
v1_clean_system_revalidation: PASS
```

The v1.0 clean-system revalidation reports:

```text
external_rows_passed: 15 / 15
external_false_support_count: 0
answer_key_visible_to_engine_count: 0
dialog_loader_strict_passed: true
no_external_regression: true
proof_gates_passed: true
topology_gates_passed: true
trace_verified: true
real_actions_allowed: false
simulation_only: true
```

This does not prove general intelligence. It proves a bounded set of runtime properties:

1. integer-only protected runtime path;
2. single-runtime integration;
3. proof-gated answers;
4. simulation-only actions;
5. no answer-key leakage in the v1.0 external-shaped test;
6. dialog/action slot correction;
7. contradiction quarantine;
8. adaptive simulation loop;
9. continual no-regression probes;
10. release integrity via manifest and trace.

---

## How HTCE-Origin Differs from Common AI Stacks

The following comparison is deliberately narrow and engineering-focused. It does not claim that HTCE-Origin is more fluent, larger, or more generally capable than modern language models.

| Capability | Typical LLM Prompt | Typical RAG | Agent Wrapper | HTCE-Origin v1.0 |
|---|---:|---:|---:|---:|
| Built-in evidence gate | No | Partial | Usually external | Yes |
| Proof-gated answer path | No | Rare | Usually external | Yes |
| Contradiction quarantine | No | Rare | Usually custom | Yes |
| Latest-state supersession | Prompt-dependent | Index-dependent | Custom | Yes |
| No-answer-leakage benchmark discipline | External | External | External | Built into release tests |
| Simulation-only action boundary | No | No | Custom | Yes |
| Protected trace hash-chain | No | Usually no | Custom | Yes |
| Integer-only decision core | No | No | No | Yes |
| Toroidal L1/L2/L3 state | No | No | No | Yes |
| Continual no-regression probes | No | No | Rare | Yes |
| Cross-domain regression matrix | No | No | Rare | Yes |
| Claim boundary encoded in artifacts | No | No | Rare | Yes |

The key difference is not that HTCE-Origin replaces an LLM. The key difference is that HTCE-Origin makes several safety and verification properties part of the runtime contract rather than optional wrapper behavior.

### What Competitors Usually Do Not Provide by Default

Most competing systems do not provide all of the following as a single built-in runtime assembly:

```text
Q256 integer-only state
+ toroidal L1/L2/L3 runtime
+ fact-as-delta memory
+ active/superseded/quarantined fact status
+ proof/evidence/policy gate stack
+ topology guard
+ protected trace hash-chain
+ simulation-only action boundary
+ no-leakage benchmark discipline
+ continual no-regression probes
+ cross-domain regression matrix
```

This is the strongest technical value proposition of HTCE-Origin v1.0.

---

## Repository Layout

Representative structure:

```text
HTCE-Origin/
├── htce_origin/
│   ├── body/
│   │   ├── runtime.py          # HTCERuntime, lifecycle, living/dialog/adaptive/multitask loops
│   │   ├── layers.py           # L1/L2/L3 body state
│   │   └── memory.py           # fact-as-delta memory
│   ├── cognition/
│   │   ├── cortex.py           # associative cortex / L3 provisional hints
│   │   ├── learning.py         # consolidation support
│   │   ├── l3_promotion.py     # L3 rule promotion
│   │   └── world.py            # Q256 world model
│   ├── control/
│   │   ├── homeostasis.py      # homeostatic integer state
│   │   └── planner.py          # simulation-only planning
│   ├── evaluation/
│   │   ├── benchmarks.py       # benchmark rows/reports
│   │   ├── official_harness.py # bounded official-style harness
│   │   ├── no_leakage.py       # answer-key leakage checks
│   │   ├── long_run_stability.py
│   │   └── training_data.py
│   ├── governance/
│   │   ├── evidence.py         # evidence/protected trace
│   │   ├── policy.py           # decision policy
│   │   ├── proof.py            # proof and authorization
│   │   └── snapshot.py
│   ├── kernel/
│   │   ├── core.py
│   │   ├── q16.py              # Q16/Q256 compatibility layer
│   │   ├── uint256.py          # bounded integer utilities
│   │   └── serialization.py
│   ├── language/
│   │   ├── air.py              # AIR language
│   │   ├── parser.py
│   │   └── nlu_air_bridge.py   # bounded NLU → AIR bridge
│   ├── sensory/
│   │   └── l1_encoder.py
│   └── topology/
│       ├── betti.py
│       ├── guard.py
│       └── acceptance.py
├── scripts/
│   ├── 00_gates/
│   ├── 01_sanity/
│   ├── 02_benchmarks/
│   ├── 03_topology/
│   ├── 04_hardware/
│   ├── 05_artifacts/
│   └── 06_verify/
├── tests/
├── artifacts/
├── docs/
├── RELEASE_MANIFEST.json
├── HASHES.txt
├── Makefile
└── pyproject.toml
```

---

## Installation

### Requirements

The clean v1.0 package is intentionally lightweight.

```text
Python >= 3.10
No required runtime dependencies in pyproject.toml
```

### Clone

```bash
git clone <your-repository-url> htce-origin
cd htce-origin
```

### Optional Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install for Development

```bash
pip install -e .
```

If you want to run tests:

```bash
pip install pytest
```

---

## Quick Start

### 1. Run Acceptance

```bash
make acceptance
```

This runs the bounded release pipeline.

### 2. Run Version Sync

```bash
python scripts/00_gates/check_version_sync.py
```

### 3. Scan for Float Literals

```bash
python scripts/00_gates/scan_float_literals.py
```

Expected release property:

```text
float_literals_in_htce_origin: 0
```

### 4. Run Invariants

```bash
python scripts/00_gates/check_invariants.py
```

Expected:

```text
16/16 invariants PASS
```

### 5. Run v1 Clean Revalidation

Depending on the script layout in your release:

```bash
python scripts/01_sanity/run_active_agent.py
```

or run the v1-specific artifact export/verification path:

```bash
python scripts/05_artifacts/export_artifacts.py
python scripts/06_verify/verify_trace.py
python scripts/06_verify/verify_manifest.py
```

---

## Acceptance and Verification

The v1.0 acceptance path is organized in stages:

```text
00_gates      hard release gates
01_sanity     organism/runtime sanity
02_benchmarks bounded benchmark and no-leakage checks
03_topology   topology acceptance
04_hardware   arithmetic hardware-width verification
05_artifacts  report export
06_verify     trace and manifest verification
```

Recommended commands:

```bash
make test
make acceptance
```

If a legacy test fails after an architectural update, do not simply delete it. First check whether it encodes:

1. a still-valid invariant;
2. an obsolete version string;
3. an old artifact schema;
4. a deprecated acceptance path;
5. a real regression.

Only update tests when the system has genuinely moved forward and the old test checks a stale contract.

---

## Example Runtime Interactions

### Python API

```python
from htce_origin import HTCERuntime, RuntimeConfig, RuntimeRequest

runtime = HTCERuntime(
    config=RuntimeConfig(
        allow_real_actions=False,
        allow_legacy_imports=False,
    )
)

runtime.wake()

response = runtime.tick(RuntimeRequest(
    "FACT mary located_in office EVID ev1",
    source="example"
))
print(response.output)

response = runtime.tick(RuntimeRequest(
    "QUERY mary location EVID q1",
    source="example"
))
print(response.output)
```

Expected:

```text
COMMIT: mary located_in office
ANSWER: office
```

### Natural Language Bridge

```python
runtime.tick(RuntimeRequest("Mary went to the office.", source="example"))
response = runtime.tick(RuntimeRequest("Where is Mary?", source="example"))
print(response.output)
```

Expected:

```text
ANSWER: office
```

### Dialog Slot Correction

```python
runtime.tick(RuntimeRequest("I want Italian food in Rome.", source="dialog"))
runtime.tick(RuntimeRequest("Actually, make it Chinese.", source="dialog"))
runtime.tick(RuntimeRequest("Make it cheap.", source="dialog"))
response = runtime.tick(RuntimeRequest("Book a table.", source="dialog"))
print(response.output)
```

Expected simulated action:

```text
api_call domain=restaurant cuisine=chinese location=rome price=cheap
```

### Missing Slot Clarification

```python
runtime.tick(RuntimeRequest("I want Italian food in Rome.", source="dialog"))
response = runtime.tick(RuntimeRequest("Book a table.", source="dialog"))
print(response.output)
```

Expected:

```text
ASK_CLARIFICATION: missing required dialog slots: price
```

### Contradiction Quarantine

```python
runtime.tick(RuntimeRequest("FACT mary located_in office EVID ev1", source="test"))
runtime.tick(RuntimeRequest("NEGATE mary located_in office EVID ev2", source="test"))
response = runtime.tick(RuntimeRequest("QUERY mary location EVID q1", source="test"))
print(response.output)
```

Expected behavior:

```text
REFUSE
```

The exact message may vary, but it must not be a normal unsupported answer.

---

## Generated Artifacts

The v1.0 release includes artifacts such as:

```text
artifacts/v1_clean_system_revalidation_report.json
artifacts/p25_unified_living_dialog_simulation_report.json
artifacts/p26_adaptive_policy_improvement_report.json
artifacts/p27_continual_adaptive_memory_report.json
artifacts/p28_continual_multitask_adaptive_memory_report.json
artifacts/topology_acceptance_summary.json
artifacts/hardware_width_verification_report.json
artifacts/long_run_stability_report.json
artifacts/closed_loop_trace_export.json
```

These artifacts are intended to support audit and technical review.

Important fields to inspect:

```text
trace_verified
false_support_count
answer_key_visible_to_engine_count
real_actions_allowed
simulation_only
proof_gates_passed
topology_gates_passed
no_cross_domain_regression
```

---

## Claim Boundary

### Claimed in v1.0

The v1.0 clean build claims only the following bounded properties:

1. Q256 integer-only protected runtime path;
2. no float literals in `htce_origin` release code;
3. single `HTCERuntime` integration line;
4. L1/L2/L3 toroidal state model;
5. fact-as-delta memory with active/superseded/quarantined status;
6. proof/evidence/policy-gated answers;
7. topology guard acceptance artifacts;
8. simulation-only action boundary;
9. protected trace verification;
10. clean release manifest;
11. bounded living active-agent simulation;
12. adaptive improvement across repeated episodes;
13. continual memory no-regression probes;
14. multi-domain no-cross-domain-regression stress;
15. external-shaped revalidation with zero answer-key visibility to engine in the built-in test.

### Not Claimed in v1.0

The v1.0 clean build does not claim:

```text
AGI
consciousness
qualia
biological life
unbounded language understanding
safe deployment in real robotics
board-measured energy performance
formal proof of correctness for every possible input
complete replacement for LLMs
```

### Recommended Public Wording

Use:

> HTCE-Origin is a bounded, auditable, Q256 integer-only toroidal cognitive runtime with proof-gated memory and simulation-only adaptive agency.

Do not use:

> HTCE-Origin is conscious.

Do not use:

> HTCE-Origin is AGI.

Do not use:

> HTCE-Origin is safe for real-world autonomous control without further audit.

---

## Development Philosophy

HTCE-Origin follows these engineering rules:

### 1. One Runtime, Not Many Shells

New behavior should be integrated into the existing runtime path when possible:

```text
HTCERuntime.tick(...)
HTCERuntime.run_living_active_agent_simulation(...)
HTCERuntime.run_continual_adaptive_memory_simulation(...)
HTCERuntime.run_continual_multitask_simulation(...)
```

Avoid creating disconnected demonstration wrappers that bypass gates.

### 2. No Unsupported Answer

If the system cannot prove or support a response, it should ask for clarification or refuse.

### 3. Simulation Before Reality

All action paths in v1.0 are simulation-only. Real-world actuation requires a separate audited release.

### 4. Trace Everything Important

If an event matters for audit, it should appear in the protected trace or release artifacts.

### 5. Prefer Updating Existing Modules

Do not create unnecessary parallel files. Improve the canonical runtime modules unless a new file is justified by clear separation of concern.

### 6. Tests Track the Current Contract

Old tests should not fossilize obsolete behavior. If the system advances, tests should be updated to the new contract while preserving the underlying safety invariant.

---

## Roadmap After v1.0

Suggested next steps:

### v1.1 — External Benchmark Expansion

- larger bAbI subset;
- repaired Dialog-bAbI task coverage;
- MultiWOZ-style slot stress, if carefully bounded;
- larger no-leakage report;
- regression matrix across external rows.

### v1.2 — Stronger Language Boundary

- broader but still controlled NLU-to-AIR bridge;
- explicit ambiguity detection;
- multilingual bounded commands;
- better entity normalization;
- no hidden answer leakage.

### v1.3 — Hardware Synthesis Preparation

- isolate synthesizable integer kernels;
- prepare RTL mapping notes;
- produce hardware claim boundary;
- run FPGA/ASIC synthesis separately from runtime claims;
- never report board-measured energy until physically measured.

### v1.4 — Simulation Environment Expansion

- richer grid-world;
- dynamic hazards;
- multi-agent symbolic environment;
- longer continual-learning episodes;
- stronger no-regression probes.

### v1.5 — External Audit Package

- minimal reproducible release;
- independent verifier script;
- frozen benchmark seed set;
- signed release manifest;
- reviewer guide.

---

## License and Commercial Use

This repository may be distributed for technical review and evaluation according to the license selected by the repository owner.

If this package is intended for commercial transfer, pilot deployment, or investor review, include:

- the exact release ZIP;
- `RELEASE_MANIFEST.json`;
- `HASHES.txt`;
- `artifacts/` reports;
- claim boundary document;
- acceptance command and environment notes;
- known limitations.

Commercial use should be governed by a separate agreement if the repository is not released under a permissive open-source license.

---

## Final Summary

HTCE-Origin v1.0 is not a chatbot and not an AGI claim. It is a clean, bounded, auditable cognitive runtime built around integer toroidal state, proof-gated memory, protected trace, simulation-only agency, and continual no-regression validation.

Its central contribution is not raw fluency. Its central contribution is runtime discipline:

```text
bounded input
→ integer toroidal state
→ proof/evidence/policy gates
→ topology-aware memory
→ simulation-only action
→ protected trace
→ no-regression verification
```

That is the system’s core identity.
