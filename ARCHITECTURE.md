# HTCE-Origin Architecture

HTCE-Origin is organized as a Q256 toroidal cognitive runtime:

1. L1 sensory torus: exact integer driver-sample quantization and deterministic ternary projection.
2. L2 separated working torus: episode-tagged active state with clean-state unbinding and same-key residual reversal.
3. L3 semantic cortex: integer phase relaxation and provisional rule promotion without authority leak.
4. World model: Q256 simulated transition predictor with raw integer EFE-style scoring.
5. Planner/control: simulation-only habitat gate; real actions are blocked.
6. Governance: evidence/proof/policy gates and protected hash-chain trace.
7. Acceptance: invariants, topology, benchmarks, no-leakage protocol, hardware-width arithmetic verification, and long-run stability.

Dataflow:

`sensor/task input -> L1/Q256 phase -> gated L2 or simulated world path -> proof/evidence/policy gates -> answer/refusal/simulated plan -> protected trace`

P20 adds long-run acceptance, not a new reasoning organ. The runtime core remains unchanged: P20 drives deterministic simulated sensory/world/planner paths and verifies trace/restore/replay stability.


## Organ coverage matrix

The current full-stack organ list remains present for release compatibility:

- Q16 kernel / Q256 compatibility naming
- Core types
- AIR language
- Runtime lifecycle
- L1/L2/L3 body
- Fact memory
- Associative cortex
- World model
- Planner
- Homeostasis
- Proof layer
- Policy gates
- Evidence trace
- Topology guard
- Betti backend
- Parser adapter
- API surface
- Snapshot
- Benchmarks
- Serialization
- Errors
- Config

| Organ | Input | Output | Transition | Evidence | Failure mode |
|---|---|---|---|---|---|
| L1/L2/L3 body | integer phase/event | toroidal state | modular Q256 update | protected trace | refusal/block |
| World model | state/action | predicted state | simulated Q256 transition | trace/report | no fact authority |
| Benchmark/stability | task/profile | matrix/report | acceptance run | JSON artifact | fail closed |


## Pipeline acceptance organization

The release is operated through a staged `scripts/` pipeline rather than a flat script namespace. Hard gates run before sanity checks, benchmarks, topology/hardware verification, long-run stability, artifact export, and manifest/trace verification. This prevents obsolete scripts from entering the buyer/reviewer acceptance path and keeps P20 long-run stability as a current stage of the release pipeline.


### Training/data readiness contour

The `data/` tree is an offline evaluation and curriculum resource. It is connected to `htce_origin.evaluation.training_data`, P17 official harness and P18 no-leakage protocol. It is explicitly not a runtime memory store and cannot authorize facts, answers or real actions.
## P21 Honest Intelligence Closure

P21 fixes release integrity, honest external benchmark execution, deterministic NLU->AIR intake, bAbI reasoning closure targets, and L3 hypothesis/proof usage without adding a new runtime organ. External bAbI/Dialog rows must run through HTCERuntime; gold answers are used only after response for scoring. L3 provisional rules may seed hypotheses/proof paths but never direct answers, L2 facts, or real actions.

Current status: Q256 integer-only runtime, raw decision core, no-leakage benchmark protocol, topology/hardware/stability/data-readiness contours, and clean pipeline acceptance are preserved.

## P24 — Dialog/Action Policy Closure Without New Architecture

P22 does not add a new organ. It hardens the existing language bridge, proof layer and runtime query path so that simple discourse markers and coreference enter AIR deterministically, object location is derived by proof-chain rather than direct parser shortcut, and induction is emitted as `HYPOTHESIS` unless sufficient proof authorizes an answer. Dialog bAbI loading now supports both numbered and `USR|`/`SYS|` formats with task-specific selection and gold-after-response scoring.


## P25 — Unified Living/Dialog Simulation Without Multiple Shells

P25 does not introduce a separate dialog manager, slot tracker or benchmark shell. The bounded restaurant/hotel dialog turns are processed by `HTCERuntime.tick` during the same active-agent heartbeat that updates L1, homeostasis and the Q256 world-model self-state. Domain isolation is represented by ordinary L2 subject keys such as `current_dialog_restaurant_1` and `current_dialog_hotel_1`; correction remains latest-state supersession.


## P27 continual adaptive memory without regression

Release 0.1.0-p27-continual-adaptive-memory-q256-final_math adds a bounded simulation-only continual adaptive memory check inside one HTCERuntime loop. It runs repeated living/dialog episodes, sleep consolidation, retained L3 provisional hints, bAbI/dialog/proof/topology probes, and verifies no regression with false_support_count=0. No real action or consciousness claim is introduced.

## P28 continual multi-task adaptation without cross-domain regression

Release 0.1.0-p28-multitask-adaptive-memory-q256-final_math adds a bounded multi-task stress test inside the same HTCERuntime. The curriculum alternates grid navigation, dialog slot/action policy, bAbI-style reasoning and contradiction quarantine, then runs a cross-domain probe matrix after every sleep/consolidation cycle. Regression is defined as an integer probe-cost increase over the domain's best historical cost. The report does not claim AGI or real autonomy; it proves a simulation-only no-cross-domain-regression property.

## v1.0 clean-system fixation

v1.0 keeps the architecture as a single `HTCERuntime` assembly.  The runtime path remains: Input -> Parser adapter/NLU/AIR language -> Policy gates -> Proof layer -> Evidence trace -> L1/L2/L3 body -> Fact memory -> Associative cortex -> World model -> Planner -> Homeostasis -> Topology guard -> Serialization/Snapshot/API surface.

| Organ | Input | Output | Transition | Evidence | Failure mode |
|---|---|---|---|---|---|
| Q16 kernel / Q256 kernel | integer phases | bounded torus deltas | modular arithmetic | invariant report | release gate fail |
| Core types | facts/entities/relations | fact deltas | canonical state hash | protected trace | serialization error |
| AIR language | checked commands | VM payloads | static gate | AIR trace | refuse |
| Runtime lifecycle | RuntimeRequest | RuntimeResponse | wake/tick/sleep/export | trace hash | blocked decision |
| L1/L2/L3 body | sensory/facts/rules | layered state | toroidal transition | layer digest | topology gate |
| Fact memory | claims/evidence | active/superseded/quarantined records | fact-as-delta | proof/evidence ids | quarantine/refuse |
| Associative cortex | provisional hints | L3 candidates | sleep consolidation | L3 rule report | hypothesis only |
| World model | state/action | prediction/error | Q256 transition | prediction trace | high surprise |
| Planner | simulated skills | simulated action | proof-guided selection | action trace | block real action |
| Homeostasis | risk/energy/uncertainty | viability state | integer update | heartbeat trace | ask/refuse |
| Proof layer | statement/evidence | judgment | theorem check | proof id | no authorization |
| Policy gates | request/support | decision | allow/ask/refuse/block | policy trace | block |
| Evidence trace | events | hash-chain | append/verify | trace head | verify fail |
| Topology guard | L2/L3 state | pass/block | Betti/anomaly check | topology report | quarantine/block |
| Betti backend | simplicial samples | beta0/beta1 | integer topology | acceptance artifact | topology fail |
| Parser adapter | text/AIR | normalized command | NLU->AIR | bridge trace | clarification/refuse |
| API surface | operator call | bounded endpoint | whitelist | API contract | legacy reject |
| Snapshot | runtime state | export/restore | canonical JSON | snapshot hash | mismatch |
| Benchmarks | rows/probes | matrix reports | post-inference scoring | artifact JSON | fail gate |
| Serialization | payload | canonical bytes | JSON-safe conversion | sha256 | float rejected |
| Errors | invalid state | bounded exception | fail-closed | trace/error | release fail |
| Config | runtime flags | safety profile | validate | config snapshot | unsafe config reject |
