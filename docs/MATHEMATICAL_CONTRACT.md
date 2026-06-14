# Mathematical Contract

Runtime state is defined over the Q256 torus:

`T_N^d = (Z / N Z)^d`, where `N = 2^256`.

Core invariants:

- Every protected state coordinate is an integer in `[0, N-1]`.
- State updates use modular wrap arithmetic.
- L1 sensory observations update L1 only.
- L2 clean working state is `raw_L2 - tag_accumulator mod N`.
- L3 semantic rules are provisional and cannot authorize answers directly.
- P13 decisions use raw integer functionals; normalized BP values are report-only.
- P19 hardware-width model uses explicit `uint256` wrap semantics.
- P20 stability uses deterministic integer closed-loop profiles and checkpoint/replay hashes.

No protected runtime float is permitted.


### P20.2 data boundary

The prepared training/benchmark corpus is an offline evaluation resource. It does not change the Q256 protected runtime contract. It does not bypass proof/evidence/policy gates. It must not be used to claim official benchmark completion unless the corresponding harness evaluates the files and writes traceable matrix evidence.
