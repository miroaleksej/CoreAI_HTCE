# HTCE-Origin v1.0 Acceptance Report

v1.0 acceptance is the clean final_math Q256 release line.  It updates prior P-step tests to the current architecture and keeps only checks that still validate the shipped system.

Required acceptance stages:

1. compile all runtime, scripts and tests;
2. scan protected runtime source for float literals;
3. verify version sync across `__init__`, `pyproject.toml`, `capabilities.json` and `RELEASE_MANIFEST.json`;
4. verify 16 invariants;
5. run organism and active-agent sanity;
6. run P25/P26/P27/P28 inside one runtime family;
7. run v1.0 external-shaped revalidation;
8. run benchmark/no-leakage/topology/hardware-width/stability smoke;
9. export buyer/reviewer artifacts;
10. verify trace and manifest.

The release boundary remains simulation-only and real actions are disabled.
