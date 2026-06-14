.PHONY: clean gates sanity benchmarks data-readiness topology hardware stability artifacts verify compile test acceptance pipeline-tree

PYTHON ?= python
PYTEST ?= pytest
TIMEOUT ?= timeout

clean:
	@echo "Cleaning generated artifacts and bytecode..."
	rm -rf artifacts
	mkdir -p artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f *.log *.tmp

compile:
	$(PYTHON) -m compileall -q htce_origin scripts tests

pipeline-tree:
	@test -d scripts/00_gates
	@test -d scripts/01_sanity
	@test -f scripts/01_sanity/run_active_agent.py
	@test -f scripts/05_artifacts/run_acceptance.py
	@test -d scripts/02_benchmarks
	@test -d scripts/03_topology_and_hardware
	@test -d scripts/04_stability
	@test -d scripts/05_artifacts
	@test ! -f scripts/run_smoke.py
	@test ! -f scripts/run_closed_loop.py

# 00 — hard blockers. If these fail, the release is invalid.
gates:
	@echo "Running hard gates..."
	$(PYTHON) scripts/00_gates/scan_float_literals.py
	$(PYTHON) scripts/00_gates/check_version_sync.py
	$(PYTHON) scripts/00_gates/check_invariants.py

# 01 — quick live-system sanity.
sanity:
	@echo "Running organism sanity check..."
	$(PYTHON) scripts/01_sanity/run_organism.py
	$(PYTHON) scripts/01_sanity/run_active_agent.py

# 02 — reasoning and no-leakage benchmarks.
data-readiness:
	@echo "Preparing training/benchmark data readiness..."
	$(PYTHON) scripts/02_benchmarks/prepare_training_data.py

benchmarks: data-readiness
	@echo "Running benchmark contour..."
	$(PYTHON) -c "import gc; gc.collect()"
	$(TIMEOUT) 120 $(PYTHON) scripts/02_benchmarks/run_benchmark.py
	$(PYTHON) -c "import gc; gc.collect()"
	$(TIMEOUT) 180 $(PYTHON) scripts/02_benchmarks/run_official_harness.py --max-examples-per-task 15 --long-memory-events 10000 --closed-loop-steps 15
	$(PYTHON) -c "import gc; gc.collect()"
	$(TIMEOUT) 120 $(PYTHON) scripts/02_benchmarks/run_no_leakage.py

# 03 — offline topology and hardware-width verification.
topology:
	@echo "Running topology acceptance..."
	$(PYTHON) scripts/03_topology_and_hardware/run_topology_acceptance.py
	$(PYTHON) scripts/03_topology_and_hardware/generate_interaction_report.py

hardware:
	@echo "Running Q256 hardware-width verification..."
	$(PYTHON) scripts/03_topology_and_hardware/run_hardware_width.py

# 04 — long-run organism stability. Smoke is default to keep local acceptance bounded.
stability:
	@echo "Running long-run stability acceptance smoke..."
	$(PYTHON) scripts/04_stability/run_long_run_stability.py --smoke

# 05 — final buyer/reviewer artifacts and manifest verification.
artifacts:
	@echo "Exporting final buyer/reviewer artifacts..."
	$(PYTHON) scripts/05_artifacts/export_artifacts.py

verify:
	@echo "Verifying protected trace and release manifest..."
	$(PYTHON) scripts/05_artifacts/verify_trace.py
	$(PYTHON) scripts/05_artifacts/verify_manifest.py

test:
	@echo "Running v1.0 current test suite file-by-file..."
	@for test_file in tests/test_1000_step_closed_loop.py tests/test_acceptance_release.py tests/test_air_language.py tests/test_air_vm.py tests/test_architecture_docs.py tests/test_associative_cortex.py tests/test_babi_minimal.py tests/test_betti_calibration.py tests/test_fact_delta_memory.py tests/test_homeostasis.py tests/test_interfaces_import.py tests/test_invariants_contract.py tests/test_l1_l2_l3.py tests/test_l1_sensory_q256.py tests/test_l3_semantic_q256.py tests/test_latest_state.py tests/test_p11_closed_loop.py tests/test_p12_release_hardening.py tests/test_p13_no_normalization_decision_core.py tests/test_p15_topology_acceptance.py tests/test_p16_l3_rule_promotion.py tests/test_p17_official_benchmark_harness.py tests/test_p18_no_leakage_dynamic_benchmark.py tests/test_p19_hardware_width_verification.py tests/test_p20_1_pipeline_organization.py tests/test_p20_2_training_data_readiness.py tests/test_p20_long_run_stability.py tests/test_p21_honest_intelligence_closure.py tests/test_p24_living_active_agent.py tests/test_p25_unified_living_dialog_simulation.py tests/test_p26_adaptive_policy_improvement.py tests/test_p27_continual_adaptive_memory.py tests/test_p28_multitask_adaptive_memory.py tests/test_proof_gate.py tests/test_q16_core.py tests/test_runtime_lifecycle.py tests/test_runtime_no_legacy_imports.py tests/test_sleep_consolidation.py tests/test_topology_guard.py tests/test_trace_snapshot.py tests/test_unknown_refusal.py tests/test_v1_clean_system_release.py tests/test_world_model.py; do \
		echo "[v1.0 test] $$test_file"; \
		$(TIMEOUT) 180 $(PYTEST) -q $$test_file || exit 1; \
	done

acceptance:
	$(PYTHON) scripts/05_artifacts/run_acceptance.py
	@echo ""
	@echo "=========================================================="
	@echo " HTCE-Origin v1.0 final_math Q256 clean system acceptance PASS"
	@echo " System verified through one bounded scripts/* pipeline: P25-P28 unified runtime, v1.0 external-shaped revalidation, no answer leakage, no false support, no real actions."
	@echo " Artifacts: artifacts/"
	@echo "=========================================================="
