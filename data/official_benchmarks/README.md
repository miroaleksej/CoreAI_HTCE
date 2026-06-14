# HTCE-Core v2.3 — Independent Benchmark Track

This directory is the offline root for **official / independent benchmark datasets**.

The repository does not bundle the Facebook bAbI or Dialog bAbI datasets. To avoid
false claims, the validation harness records `official_dataset_present=0` until the
user places the original dataset files here and reruns:

```bash
python3 scripts/run_independent_benchmarks.py   --root .   --official-root data/official_benchmarks   --out runs/validation/independent_benchmark_evidence.json
```

## Expected layout

```text
data/official_benchmarks/
  babi/
    qa1_single-supporting-fact_train.txt
    qa1_single-supporting-fact_test.txt
    qa15_basic-deduction_train.txt
    qa15_basic-deduction_test.txt
    qa16_basic-induction_train.txt
    qa16_basic-induction_test.txt
  dialog_babi/
    dialog-babi-task*.txt
```

The adapter is intentionally conservative:

- absent official files produce `SKIPPED_EXTERNAL_DATASET_MISSING`, not PASS;
- synthetic/file-backed internal datasets do not count as official benchmark completion;
- all discovered official files are hashed and listed in provenance evidence;
- no `official benchmark passed` claim is emitted unless the files are present and evaluated.
