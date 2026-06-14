#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from htce_origin.evaluation.benchmarks import MinimalBABIHarness

report = MinimalBABIHarness().run_all()
print(report.summary())
for result in report.results:
    print(f"{result.task}/{result.name}: {'PASS' if result.passed else 'FAIL'} decision={result.decision.value} trace={result.trace_id}")
