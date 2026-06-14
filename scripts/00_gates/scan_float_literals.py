#!/usr/bin/env python3
"""Release-gate scan: protected runtime source must not contain float literals."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCAN_ROOTS = (ROOT / "htce_origin",)

failures: list[tuple[str, int, str]] = []
for root in SCAN_ROOTS:
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                failures.append((str(path.relative_to(ROOT)), int(getattr(node, "lineno", 0)), repr(node.value)))

if failures:
    for relpath, lineno, value in failures:
        print(f"FLOAT_LITERAL {relpath}:{lineno} {value}")
    raise SystemExit(1)
print("float_literals_in_htce_origin: 0")
