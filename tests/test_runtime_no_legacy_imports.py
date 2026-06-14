from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "htce_origin"


def test_runtime_does_not_import_legacy():
    offenders = []
    for path in RUNTIME.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("legacy"):
                        offenders.append((path.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("legacy"):
                    offenders.append((path.name, node.module))
    assert not offenders, offenders
