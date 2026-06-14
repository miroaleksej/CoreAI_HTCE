from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "ARCHITECTURE.md",
    "MATHEMATICAL_CONTRACT.md",
    "CLAIM_BOUNDARY.md",
    "FILE_TREE.md",
]

REQUIRED_ORGANS = [
    "Q16 kernel", "Core types", "AIR language", "Runtime lifecycle",
    "L1/L2/L3 body", "Fact memory", "Associative cortex", "World model",
    "Planner", "Homeostasis", "Proof layer", "Policy gates",
    "Evidence trace", "Topology guard", "Betti backend", "Parser adapter",
    "API surface", "Snapshot", "Benchmarks", "Serialization", "Errors", "Config",
]

REQUIRED_COLUMNS = ["Input", "Output", "Transition", "Evidence", "Failure mode"]


def test_required_docs_exist():
    for doc in REQUIRED_DOCS:
        assert (ROOT / doc).exists(), doc


def test_architecture_lists_all_organs_and_contract_columns():
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for organ in REQUIRED_ORGANS:
        assert organ in text
    for column in REQUIRED_COLUMNS:
        assert column in text
