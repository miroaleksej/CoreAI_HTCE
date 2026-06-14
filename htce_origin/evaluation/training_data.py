"""Training/data readiness contour for HTCE-Origin.

Scope boundary:
- Organizes user-supplied benchmark/training/research data for evaluation and
  curriculum planning.
- Does not mutate runtime L1/L2/L3 state.
- Does not train a model in the protected runtime.
- Does not expose hidden gold answers to engines; gold-like fields are reported
  as data-corpus metadata only and must be converted to commitments by P18
  before benchmark execution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from htce_origin.kernel.serialization import sha256_hex

_IGNORED_NAMES = {".DS_Store"}
_IGNORED_PREFIXES = ("._",)


@dataclass(frozen=True)
class TrainingDataAsset:
    relative_path: str
    category: str
    file_type: str
    size_bytes: int
    sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "file_type": self.file_type,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DatasetCompleteness:
    dataset_id: str
    present: int
    complete_for_claim: int
    file_count: int
    notes: str

    def as_payload(self) -> dict[str, object]:
        return {
            "complete_for_claim": self.complete_for_claim,
            "dataset_id": self.dataset_id,
            "file_count": self.file_count,
            "notes": self.notes,
            "present": self.present,
        }


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    purpose: str
    data_categories: tuple[str, ...]
    htce_modules: tuple[str, ...]
    safety_boundary: str
    ready: int

    def as_payload(self) -> dict[str, object]:
        return {
            "data_categories": list(self.data_categories),
            "htce_modules": list(self.htce_modules),
            "purpose": self.purpose,
            "ready": self.ready,
            "safety_boundary": self.safety_boundary,
            "stage_id": self.stage_id,
        }


@dataclass(frozen=True)
class TrainingDataReadinessReport:
    schema_version: str
    data_root: str
    asset_count: int
    total_size_bytes: int
    assets_by_category: Mapping[str, int]
    datasets: tuple[DatasetCompleteness, ...]
    curriculum: tuple[CurriculumStage, ...]
    manifest_entries_checked: int
    manifest_hash_mismatches: int
    ignored_metadata_files_removed_required: int
    license_review_required: int
    ready_for_htce_training_contour: int
    artifact_hash: str | None = None

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "asset_count": self.asset_count,
            "assets_by_category": dict(self.assets_by_category),
            "curriculum": [item.as_payload() for item in self.curriculum],
            "data_root": self.data_root,
            "datasets": [item.as_payload() for item in self.datasets],
            "ignored_metadata_files_removed_required": self.ignored_metadata_files_removed_required,
            "license_review_required": self.license_review_required,
            "manifest_entries_checked": self.manifest_entries_checked,
            "manifest_hash_mismatches": self.manifest_hash_mismatches,
            "ready_for_htce_training_contour": self.ready_for_htce_training_contour,
            "schema_version": self.schema_version,
            "total_size_bytes": self.total_size_bytes,
        }
        if self.artifact_hash is not None:
            payload["artifact_hash"] = self.artifact_hash
        return payload


def _is_ignored(path: Path) -> bool:
    if any(part == "__MACOSX" for part in path.parts):
        return True
    if path.name in _IGNORED_NAMES:
        return True
    if path.name.startswith(_IGNORED_PREFIXES):
        return True
    return False


def _category(rel: str) -> str:
    parts = Path(rel).parts
    if len(parts) >= 4 and parts[0] == "official_benchmarks" and parts[1] == "babi":
        return "official_babi_qa"
    if len(parts) >= 2 and parts[0] == "official_benchmarks" and parts[1] == "dialog_babi":
        return "dialog_babi_original"
    if len(parts) >= 2 and parts[0] == "official_benchmarks" and parts[1] == "dialog_babi_modified":
        return "dialog_babi_modified"
    if len(parts) >= 2 and parts[0] == "official_benchmarks" and parts[1] == "dialog_babi_permuted":
        return "dialog_babi_permuted"
    if "noisy" in rel and parts[0] == "official_benchmarks":
        return "noisy_dialogue"
    if parts and parts[0] == "external_pdf_corpus":
        return "research_pdf_corpus"
    if parts and parts[0] in {"sources", "research_sources"}:
        return "evidence_source_cards"
    if parts and parts[0] == "connector_fixtures":
        return "connector_fixtures"
    if parts and parts[0] == "official_benchmarks":
        return "official_benchmark_metadata"
    return "misc_data"


def scan_assets(data_root: str | Path) -> tuple[TrainingDataAsset, ...]:
    root = Path(data_root)
    assets: list[TrainingDataAsset] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored(path.relative_to(root)):
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower().lstrip(".") or "noext"
        assets.append(TrainingDataAsset(
            relative_path=rel,
            category=_category(rel),
            file_type=suffix,
            size_bytes=path.stat().st_size,
            sha256=sha256_hex(path.read_bytes()),
        ))
    return tuple(assets)


def _has_files(root: Path, pattern: str) -> int:
    return int(any(root.glob(pattern)))


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for item in root.glob(pattern) if item.is_file())


def _babi_complete(root: Path) -> DatasetCompleteness:
    babi = root / "official_benchmarks" / "babi"
    train_test_pairs = 0
    for task_id in range(1, 21):
        train = list(babi.rglob(f"qa{task_id}_*_train.txt"))
        test = list(babi.rglob(f"qa{task_id}_*_test.txt"))
        if train and test:
            train_test_pairs += 1
    return DatasetCompleteness(
        dataset_id="babi_qa_20",
        present=int(babi.exists()),
        complete_for_claim=int(train_test_pairs == 20),
        file_count=_count_files(babi, "**/*.txt") if babi.exists() else 0,
        notes=f"train/test task pairs present: {train_test_pairs}/20",
    )


def _dialog_complete(root: Path, name: str, expected_tasks: int) -> DatasetCompleteness:
    base = root / "official_benchmarks" / name
    files = list(base.glob("*.txt")) if base.exists() else []
    present_task_ids: set[int] = set()
    for file in files:
        text = file.name
        marker = "task"
        if marker in text:
            after = text.split(marker, 1)[1]
            digits = ""
            for char in after:
                if char.isdigit():
                    digits += char
                elif digits:
                    break
            if digits:
                present_task_ids.add(int(digits))
    complete = int(len(present_task_ids) >= expected_tasks and expected_tasks > 0)
    return DatasetCompleteness(
        dataset_id=name,
        present=int(base.exists() and len(files) > 0),
        complete_for_claim=complete,
        file_count=len(files),
        notes=f"task ids present: {sorted(present_task_ids)}; expected 1..{expected_tasks}",
    )


def _simple_dataset(root: Path, dataset_id: str, path: str, pattern: str, complete_note: str) -> DatasetCompleteness:
    base = root / path
    count = _count_files(base, pattern) if base.exists() else 0
    return DatasetCompleteness(dataset_id=dataset_id, present=int(count > 0), complete_for_claim=int(count > 0), file_count=count, notes=complete_note)


def verify_bundled_manifest(data_root: str | Path) -> tuple[int, int]:
    root = Path(data_root)
    manifest_path = root / "official_benchmarks" / "manifest.json"
    if not manifest_path.exists():
        return 0, 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    mismatches = 0
    for entry in manifest.get("entries", []):
        rel = str(entry.get("relative_path", ""))
        if rel.startswith("data/"):
            rel = rel[len("data/"):]
        expected = entry.get("sha256")
        path = root / rel
        if not path.exists() or not isinstance(expected, str):
            mismatches += 1
            continue
        checked += 1
        if sha256_hex(path.read_bytes()) != expected:
            mismatches += 1
    return checked, mismatches


def build_curriculum(datasets: Sequence[DatasetCompleteness]) -> tuple[CurriculumStage, ...]:
    status = {item.dataset_id: item for item in datasets}
    babi_ready = status.get("babi_qa_20", DatasetCompleteness("x", 0, 0, 0, "")).complete_for_claim
    modified_ready = status.get("dialog_babi_modified", DatasetCompleteness("x", 0, 0, 0, "")).present
    permuted_ready = status.get("dialog_babi_permuted", DatasetCompleteness("x", 0, 0, 0, "")).present
    pdf_ready = status.get("external_pdf_corpus", DatasetCompleteness("x", 0, 0, 0, "")).present
    noisy_ready = status.get("noisy_dialogue", DatasetCompleteness("x", 0, 0, 0, "")).present
    return (
        CurriculumStage("stage_00_integrity", "hash/manifest/data-root readiness", ("official_benchmark_metadata",), ("evaluation.training_data", "kernel.serialization"), "no runtime mutation", 1),
        CurriculumStage("stage_01_babi_reasoning", "external bAbI QA 1..20 for memory, deduction, induction and refusal boundaries", ("official_babi_qa",), ("evaluation.official_harness", "body.memory", "governance.proof"), "gold answers only through P18 commitments during evaluation", babi_ready),
        CurriculumStage("stage_02_dialog_robustness", "Dialog bAbI modified/permuted robustness and response variation", ("dialog_babi_modified", "dialog_babi_permuted"), ("evaluation.official_harness", "language.parser", "body.memory"), "commercial license review before redistribution", int(modified_ready and permuted_ready)),
        CurriculumStage("stage_03_noisy_dialogue", "noisy dialogue, correction/rejection and clarification stress", ("noisy_dialogue",), ("evaluation.no_leakage", "governance.evidence", "body.memory"), "not official; generated/curated stress data", noisy_ready),
        CurriculumStage("stage_04_evidence_grounding", "research PDF/source-card claim grounding and contradiction/retraction tests", ("research_pdf_corpus", "evidence_source_cards", "connector_fixtures"), ("governance.evidence", "evaluation.no_leakage", "connectors.fixtures"), "manual anchors are validation only, not injected truth", pdf_ready),
    )


def build_training_data_report(data_root: str | Path = "data") -> tuple[TrainingDataReadinessReport, tuple[TrainingDataAsset, ...]]:
    root = Path(data_root)
    assets = scan_assets(root)
    by_category: dict[str, int] = {}
    for asset in assets:
        by_category[asset.category] = by_category.get(asset.category, 0) + 1
    checked, mismatches = verify_bundled_manifest(root)
    datasets = (
        _babi_complete(root),
        _dialog_complete(root, "dialog_babi", 6),
        _dialog_complete(root, "dialog_babi_modified", 6),
        _dialog_complete(root, "dialog_babi_permuted", 6),
        _simple_dataset(root, "noisy_dialogue", "official_benchmarks", "*noisy*.jsonl", "bounded generated noisy dialogue corpus"),
        _simple_dataset(root, "external_pdf_corpus", "external_pdf_corpus/pdfs", "*.pdf", "open-access PDF corpus; validate licenses before commercial distribution"),
        _simple_dataset(root, "source_cards", "sources", "*.card", "source/evidence cards including weak/retracted cases"),
        _simple_dataset(root, "connector_fixtures", "connector_fixtures", "*", "connector replay fixtures for arXiv/PubMed/Semantic Scholar style payloads"),
    )
    curriculum = build_curriculum(datasets)
    ready = int(bool(assets) and mismatches == 0 and any(item.ready for item in curriculum))
    report = TrainingDataReadinessReport(
        schema_version="htce-training-data-readiness-v1",
        data_root=str(root),
        asset_count=len(assets),
        total_size_bytes=sum(asset.size_bytes for asset in assets),
        assets_by_category=by_category,
        datasets=datasets,
        curriculum=curriculum,
        manifest_entries_checked=checked,
        manifest_hash_mismatches=mismatches,
        ignored_metadata_files_removed_required=0,
        license_review_required=1,
        ready_for_htce_training_contour=ready,
        artifact_hash=None,
    )
    payload = report.as_payload()
    report = TrainingDataReadinessReport(**{**report.__dict__, "artifact_hash": sha256_hex(payload)})
    return report, assets
