from pathlib import Path

from htce_origin.evaluation.training_data import build_training_data_report


def test_training_data_readiness_report_for_bundled_data():
    report, assets = build_training_data_report("data")
    assert report.ready_for_htce_training_contour == 1
    assert report.manifest_hash_mismatches == 0
    assert report.asset_count == len(assets)
    assert report.assets_by_category.get("official_babi_qa", 0) >= 40
    datasets = {item.dataset_id: item for item in report.datasets}
    assert datasets["babi_qa_20"].complete_for_claim == 1
    assert datasets["external_pdf_corpus"].present == 1
    assert datasets["source_cards"].present == 1


def test_training_data_contains_no_macos_metadata():
    root = Path("data")
    assert root.exists()
    bad = [item for item in root.rglob("*") if item.name == ".DS_Store" or item.name.startswith("._") or "__MACOSX" in item.parts]
    assert bad == []
