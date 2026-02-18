from __future__ import annotations

from pathlib import Path

from scripts.batch_infer import BATCH_FIELDNAMES, inference_to_csv_row


def test_inference_to_csv_row_contract() -> None:
    result = {
        "plate_text": "760AKJ03",
        "confidence": 0.91234,
        "plate_valid": True,
        "plate_format": "std_private",
        "region_valid": True,
        "postprocess_score": 99,
        "normalization_steps": ["drop_non_alnum"],
        "status": "ok",
    }
    row = inference_to_csv_row("sample.jpg", result, Path("outputs/sample.jpg"))
    assert list(row.keys()) == BATCH_FIELDNAMES
    assert row["image"] == "sample.jpg"
    assert row["pred_text"] == "760AKJ03"
    assert row["confidence"] == "0.9123"
    assert row["plate_valid"] == "True"
    assert row["plate_format"] == "std_private"
    assert row["region_valid"] == "True"
    assert row["postprocess_score"] == "99"
    assert row["normalization_steps"] == "drop_non_alnum"
    assert row["status"] == "ok"
