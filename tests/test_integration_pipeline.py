from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from scripts.batch_infer import BATCH_FIELDNAMES, inference_to_csv_row
from scripts.infer_plate import run_plate_inference


def _pick_gt_sample(
    labels_csv: Path,
    split: str,
    images_root: Path,
) -> tuple[Path, str] | None:
    if not labels_csv.exists():
        return None
    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            image = (r.get("image") or "").strip()
            if not image.startswith(f"images/{split}/"):
                continue
            gt = (r.get("plate_text") or "").strip()
            if not gt:
                continue
            filename = Path(image).name
            image_path = images_root / filename
            if image_path.exists():
                return image_path, gt
    return None


def test_inference_to_csv_row_contract() -> None:
    result = {
        "plate_text": "760AKJ03",
        "confidence": 0.91234,
        "plate_valid": True,
        "plate_format": "std_private",
        "plate_type": "NEW_STANDARD",
        "region_valid": True,
        "region_code": "03",
        "region_name": "Akmola Region",
        "region_scheme": "new",
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
    assert row["plate_type"] == "NEW_STANDARD"
    assert row["region_valid"] == "True"
    assert row["region_code"] == "03"
    assert row["region_name"] == "Akmola Region"
    assert row["region_scheme"] == "new"
    assert row["postprocess_score"] == "99"
    assert row["normalization_steps"] == "drop_non_alnum"
    assert row["status"] == "ok"


def test_inference_on_sample_image_if_enabled() -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Integration test disabled. Set RUN_INTEGRATION=1 to enable.")

    pytest.importorskip("paddleocr")
    pytest.importorskip("ultralytics")

    weights = Path("runs/detect/plate_kz/weights/best.pt")
    if not weights.exists():
        pytest.skip("Detector weights not found.")

    sample = _pick_gt_sample(
        labels_csv=Path("data/kz_plate/ocr_labels_template.csv"),
        split="test",
        images_root=Path("data/cars"),
    )
    if sample is None:
        pytest.skip("No GT sample found for split=test.")
    image_path, gt_text = sample

    result = run_plate_inference(
        image_path=image_path,
        detector_path=weights,
        conf=0.2,
        save_vis=None,
    )
    assert result["status"] == "ok"
    assert result["plate_text"] == gt_text


def test_inference_smoke_if_enabled() -> None:
    if os.getenv("RUN_SMOKE") != "1":
        pytest.skip("Smoke test disabled. Set RUN_SMOKE=1 to enable.")

    pytest.importorskip("paddleocr")
    pytest.importorskip("ultralytics")

    weights = Path("runs/detect/plate_kz/weights/best.pt")
    if not weights.exists():
        pytest.skip("Detector weights not found.")

    image_path = Path("examples/test_img_11.jpg")
    if not image_path.exists():
        pytest.skip("Sample image not found.")

    result = run_plate_inference(
        image_path=image_path,
        detector_path=weights,
        conf=0.2,
        save_vis=None,
    )
    assert result["status"] in {"ok", "no_plate"}
