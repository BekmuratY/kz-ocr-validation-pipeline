from __future__ import annotations

from scripts.batch_infer import BATCH_FIELDNAMES


def test_batch_csv_contract() -> None:
    assert BATCH_FIELDNAMES == [
        "image",
        "pred_text",
        "confidence",
        "plate_valid",
        "plate_format",
        "region_valid",
        "postprocess_score",
        "normalization_steps",
        "status",
        "visualization",
    ]
