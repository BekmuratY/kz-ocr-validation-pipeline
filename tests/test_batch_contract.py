from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_infer import BATCH_FIELDNAMES


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
