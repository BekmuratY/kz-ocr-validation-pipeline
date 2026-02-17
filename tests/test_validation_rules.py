from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_plate import (
    detect_plate_format,
    normalize_plate_text_verbose,
    postprocess_score,
    score_plate_text,
)


def test_detect_plate_format_variants() -> None:
    assert detect_plate_format("123ABC10") == "std_private"
    assert detect_plate_format("A123BC10") == "std_prefixed"
    assert detect_plate_format("123AB10") == "trailer_like"
    assert detect_plate_format("12AB10") == "moto_like"
    assert detect_plate_format("BADFORMAT") == "unknown"


def test_postprocess_score_prefers_valid_pattern() -> None:
    good = postprocess_score("323BIP05")
    bad = postprocess_score("32381P05")
    assert good > bad


def test_score_plate_text_prefers_expected_length() -> None:
    exact = score_plate_text("760AKJ03")
    short = score_plate_text("760AKJ0")
    assert exact > short


def test_normalize_verbose_reports_steps() -> None:
    text, steps = normalize_plate_text_verbose("..kz-760akj03..")
    assert text == "760AKJ03"
    assert "upper_strip" in steps
    assert "drop_non_alnum" in steps
    assert "drop_prefix_kz" in steps
