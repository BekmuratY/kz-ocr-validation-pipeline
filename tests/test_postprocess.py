from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_plate import (
    clean_plate_text,
    detect_plate_format,
    is_region_code_valid,
    normalize_kz_plate,
    normalize_plate_text_verbose,
    pattern_valid_count,
)


def test_clean_strips_kz_prefix_suffix() -> None:
    assert clean_plate_text("KZ675BME05") == "675BME05"
    assert clean_plate_text("675BME05KZ") == "675BME05"


def test_normalize_common_ocr_confusions() -> None:
    assert normalize_kz_plate("32381P05") == "323BIP05"
    assert normalize_kz_plate("K760AKJ03") == "760AKJ03"


def test_pattern_valid_count_full_for_valid_plate() -> None:
    assert pattern_valid_count("760AKJ03") == 8
    assert pattern_valid_count("7604KJ03") < 8


def test_format_and_region_validation() -> None:
    assert detect_plate_format("760AKJ03") == "std_private"
    assert is_region_code_valid("760AKJ03")
    assert not is_region_code_valid("760AKJ99")


def test_verbose_normalization_steps_present() -> None:
    text, steps = normalize_plate_text_verbose(" KZ32381P05 ")
    assert text == "323BIP05"
    assert "drop_prefix_kz" in steps
