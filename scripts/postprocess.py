from __future__ import annotations

import re

KZ_DIGIT_POS = {0, 1, 2, 6, 7}
TO_DIGIT = {
"O": "0",
"Q": "0",
"D": "0",
"I": "1",
"L": "1",
"Z": "2",
"S": "5",
"B": "8",
}
TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}

SUPPORTED_FORMATS: dict[str, re.Pattern[str]] = {
    "std_private": re.compile(r"^\d{3}[A-Z]{3}\d{2}$"),
    "std_prefixed": re.compile(r"^[A-Z]\d{3}[A-Z]{2}\d{2}$"),
    "trailer_like": re.compile(r"^\d{3}[A-Z]{2}\d{2}$"),
    "moto_like": re.compile(r"^\d{2}[A-Z]{2}\d{2}$"),
    # Old KZ format (pre-2012): one region letter + 3 digits + 2 letters.
    "old_private": re.compile(r"^[A-Z]\d{3}[A-Z]{2}$"),
    # Old KZ format (pre-2012) for individual cars: one region letter + 3 digits + 3 letters.
    "old_private_long": re.compile(r"^[A-Z]\d{3}[A-Z]{3}$"),
}

REGION_CODES_NEW: dict[str, str] = {
    "01": "Astana city",
    "02": "Almaty city",
    "03": "Akmola Region",
    "04": "Aktobe Region",
    "05": "Almaty Region",
    "06": "Atyrau Region",
    "07": "West Kazakhstan Region",
    "08": "Zhambyl Region",
    "09": "Karaganda Region",
    "10": "Kostanay Region",
    "11": "Kyzylorda Region",
    "12": "Mangistau Region",
    "13": "Turkestan Region",
    "14": "Pavlodar Region",
    "15": "North Kazakhstan Region",
    "16": "East Kazakhstan Region",
    "17": "Shymkent city",
    "18": "Abai Region",
    "19": "Zhetysu Region",
    "20": "Ulytau Region",
}

# Old letter codes used before the 2012 format change.
REGION_CODES_OLD: dict[str, str] = {
    "Z": "Astana city",
    "A": "Almaty city",
    "C": "Akmola Region",
    "W": "Akmola or Kostanay Region",
    "D": "Aktobe Region",
    "B": "Almaty Region or Zhetysu Region",
    "V": "Almaty Region or Zhetysu Region",
    "E": "Atyrau Region",
    "L": "West Kazakhstan Region",
    "H": "Zhambyl Region",
    "K": "Karaganda or Ulytau Region",
    "M": "Karaganda or Ulytau Region",
    "P": "Kostanay Region",
    "N": "Kyzylorda Region",
    "R": "Mangistau Region",
    "X": "Turkestan Region or Shymkent city",
    "S": "Pavlodar Region",
    "O": "North Kazakhstan or Akmola Region",
    "T": "North Kazakhstan Region",
    "F": "East Kazakhstan or Abai Region",
    "U": "East Kazakhstan or Abai Region",
}


def fit_kz_pattern(s: str) -> tuple[str, int, int]:
    chars = list(s)
    replacements = 0
    for i, ch in enumerate(chars):
        if i in KZ_DIGIT_POS:
            if ch.isdigit():
                continue
            sub = TO_DIGIT.get(ch, ch)
            if sub.isdigit():
                chars[i] = sub
                replacements += int(sub != ch)
        else:
            if ch.isalpha():
                continue
            sub = TO_LETTER.get(ch, ch)
            if sub.isalpha():
                chars[i] = sub
                replacements += int(sub != ch)
    out = "".join(chars)
    valid_after = sum(
        out[i].isdigit() if i in KZ_DIGIT_POS else out[i].isalpha()
        for i in range(min(8, len(out)))
    )
    return out, replacements, valid_after


def normalize_kz_plate(text: str) -> str:
    if not text:
        return text

    variants = {text}
    if text.startswith("K"):
        variants.add(text[1:])
    if text.endswith("K"):
        variants.add(text[:-1])

    candidates = set()
    for v in variants:
        if len(v) >= 8:
            for i in range(len(v) - 8 + 1):
                candidates.add(v[i : i + 8])
        else:
            candidates.add(v)

    best = text
    best_key = (-1, -10, -10)
    for cand in candidates:
        if len(cand) == 8:
            fitted, repl, valid = fit_kz_pattern(cand)
            key = (valid, -repl, int(fitted[0].isdigit()))
            if key > best_key:
                best_key = key
                best = fitted
        else:
            key = (0, 0, 0)
            if key > best_key:
                best_key = key
                best = cand
    return best


def clean_plate_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"[^A-ZА-Я0-9]", "", text)
    text = re.sub(r"^KZ", "", text)
    text = re.sub(r"KZ$", "", text)
    # Always drop leading 'K' (often comes from KZ prefix on the plate).
    if text.startswith("K"):
        text = text[1:]
    return normalize_kz_plate(text)


def pattern_valid_count(text: str) -> int:
    if len(text) != 8:
        return 0
    return sum(
        text[i].isdigit() if i in KZ_DIGIT_POS else text[i].isalpha()
        for i in range(8)
    )


def score_plate_text(text: str) -> tuple[int, int]:
    return (pattern_valid_count(text), -abs(len(text) - 8))


def detect_plate_format(text: str) -> str:
    for fmt_name, pattern in SUPPORTED_FORMATS.items():
        if pattern.fullmatch(text):
            return fmt_name
    return "unknown"


def is_region_code_valid(text: str) -> bool:
    code, name, scheme = get_region_info(text)
    if scheme == "new" and code is not None and name is not None:
        return True
    if scheme == "old" and code is not None and name is not None:
        return True
    return False


def get_region_info(text: str) -> tuple[str | None, str | None, str]:
    if len(text) >= 2 and text[-2:].isdigit():
        code = text[-2:]
        name = REGION_CODES_NEW.get(code)
        if name:
            return code, name, "new"
        return code, None, "new"
    if text and text[0].isalpha():
        code = text[0].upper()
        name = REGION_CODES_OLD.get(code)
        if name:
            return code, name, "old"
        return code, None, "old"
    return None, None, "unknown"


def postprocess_score(text: str) -> int:
    base = pattern_valid_count(text) * 10 - abs(len(text) - 8)
    fmt = detect_plate_format(text)
    if fmt != "unknown":
        base += 20
    if is_region_code_valid(text):
        base += 5
    return base


def normalize_plate_text_verbose(raw_text: str) -> tuple[str, list[str]]:
    steps: list[str] = []
    s = raw_text
    t1 = s.upper().strip()
    if t1 != s:
        steps.append("upper_strip")
    s = t1

    t2 = re.sub(r"[^A-ZА-Я0-9]", "", s)
    if t2 != s:
        steps.append("drop_non_alnum")
    s = t2

    t3 = re.sub(r"^KZ", "", s)
    if t3 != s:
        steps.append("drop_prefix_kz")
    s = t3

    t4 = re.sub(r"KZ$", "", s)
    if t4 != s:
        steps.append("drop_suffix_kz")
    s = t4

    t5 = normalize_kz_plate(s)
    if t5 != s:
        steps.append("normalize_kz_pattern")
    s = t5
    return s, steps
