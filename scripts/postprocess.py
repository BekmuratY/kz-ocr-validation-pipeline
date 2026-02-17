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
    if len(text) < 2:
        return False
    tail = text[-2:]
    if not tail.isdigit():
        return False
    code = int(tail)
    return 1 <= code <= 20


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
