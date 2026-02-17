#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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
    # Most common format in this dataset.
    "std_private": re.compile(r"^\d{3}[A-Z]{3}\d{2}$"),
    # Additional patterns (for broader validation reporting).
    "std_prefixed": re.compile(r"^[A-Z]\d{3}[A-Z]{2}\d{2}$"),
    "trailer_like": re.compile(r"^\d{3}[A-Z]{2}\d{2}$"),
    "moto_like": re.compile(r"^\d{2}[A-Z]{2}\d{2}$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and read license plate text from one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file.")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("runs/plate_kz/weights/best.pt"),
        help="Path to trained detector weights.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["paddle", "tesseract"],
        default="paddle",
        help="OCR engine.",
    )
    parser.add_argument(
        "--tesseract-lang",
        type=str,
        default="eng",
        help="Tesseract language code, if backend=tesseract.",
    )
    parser.add_argument(
        "--save-vis",
        type=Path,
        default=Path("outputs/result.jpg"),
        help="Where to save visualization image.",
    )
    return parser.parse_args()


def resolve_detector_path(detector: Path) -> Path:
    if detector.exists():
        return detector

    candidates = sorted(
        Path("runs").rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"[warn] Detector not found at {detector}, using latest: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(
        f"Detector weights not found: {detector}. Also no best.pt found under runs/."
    )


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
    # Regional code range used in current project validation policy.
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


def preprocess_crop_tesseract(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def preprocess_crop_paddle(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = max(1.0, 64.0 / max(1, min(h, w)))
    resized = cv2.resize(
        crop,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
    return cv2.bilateralFilter(resized, 7, 50, 50)


def preprocess_crop_paddle_clahe(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = max(1.0, 72.0 / max(1, min(h, w)))
    resized = cv2.resize(
        crop,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_crop_paddle_sharp(crop: np.ndarray) -> np.ndarray:
    base = preprocess_crop_paddle(crop)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(base, -1, kernel)


def extract_text_from_paddle_result(result: object) -> str:
    if not result:
        return ""

    first = result[0]
    if isinstance(first, dict):
        rec_texts = first.get("rec_texts")
        if isinstance(rec_texts, list) and rec_texts:
            return "".join(str(x) for x in rec_texts)

    chunks: list[str] = []
    for item in result[0] if isinstance(result, list) else []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            txt_info = item[1]
            if isinstance(txt_info, (list, tuple)) and txt_info:
                chunks.append(str(txt_info[0]))
            elif isinstance(txt_info, str):
                chunks.append(txt_info)
        elif isinstance(item, dict):
            txt = item.get("rec_text") or item.get("text")
            if txt:
                chunks.append(str(txt))
    return "".join(chunks)


@lru_cache(maxsize=1)
def get_paddle_ocr():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    from paddleocr import PaddleOCR

    common_kwargs: dict[str, object] = {"lang": "en"}
    det_dir = os.getenv("PADDLEOCR_TEXT_DET_MODEL_DIR")
    rec_dir = os.getenv("PADDLEOCR_TEXT_REC_MODEL_DIR")
    if det_dir:
        common_kwargs["text_detection_model_dir"] = det_dir
    if rec_dir:
        common_kwargs["text_recognition_model_dir"] = rec_dir

    try:
        return PaddleOCR(
            **common_kwargs,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except (TypeError, ValueError):
        return PaddleOCR(use_angle_cls=True, show_log=False, **common_kwargs)


@lru_cache(maxsize=2)
def get_detector(model_path: str) -> YOLO:
    return YOLO(model_path)


def ocr_paddle(image: np.ndarray) -> str:
    ocr = get_paddle_ocr()
    try:
        result = ocr.predict(image)
    except Exception:
        try:
            result = ocr.ocr(image)
        except TypeError:
            result = ocr.ocr(image, cls=True)
    return extract_text_from_paddle_result(result)


def ocr_tesseract(image: np.ndarray, lang: str) -> str:
    import pytesseract

    config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return pytesseract.image_to_string(image, lang=lang, config=config)


def run_plate_inference(
    image_path: Path,
    detector_path: Path,
    ocr_backend: str = "paddle",
    tesseract_lang: str = "eng",
    save_vis: Path | None = None,
) -> dict[str, object]:
    detector_path = resolve_detector_path(detector_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    model = get_detector(str(detector_path))
    pred = model.predict(source=img, conf=0.2, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return {
            "image": str(image_path),
            "detector": str(detector_path),
            "bbox": None,
            "confidence": 0.0,
            "raw_text": "",
            "plate_text": "",
            "visualization": str(save_vis) if save_vis else "",
            "status": "no_plate",
        }

    best_idx = int(np.argmax(pred.boxes.conf.cpu().numpy()))
    box = pred.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    conf = float(pred.boxes.conf[best_idx].cpu().numpy())

    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("Detected bbox is empty after clipping.")

    if ocr_backend == "paddle":
        candidates = [
            preprocess_crop_paddle(crop),
            preprocess_crop_paddle_clahe(crop),
            preprocess_crop_paddle_sharp(crop),
        ]
        best_raw = ""
        best_norm = ""
        best_steps: list[str] = []
        best_pp_score = -10_000
        for cand in candidates:
            raw = ocr_paddle(cand)
            norm, steps = normalize_plate_text_verbose(raw)
            score = postprocess_score(norm)
            if score > best_pp_score:
                best_pp_score = score
                best_raw = raw
                best_norm = norm
                best_steps = steps
        raw_text = best_raw
        plate_text = best_norm
        normalization_steps = best_steps
        pp_score = best_pp_score
    else:
        proc = preprocess_crop_tesseract(crop)
        raw_text = ocr_tesseract(proc, tesseract_lang)
        plate_text, normalization_steps = normalize_plate_text_verbose(raw_text)
        pp_score = postprocess_score(plate_text)

    plate_format = detect_plate_format(plate_text)
    region_valid = is_region_code_valid(plate_text)
    plate_valid = plate_format != "unknown" and region_valid

    if save_vis:
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{plate_text or 'UNKNOWN'} ({conf:.2f})"
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        save_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_vis), vis)

    return {
        "image": str(image_path),
        "detector": str(detector_path),
        "bbox": (x1, y1, x2, y2),
        "confidence": conf,
        "raw_text": raw_text.strip(),
        "plate_text": plate_text,
        "plate_valid": plate_valid,
        "plate_format": plate_format,
        "region_valid": region_valid,
        "postprocess_score": pp_score,
        "normalization_steps": normalization_steps,
        "visualization": str(save_vis) if save_vis else "",
        "status": "ok",
    }


def main() -> None:
    args = parse_args()

    result = run_plate_inference(
        image_path=args.image,
        detector_path=args.detector,
        ocr_backend=args.ocr_backend,
        tesseract_lang=args.tesseract_lang,
        save_vis=args.save_vis,
    )

    if result["status"] == "no_plate":
        print("No plate detected.")
        return

    x1, y1, x2, y2 = result["bbox"]
    print(f"image={result['image']}")
    print(f"detector={result['detector']}")
    print(f"bbox=({x1}, {y1}, {x2}, {y2})")
    print(f"confidence={result['confidence']:.4f}")
    print(f"raw_text={result['raw_text']}")
    print(f"plate_text={result['plate_text']}")
    print(f"plate_valid={result['plate_valid']}")
    print(f"plate_format={result['plate_format']}")
    print(f"region_valid={result['region_valid']}")
    print(f"postprocess_score={result['postprocess_score']}")
    print(f"normalization_steps={','.join(result['normalization_steps'])}")
    print(f"visualization={result['visualization']}")


if __name__ == "__main__":
    main()
