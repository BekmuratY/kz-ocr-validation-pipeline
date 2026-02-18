#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from config_utils import cfg_get, load_config
except ModuleNotFoundError:
    from scripts.config_utils import cfg_get, load_config

try:
    from postprocess import (
        clean_plate_text,
        detect_plate_format,
        get_region_info,
        is_region_code_valid,
        normalize_plate_text_verbose,
        postprocess_score,
        score_plate_text,
    )
except ModuleNotFoundError:
    from scripts.postprocess import (
        clean_plate_text,
        detect_plate_format,
        get_region_info,
        is_region_code_valid,
        normalize_plate_text_verbose,
        postprocess_score,
        score_plate_text,
    )
try:
    from plate_type import detect_plate_type
except ModuleNotFoundError:
    from scripts.plate_type import detect_plate_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and read license plate text from one image.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file.")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("runs/detect/plate_kz/weights/best.pt"),
        help="Path to trained detector weights.",
    )
    # OCR backend is fixed to PaddleOCR in this project.
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
    if det_dir and not Path(det_dir).exists():
        raise FileNotFoundError(f"PADDLEOCR_TEXT_DET_MODEL_DIR not found: {det_dir}")
    if rec_dir and not Path(rec_dir).exists():
        raise FileNotFoundError(f"PADDLEOCR_TEXT_REC_MODEL_DIR not found: {rec_dir}")
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


def run_plate_inference(
    image_path: Path,
    detector_path: Path,
    conf: float = 0.2,
    save_vis: Path | None = None,
) -> dict[str, object]:
    detector_path = resolve_detector_path(detector_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    model = get_detector(str(detector_path))
    pred = model.predict(source=img, conf=conf, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return {
            "image": str(image_path),
            "detector": str(detector_path),
            "bbox": None,
            "confidence": 0.0,
            "raw_text": "",
            "plate_text": "",
            "plate_valid": False,
            "plate_format": "unknown",
            "plate_type": "INVALID",
            "region_valid": False,
            "region_code": "",
            "region_name": "",
            "region_scheme": "unknown",
            "postprocess_score": 0,
            "normalization_steps": [],
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

    plate_format = detect_plate_format(plate_text)
    region_valid = is_region_code_valid(plate_text)
    region_code, region_name, region_scheme = get_region_info(plate_text)
    plate_valid = plate_format != "unknown" and region_valid
    plate_type = detect_plate_type(plate_text)

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
        "plate_type": plate_type,
        "region_valid": region_valid,
        "region_code": region_code or "",
        "region_name": region_name or "",
        "region_scheme": region_scheme,
        "postprocess_score": pp_score,
        "normalization_steps": normalization_steps,
        "visualization": str(save_vis) if save_vis else "",
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    detector = args.detector
    conf = args.conf
    if args.config is not None:
        detector = Path(cfg_get(cfg, "detector.weights", str(detector)))
        if conf is None:
            conf = float(cfg_get(cfg, "detector.conf", 0.2))
    if conf is None:
        conf = 0.2

    result = run_plate_inference(
        image_path=args.image,
        detector_path=detector,
        conf=conf,
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
    print(f"plate_type={result['plate_type']}")
    print(f"region_valid={result['region_valid']}")
    print(f"region_code={result['region_code']}")
    print(f"region_name={result['region_name']}")
    print(f"region_scheme={result['region_scheme']}")
    print(f"postprocess_score={result['postprocess_score']}")
    print(f"normalization_steps={','.join(result['normalization_steps'])}")
    print(f"visualization={result['visualization']}")


if __name__ == "__main__":
    main()
