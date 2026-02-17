#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from infer_plate import run_plate_inference

BATCH_FIELDNAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch plate inference for all images in a directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with images.")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("runs/detect/plate_kz/weights/best.pt"),
        help="Path to detector weights.",
    )
    parser.add_argument("--ocr-backend", choices=["paddle", "tesseract"], default="paddle")
    parser.add_argument("--tesseract-lang", type=str, default="eng")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/batch_predictions.csv"),
        help="Where to save predictions CSV.",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("outputs/batch_vis"),
        help="Where to save visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path in images:
        vis_path = args.vis_dir / image_path.name
        result = run_plate_inference(
            image_path=image_path,
            detector_path=args.detector,
            ocr_backend=args.ocr_backend,
            tesseract_lang=args.tesseract_lang,
            save_vis=vis_path,
        )
        rows.append(
            {
                "image": image_path.name,
                "pred_text": str(result["plate_text"]),
                "confidence": f"{float(result['confidence']):.4f}",
                "plate_valid": str(result["plate_valid"]),
                "plate_format": str(result["plate_format"]),
                "region_valid": str(result["region_valid"]),
                "postprocess_score": str(result["postprocess_score"]),
                "normalization_steps": ",".join(result["normalization_steps"]),
                "status": str(result["status"]),
                "visualization": str(vis_path),
            }
        )
        print(f"{image_path.name} -> {result['plate_text']} [{result['status']}]")

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved visualizations: {args.vis_dir}")


if __name__ == "__main__":
    main()
