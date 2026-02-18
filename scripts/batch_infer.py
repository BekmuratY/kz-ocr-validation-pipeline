#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil

try:
    from config_utils import cfg_get, load_config
except ModuleNotFoundError:
    from scripts.config_utils import cfg_get, load_config

try:
    from infer_plate import run_plate_inference
except ModuleNotFoundError:
    from scripts.infer_plate import run_plate_inference

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
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with images.")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold.")
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
    parser.add_argument(
        "--run-log-root",
        type=Path,
        default=Path("outputs/runs"),
        help="Root folder for timestamped run logs.",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="Disable timestamped run log copy.",
    )
    return parser.parse_args()


def inference_to_csv_row(image_name: str, result: dict[str, object], vis_path: Path) -> dict[str, str]:
    return {
        "image": image_name,
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = args.input_dir
    detector = args.detector
    conf = args.conf
    ocr_backend = args.ocr_backend
    tesseract_lang = args.tesseract_lang
    output_csv = args.output_csv
    vis_dir = args.vis_dir
    run_log_root = args.run_log_root
    if args.config is not None:
        input_dir = Path(cfg_get(cfg, "paths.batch_input_dir", str(input_dir)))
        detector = Path(cfg_get(cfg, "detector.weights", str(detector)))
        if conf is None:
            conf = float(cfg_get(cfg, "detector.conf", 0.2))
        ocr_backend = str(cfg_get(cfg, "ocr.backend", ocr_backend))
        tesseract_lang = str(cfg_get(cfg, "ocr.tesseract_lang", tesseract_lang))
        output_csv = Path(cfg_get(cfg, "paths.batch_output_csv", str(output_csv)))
        vis_dir = Path(cfg_get(cfg, "paths.batch_vis_dir", str(vis_dir)))
        run_log_root = Path(cfg_get(cfg, "paths.run_log_root", str(run_log_root)))

    if conf is None:
        conf = 0.2

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path in images:
        vis_path = vis_dir / image_path.name
        result = run_plate_inference(
            image_path=image_path,
            detector_path=detector,
            conf=conf,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            save_vis=vis_path,
        )
        rows.append(inference_to_csv_row(image_path.name, result, vis_path))
        print(f"{image_path.name} -> {result['plate_text']} [{result['status']}]")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved visualizations: {vis_dir}")

    if not args.no_run_log:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_log_root / f"batch_{ts}"
        run_vis = run_dir / "vis"
        run_csv = run_dir / "predictions.csv"
        run_vis.mkdir(parents=True, exist_ok=True)
        # Save CSV snapshot.
        with run_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BATCH_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            src = Path(row["visualization"])
            if src.exists():
                shutil.copy2(src, run_vis / src.name)
        print(f"Saved run log: {run_dir}")


if __name__ == "__main__":
    main()
