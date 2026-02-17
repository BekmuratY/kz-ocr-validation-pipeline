#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO detector to mobile-friendly formats.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/plate_kz/weights/best.pt"),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "tflite"],
        default="onnx",
        help="Export format.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input size for export.")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 export if supported.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 export if supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    exported = model.export(format=args.format, imgsz=args.imgsz, int8=args.int8, half=args.half)
    print(f"source_weights={args.weights}")
    print(f"export_format={args.format}")
    print(f"exported_artifact={exported}")


if __name__ == "__main__":
    main()
