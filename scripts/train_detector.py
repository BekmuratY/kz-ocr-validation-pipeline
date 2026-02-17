#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for license plates.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/kz_plate/data.yaml"),
        help="Path to YOLO data.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="plate_kz")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def check_labels(dataset_root: Path) -> None:
    labels_root = dataset_root / "labels"
    txt_files = list(labels_root.rglob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No label files found in {labels_root}")

    non_empty = 0
    for f in txt_files:
        if f.read_text(encoding="utf-8").strip():
            non_empty += 1
    if non_empty == 0:
        raise RuntimeError("All label files are empty. Annotate bboxes before training.")

    print(f"Found {len(txt_files)} label files, {non_empty} non-empty.")


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    dataset_root = data_yaml.parent
    check_labels(dataset_root)

    project_dir = Path(args.project).resolve()
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        device=args.device,
        degrees=3.0,
        translate=0.03,
        scale=0.25,
        perspective=0.0005,
        hsv_v=0.2,
        fliplr=0.5,
        mosaic=0.5,
    )

    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print("Validation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
