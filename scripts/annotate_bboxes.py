#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual bbox annotation with OpenCV selectROI for YOLO format."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/kz_plate"),
        help="Dataset root with images/ and labels/ folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split to annotate.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Index to resume annotation from.",
    )
    return parser.parse_args()


def to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    ww = w / img_w
    hh = h / img_h
    return xc, yc, ww, hh


def main() -> None:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise RuntimeError(f"No images in {image_dir}")

    print("Controls:")
    print(" - Draw bbox and press ENTER/SPACE to confirm")
    print(" - Press c to clear selection")
    print(" - Press q in image window to quit")

    for i, img_path in enumerate(images[args.start_from :], start=args.start_from):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] Could not read {img_path}")
            continue

        title = f"{args.split} [{i+1}/{len(images)}] {img_path.name}"
        r = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)

        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            print(f"[skip] {img_path.name} no bbox selected")
            continue

        img_h, img_w = img.shape[:2]
        xc, yc, ww, hh = to_yolo(x, y, w, h, img_w, img_h)
        label = f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n"

        out_label = label_dir / f"{img_path.stem}.txt"
        out_label.write_text(label, encoding="utf-8")
        print(f"[ok] {img_path.name} -> {out_label}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
