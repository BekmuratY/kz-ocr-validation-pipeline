#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-automatic plate bbox annotation with suggestion + review."
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
        help="Index to resume from.",
    )
    parser.add_argument(
        "--only-empty",
        action="store_true",
        help="Process only images that do not have non-empty label yet.",
    )
    return parser.parse_args()


def to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    ww = w / img_w
    hh = h / img_h
    return xc, yc, ww, hh


def from_yolo(line: str, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, xc, yc, ww, hh = map(float, parts)
    w = int(ww * img_w)
    h = int(hh * img_h)
    x = int(xc * img_w - w / 2)
    y = int(yc * img_h - h / 2)
    return x, y, w, h


def clamp_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def suggest_plate_bbox(img: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 50, 50)
    grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.002 * w * h:
            continue
        if area > 0.25 * w * h:
            continue
        aspect = ww / max(hh, 1)
        if not (2.0 <= aspect <= 8.5):
            continue
        cy = y + hh / 2
        vertical_bias = 1.0 - abs(cy - h * 0.7) / h
        score = area * (0.7 + 0.3 * max(0.0, vertical_bias))
        candidates.append((score, (x, y, ww, hh)))

    if not candidates:
        return None

    candidates.sort(key=lambda z: z[0], reverse=True)
    x, y, ww, hh = candidates[0][1]

    pad_w = int(ww * 0.08)
    pad_h = int(hh * 0.20)
    x -= pad_w
    y -= pad_h
    ww += 2 * pad_w
    hh += 2 * pad_h
    return clamp_box(x, y, ww, hh, w, h)


def read_existing_label(path: Path, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    line = content.splitlines()[0]
    box = from_yolo(line, img_w, img_h)
    if box is None:
        return None
    return clamp_box(*box, img_w, img_h)


def draw_preview(img: np.ndarray, box: tuple[int, int, int, int] | None, title: str) -> np.ndarray:
    out = img.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
    cv2.putText(
        out,
        "A=accept E=edit S=skip Q=quit",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
    )
    cv2.putText(out, title, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main() -> None:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise RuntimeError(f"No images in {image_dir}")

    print("Controls: A=accept suggestion, E=edit ROI, S=skip, Q=quit")
    for i, img_path in enumerate(images[args.start_from :], start=args.start_from):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] unreadable {img_path.name}")
            continue
        h, w = img.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"

        existing = read_existing_label(label_path, w, h)
        if args.only_empty and existing is not None:
            continue

        suggestion = existing if existing is not None else suggest_plate_bbox(img)
        title = f"{args.split} [{i+1}/{len(images)}] {img_path.name}"

        window = "semi_auto_annotate"
        preview = draw_preview(img, suggestion, title)
        cv2.imshow(window, preview)

        key = cv2.waitKey(0) & 0xFF
        # Accept both lowercase/uppercase shortcuts and ENTER/SPACE as quick accept.
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            print(f"[skip] {img_path.name}")
            continue

        final_box: tuple[int, int, int, int] | None = None
        if key in (ord("a"), ord("A"), 13, 32):
            final_box = suggestion
            if final_box is None:
                print(f"[skip] {img_path.name} no suggestion, use 'e' to draw manually")
                continue
        elif key in (ord("e"), ord("E")):
            r = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(title)
            x, y, ww, hh = map(int, r)
            if ww == 0 or hh == 0:
                print(f"[skip] {img_path.name} empty selection")
                continue
            final_box = clamp_box(x, y, ww, hh, w, h)
        else:
            print(f"[skip] {img_path.name} unknown key")
            continue

        x, y, ww, hh = final_box
        xc, yc, yw, yh = to_yolo(x, y, ww, hh, w, h)
        label_path.write_text(f"0 {xc:.6f} {yc:.6f} {yw:.6f} {yh:.6f}\n", encoding="utf-8")
        print(f"[ok] {img_path.name} -> {label_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
