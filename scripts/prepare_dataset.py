#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO-style dataset structure from raw car images."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/cars"),
        help="Path to raw images folder (default: data/cars).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/kz_plate"),
        help="Output dataset root (default: data/kz_plate).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks.",
    )
    return parser.parse_args()


def make_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    n = len(items)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    return {
        "train": items[:train_n],
        "val": items[train_n : train_n + val_n],
        "test": items[train_n + val_n :],
    }


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_data_yaml(root: Path) -> None:
    content = "\n".join(
        [
            f"path: {root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: plate",
            "",
        ]
    )
    (root / "data.yaml").write_text(content, encoding="utf-8")


def write_ocr_template(root: Path, split_map: dict[str, list[Path]]) -> None:
    out_csv = root / "ocr_labels_template.csv"
    lines = ["image,plate_text"]
    for split, files in split_map.items():
        for file in files:
            rel = f"images/{split}/{file.name}"
            lines.append(f"{rel},")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    images = sorted(
        [p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not images:
        raise RuntimeError(f"No images found in {args.source}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    make_dirs(args.out)
    split_map = split_items(images, args.train_ratio, args.val_ratio)

    for split, files in split_map.items():
        for src in files:
            dst = args.out / "images" / split / src.name
            link_or_copy(src, dst, args.copy)

            # Empty label file placeholder (you fill bbox annotations later).
            label_path = args.out / "labels" / split / f"{src.stem}.txt"
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")

    write_data_yaml(args.out)
    write_ocr_template(args.out, split_map)

    print(f"Prepared dataset at: {args.out}")
    for split, files in split_map.items():
        print(f"{split}: {len(files)} images")
    print("Next step: annotate bbox labels in data/kz_plate/labels/*/*.txt")


if __name__ == "__main__":
    main()
