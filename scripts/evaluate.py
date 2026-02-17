#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from infer_plate import run_plate_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR quality on a split using ground-truth texts.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/kz_plate/ocr_labels_template.csv"),
        help="CSV with columns: image,plate_text",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("Cars"),
        help="Folder with original images by filename.",
    )
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
        default=Path("outputs/test_ocr_results.csv"),
        help="Per-image metrics output CSV.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/test_ocr_report.txt"),
        help="Summary report path.",
    )
    return parser.parse_args()


def clean(s: str) -> str:
    s = re.sub(r"[^A-ZА-Я0-9]", "", (s or "").upper())
    s = re.sub(r"^KZ", "", s)
    s = re.sub(r"KZ$", "", s)
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["image"].startswith(f"images/{args.split}/"):
                rows.append({"image": r["image"], "gt_text": clean(r.get("plate_text", ""))})

    if not rows:
        raise RuntimeError(f"No rows found for split={args.split} in {args.labels_csv}")

    out_rows: list[dict[str, str]] = []
    for row in rows:
        filename = Path(row["image"]).name
        image_path = args.images_root / filename
        result = run_plate_inference(
            image_path=image_path,
            detector_path=args.detector,
            ocr_backend=args.ocr_backend,
            tesseract_lang=args.tesseract_lang,
            save_vis=None,
        )
        pred_text = clean(str(result["plate_text"]))
        gt_text = row["gt_text"]
        exact_match = int(gt_text != "" and pred_text == gt_text)
        char_dist = levenshtein(gt_text, pred_text) if gt_text else ""
        gt_len = len(gt_text) if gt_text else ""
        out_rows.append(
            {
                "image": row["image"],
                "gt_text": gt_text,
                "pred_text": pred_text,
                "status": str(result["status"]),
                "exact_match": str(exact_match),
                "char_dist": str(char_dist),
                "gt_len": str(gt_len),
            }
        )

    with_gt = [r for r in out_rows if r["gt_text"]]
    exact_correct = sum(int(r["exact_match"]) for r in with_gt)
    exact_total = len(with_gt)
    exact_acc = (exact_correct / exact_total) if exact_total else 0.0
    char_err = sum(int(r["char_dist"]) for r in with_gt if r["char_dist"] != "")
    char_total = sum(int(r["gt_len"]) for r in with_gt if r["gt_len"] != "")
    cer = (char_err / char_total) if char_total else 0.0

    mismatch_pairs: Counter[str] = Counter()
    gt_char_counter: Counter[str] = Counter()
    pred_char_counter: Counter[str] = Counter()
    for r in with_gt:
        gt = r["gt_text"]
        pred = r["pred_text"]
        gt_char_counter.update(gt)
        pred_char_counter.update(pred)
        if len(gt) == len(pred):
            for g, p in zip(gt, pred):
                if g != p:
                    mismatch_pairs[f"{g}->{p}"] += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "gt_text", "pred_text", "status", "exact_match", "char_dist", "gt_len"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        f"split={args.split}",
        f"test_images={len(out_rows)}",
        f"with_gt={exact_total}",
        f"exact_accuracy={exact_acc:.4f} ({exact_correct}/{exact_total})" if exact_total else "exact_accuracy=N/A",
        f"cer={cer:.4f} ({char_err}/{char_total})" if char_total else "cer=N/A",
        "",
        "top_char_mismatches:",
    ]
    if mismatch_pairs:
        for pair, cnt in mismatch_pairs.most_common(10):
            lines.append(f"{pair}: {cnt}")
    else:
        lines.append("none")

    lines += [
        "",
        "char_distribution_gt:",
    ]
    if gt_char_counter:
        for ch, cnt in gt_char_counter.most_common():
            lines.append(f"{ch}: {cnt}")
    else:
        lines.append("none")

    lines += [
        "",
        "char_distribution_pred:",
    ]
    if pred_char_counter:
        for ch, cnt in pred_char_counter.most_common():
            lines.append(f"{ch}: {cnt}")
    else:
        lines.append("none")

    lines += [
        "",
        "per_image:",
    ]
    for r in out_rows:
        lines.append(
            f"{r['image']} | gt={r['gt_text']} | pred={r['pred_text']} | exact={r['exact_match']} | status={r['status']}"
        )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"exact_accuracy={exact_acc:.4f} ({exact_correct}/{exact_total})")
    print(f"cer={cer:.4f} ({char_err}/{char_total})")
    print(f"saved_csv={args.output_csv}")
    print(f"saved_report={args.report_path}")


if __name__ == "__main__":
    main()
