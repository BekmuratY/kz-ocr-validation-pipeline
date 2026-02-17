# KZ Plate Recognition (Local, Open Source)

This project is a local pipeline for Kazakhstan license plate recognition:
1. Prepare dataset from `Cars` (100 images).
2. Annotate plate bounding boxes.
3. Train a detector (YOLOv8).
4. Run OCR and output plate text.

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If `paddlepaddle` is heavy for your machine, you can use `tesseract` backend in inference.
For reproducible environment, you can pin exact versions:

```bash
pip install -r requirements-lock.txt
```

Offline env setup (optional):

```bash
cp .env.example .env
set -a && source .env && set +a
```

## 2) Prepare dataset split

```bash
python scripts/prepare_dataset.py --source Cars --out data/kz_plate --copy
```

Result:
- `data/kz_plate/images/{train,val,test}`
- `data/kz_plate/labels/{train,val,test}` (empty YOLO label placeholders)
- `data/kz_plate/data.yaml`
- `data/kz_plate/ocr_labels_template.csv` (fill true plate texts later)

## 3) Annotate plate bbox (YOLO format)

Recommended (faster): semi-automatic annotation with review.

```bash
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split train --only-empty
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split val --only-empty
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split test --only-empty
```

Controls:
- `A` accept suggested bbox
- `E` draw/edit bbox manually
- `S` skip image
- `Q` quit

Manual-only fallback:

```bash
python scripts/annotate_bboxes.py --dataset data/kz_plate --split train
python scripts/annotate_bboxes.py --dataset data/kz_plate --split val
python scripts/annotate_bboxes.py --dataset data/kz_plate --split test
```

Each label file must contain one line:
`0 x_center y_center width height` (normalized to [0..1]).

## 4) Train detector

```bash
python scripts/train_detector.py \
  --data data/kz_plate/data.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

Model output:
- `runs/detect/plate_kz/weights/best.pt`

## 5) Inference: image -> text

```bash
python scripts/infer_plate.py \
  --image Cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --ocr-backend paddle \
  --save-vis outputs/img_1_result.jpg
```

Tesseract fallback:

```bash
python scripts/infer_plate.py \
  --image Cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --ocr-backend tesseract \
  --save-vis outputs/img_1_result.jpg
```

If the detector path is wrong, `infer_plate.py` automatically picks the latest `runs/**/best.pt`.

## 6) Batch inference (folder -> CSV)

```bash
python scripts/batch_infer.py \
  --input-dir new_images \
  --detector runs/detect/plate_kz/weights/best.pt \
  --ocr-backend paddle \
  --output-csv outputs/new_images_csv/predictions.csv \
  --vis-dir outputs/new_images_vis
```

`predictions.csv` now includes:
- `plate_valid` (format + region check)
- `plate_format` (detected format label)
- `region_valid` (last 2 digits in expected region range)
- `postprocess_score` (normalization confidence proxy)
- `normalization_steps` (applied cleanup steps)

## 7) OCR evaluation (Exact Accuracy + CER)

```bash
python scripts/evaluate.py \
  --labels-csv data/kz_plate/ocr_labels_template.csv \
  --split test \
  --images-root Cars \
  --detector runs/detect/plate_kz/weights/best.pt \
  --ocr-backend paddle \
  --output-csv outputs/test_ocr_results.csv \
  --report-path outputs/test_ocr_report.txt
```

The report now also includes top character mismatch pairs (e.g., `B->8`).

## 8) Export detector for mobile/offline

```bash
python scripts/export_model.py \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format onnx \
  --imgsz 640
```

TFLite export:

```bash
python scripts/export_model.py \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format tflite \
  --imgsz 640
```

## 9) Run unit tests

```bash
pytest -q
```

## Mobile + Offline direction

- Export detector to ONNX/TFLite from Ultralytics.
- Keep OCR local (`Paddle Lite` or `Tesseract` on-device).
- Use same preprocessing + postprocessing logic from `scripts/infer_plate.py`.
# kz-ocr-validation-pipeline
