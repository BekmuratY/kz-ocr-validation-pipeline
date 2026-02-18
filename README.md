# KZ Plate Recognition (Local, Open Source)

This project is a local pipeline for Kazakhstan license plate recognition:
1. Prepare dataset from `data/cars` (100 images).
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

PaddleOCR is the only OCR backend used in this project.
For reproducible environment, you can pin exact versions:

```bash
pip install -r requirements-lock.txt
```

Offline env setup (optional):

```bash
cp .env.example .env
set -a && source .env && set +a
```

Offline PaddleOCR model paths (required for fully offline runs):
- `PADDLEOCR_TEXT_DET_MODEL_DIR`
- `PADDLEOCR_TEXT_REC_MODEL_DIR`

If these are set and the paths do not exist, inference will fail fast with a clear error.

Developer tooling (optional):

```bash
pre-commit install
```

## 2) Prepare dataset split

```bash
python scripts/prepare_dataset.py --source data/cars --out data/kz_plate --copy
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
  --model models/yolov8n.pt \
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
  --config configs/default.yaml \
  --image data/cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --save-vis outputs/img_1_result.jpg
```

If the detector path is wrong, `infer_plate.py` automatically picks the latest `runs/**/best.pt`.

## 6) Batch inference (folder -> CSV)

```bash
python scripts/batch_infer.py \
  --config configs/default.yaml \
  --input-dir data/new_images \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/new_images_csv/predictions.csv \
  --vis-dir outputs/new_images_vis
```

`predictions.csv` now includes:
- `plate_valid` (format + region check)
- `plate_format` (detected format label)
- `plate_type` (NEW_STANDARD / OLD_STANDARD / DIPLOMATIC / TRANSIT / TRAILER / INVALID)
- `region_valid` (last 2 digits in expected region range)
- `region_code` (new: last 2 digits, old: leading letter)
- `region_name` (mapped region name when available)
- `region_scheme` (`new` or `old`)
- `postprocess_score` (normalization confidence proxy)
- `normalization_steps` (applied cleanup steps)

## 7) OCR evaluation (Exact Accuracy + CER)

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --labels-csv data/kz_plate/ocr_labels_template.csv \
  --split test \
  --images-root data/cars \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/test_ocr_results.csv \
  --report-path outputs/test_ocr_report.txt
```

The report now also includes top character mismatch pairs (e.g., `B->8`).

## 8) Export detector for mobile/offline

```bash
python scripts/export_model.py \
  --config configs/default.yaml \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format onnx \
  --imgsz 640
```

TFLite export:

```bash
python scripts/export_model.py \
  --config configs/default.yaml \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format tflite \
  --imgsz 640
```

## 9) Run unit tests

```bash
pytest -q
```

Optional integration test (requires weights + PaddleOCR + Ultralytics):

```bash
RUN_INTEGRATION=1 pytest -q tests/test_integration_pipeline.py
```

Optional smoke test (no strict assertions, just no exceptions):

```bash
RUN_SMOKE=1 pytest -q tests/test_integration_pipeline.py
```

## 10) Project Quality Helpers

- `scripts/postprocess.py` contains normalization and validation rules (separate from model code).
- Most runtime scripts support `--config configs/default.yaml` and load defaults from YAML.
- `Makefile` shortcuts:
  - `make test`
  - `make eval`
  - `make batch`
  - `make export-onnx`
  - `make lint`
  - `make format`
- `.pre-commit-config.yaml` + `pyproject.toml` provide lint/format rules.
- `configs/default.yaml` contains default runtime paths and values.
- `demo/RUN_DEMO.md` has minimal demo run steps.

Known limitations:
- strong tilt or motion blur,
- very small plates in the frame,
- night/backlight glare and dirty plates,
- rare or non-standard plate layouts.

Recommended next steps:
- expand the dataset with hard conditions,
- add augmentations for blur/glare/low-res,
- consider OCR fine-tuning on KZ-specific plates.

## 11) Final Repro Checklist

1. Train detector (or ensure weights exist):
```bash
python scripts/train_detector.py \
  --data data/kz_plate/data.yaml \
  --model models/yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

2. Single image inference:
```bash
python scripts/infer_plate.py \
  --config configs/default.yaml \
  --image data/cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --save-vis outputs/img_1_result.jpg
```

3. Batch inference:
```bash
python scripts/batch_infer.py \
  --config configs/default.yaml \
  --input-dir data/new_images \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/new_images_csv/predictions.csv \
  --vis-dir outputs/new_images_vis
```

4. OCR evaluation:
```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --labels-csv data/kz_plate/ocr_labels_template.csv \
  --split test \
  --images-root data/cars \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/test_ocr_results.csv \
  --report-path outputs/test_ocr_report.txt
```

5. UI:
```bash
make ui
```

6. Tests:
```bash
pytest -q
```

## 12) Финальный чек‑лист (RU)

1. Обучение детектора (или убедиться, что веса есть):
```bash
python scripts/train_detector.py \
  --data data/kz_plate/data.yaml \
  --model models/yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

2. Инференс одного изображения:
```bash
python scripts/infer_plate.py \
  --config configs/default.yaml \
  --image data/cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --save-vis outputs/img_1_result.jpg
```

3. Batch‑инференс:
```bash
python scripts/batch_infer.py \
  --config configs/default.yaml \
  --input-dir data/new_images \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/new_images_csv/predictions.csv \
  --vis-dir outputs/new_images_vis
```

4. Оценка OCR:
```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --labels-csv data/kz_plate/ocr_labels_template.csv \
  --split test \
  --images-root data/cars \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/test_ocr_results.csv \
  --report-path outputs/test_ocr_report.txt
```

5. UI:
```bash
make ui
```

6. Тесты:
```bash
pytest -q
```

## 13) Project Structure

```text
.
├── configs
│   └── default.yaml
├── data
│   ├── cars
│   │   ├── img_1.jpg
│   │   ├── img_10.jpg
│   │   ├── img_100.jpg
│   │   ├── img_11.jpg
│   │   ├── img_12.jpg
│   │   ├── ...
│   │   ├── img_95.jpg
│   │   ├── img_96.jpg
│   │   ├── img_97.jpg
│   │   ├── img_98.jpg
│   │   └── img_99.jpg
│   ├── kz_plate
│   │   ├── images
│   │   │   ├── test
│   │   │   │   ├── img_11.jpg
│   │   │   │   ├── img_20.jpg
│   │   │   │   ├── img_21.jpg
│   │   │   │   ├── img_24.jpg
│   │   │   │   ├── img_34.jpg
│   │   │   │   ├── img_37.jpg
│   │   │   │   ├── img_40.jpg
│   │   │   │   ├── img_82.jpg
│   │   │   │   ├── img_87.jpg
│   │   │   │   └── img_94.jpg
│   │   │   ├── train
│   │   │   │   ├── img_10.jpg
│   │   │   │   ├── img_100.jpg
│   │   │   │   ├── img_13.jpg
│   │   │   │   ├── img_14.jpg
│   │   │   │   ├── img_15.jpg
│   │   │   │   ├── ...
│   │   │   │   ├── img_91.jpg
│   │   │   │   ├── img_92.jpg
│   │   │   │   ├── img_96.jpg
│   │   │   │   ├── img_98.jpg
│   │   │   │   └── img_99.jpg
│   │   │   └── val
│   │   │       ├── img_1.jpg
│   │   │       ├── img_12.jpg
│   │   │       ├── img_19.jpg
│   │   │       ├── img_31.jpg
│   │   │       ├── img_33.jpg
│   │   │       ├── ...
│   │   │       ├── img_89.jpg
│   │   │       ├── img_9.jpg
│   │   │       ├── img_93.jpg
│   │   │       ├── img_95.jpg
│   │   │       └── img_97.jpg
│   │   ├── labels
│   │   │   ├── test
│   │   │   │   ├── img_11.txt
│   │   │   │   ├── img_20.txt
│   │   │   │   ├── img_21.txt
│   │   │   │   ├── img_24.txt
│   │   │   │   ├── img_34.txt
│   │   │   │   ├── img_37.txt
│   │   │   │   ├── img_40.txt
│   │   │   │   ├── img_82.txt
│   │   │   │   ├── img_87.txt
│   │   │   │   └── img_94.txt
│   │   │   ├── train
│   │   │   │   ├── img_10.txt
│   │   │   │   ├── img_100.txt
│   │   │   │   ├── img_13.txt
│   │   │   │   ├── img_14.txt
│   │   │   │   ├── img_15.txt
│   │   │   │   ├── ...
│   │   │   │   ├── img_91.txt
│   │   │   │   ├── img_92.txt
│   │   │   │   ├── img_96.txt
│   │   │   │   ├── img_98.txt
│   │   │   │   └── img_99.txt
│   │   │   ├── val
│   │   │   │   ├── img_1.txt
│   │   │   │   ├── img_12.txt
│   │   │   │   ├── img_19.txt
│   │   │   │   ├── img_31.txt
│   │   │   │   ├── img_33.txt
│   │   │   │   ├── ...
│   │   │   │   ├── img_89.txt
│   │   │   │   ├── img_9.txt
│   │   │   │   ├── img_93.txt
│   │   │   │   ├── img_95.txt
│   │   │   │   └── img_97.txt
│   │   │   ├── train.cache
│   │   │   └── val.cache
│   │   ├── data.yaml
│   │   └── ocr_labels_template.csv
│   └── new_images
│       ├── test_img_1.jpg
│       ├── test_img_10.jpg
│       ├── test_img_11.jpg
│       ├── test_img_12.jpg
│       ├── test_img_13.jpg
│       ├── test_img_14.jpg
│       ├── test_img_15.jpg
│       ├── test_img_2.jpg
│       ├── test_img_3.jpg
│       ├── test_img_4.jpg
│       ├── test_img_5.jpg
│       ├── test_img_6.jpg
│       ├── test_img_7.webp
│       ├── test_img_8.jpg
│       └── test_img_9.jpg
├── demo
│   └── RUN_DEMO.md
├── examples
│   ├── for_test_1.webp
│   ├── for_test_2.webp
│   ├── for_test_3.webp
│   ├── for_test_4.webp
│   ├── image.webp
│   ├── img_test_1.webp
│   ├── test_img_11.jpg
│   ├── test_img_12.jpg
│   └── акцент_2014.webp
├── models
│   └── yolov8n.pt
├── outputs
│   ├── exports
│   │   └── 2026-02-18T16-43_export.csv
│   ├── new_images_csv
│   │   └── predictions.csv
│   ├── new_images_vis
│   │   ├── test_img_1.jpg
│   │   ├── test_img_10.jpg
│   │   ├── test_img_11.jpg
│   │   ├── test_img_12.jpg
│   │   ├── test_img_13.jpg
│   │   ├── test_img_14.jpg
│   │   ├── test_img_15.jpg
│   │   ├── test_img_2.jpg
│   │   ├── test_img_3.jpg
│   │   ├── test_img_4.jpg
│   │   ├── test_img_5.jpg
│   │   ├── test_img_6.jpg
│   │   ├── test_img_7.webp
│   │   ├── test_img_8.jpg
│   │   └── test_img_9.jpg
│   ├── releases
│   │   └── final_release.tgz
│   ├── runs
│   │   ├── batch_20260218_215444
│   │   └── ui_20260218_220543
│   ├── test_ocr_report.txt
│   └── test_ocr_results.csv
├── runs
│   └── detect
│       └── plate_kz
│           ├── args.yaml
│           ├── labels.jpg
│           ├── results.csv
│           └── weights
│               ├── best.pt
│               └── last.pt
├── scripts
│   ├── annotate_bboxes.py
│   ├── batch_infer.py
│   ├── config_utils.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── infer_plate.py
│   ├── plate_type.py
│   ├── postprocess.py
│   ├── prepare_dataset.py
│   ├── semi_auto_annotate.py
│   └── train_detector.py
├── tests
│   ├── conftest.py
│   ├── test_batch_contract.py
│   ├── test_integration_pipeline.py
│   ├── test_postprocess.py
│   └── test_validation_rules.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── app.py
├── Makefile
├── pyproject.toml
├── README.md
├── REPORT.md
├── REPORT_RU.md
├── requirements-lock.txt
└── requirements.txt
```

## 14) Streamlit UI

Run local web interface:

```bash
make ui
```

or:

```bash
streamlit run app.py
```

In UI you can:
- upload one image and see detected bbox + plate text
- upload multiple images and get batch table results
- choose detector runtime path (`.pt` or `.onnx`)
- enable auto-lower confidence if plate is not detected
- download single-image visualization and CSV
- download batch CSV
- get timestamped UI run logs in `outputs/runs/ui_*`

## Mobile + Offline direction

- Export detector to ONNX/TFLite from Ultralytics.
- Keep OCR local (Paddle Lite on-device).
- Use same preprocessing + postprocessing logic from `scripts/infer_plate.py`.
# kz-ocr-validation-pipeline
