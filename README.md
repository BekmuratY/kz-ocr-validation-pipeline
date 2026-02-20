# Распознавание номеров РК (локально, open source)

Проект — локальный пайплайн распознавания государственных номерных знаков Казахстана:
1. Подготовка датасета из `data/cars` (100 изображений).
2. Разметка bbox номера.
3. Обучение детектора (YOLOv8).
4. OCR + постобработка и вывод текста номера.

## 1) Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Для воспроизводимости можно использовать фиксированные версии:

```bash
pip install -r requirements-lock.txt
```

Оффлайн‑настройка (опционально):

```bash
cp .env.example .env
set -a && source .env && set +a
```

Оффлайн‑модели PaddleOCR (обязательно для полностью оффлайн режима):
- `PADDLEOCR_TEXT_DET_MODEL_DIR`
- `PADDLEOCR_TEXT_REC_MODEL_DIR`

Если пути заданы, но не существуют — инференс завершится с ошибкой.

Инструменты разработчика (опционально):

```bash
pre-commit install
```

## 2) Подготовка датасета

```bash
python scripts/prepare_dataset.py --source data/cars --out data/kz_plate --copy
```

Результат:
- `data/kz_plate/images/{train,val,test}`
- `data/kz_plate/labels/{train,val,test}` (пустые YOLO‑лейблы)
- `data/kz_plate/data.yaml`
- `data/kz_plate/ocr_labels_template.csv` (заполнить истинные номера)

## 3) Разметка bbox номера (YOLO)

Рекомендуется полуавтоматическая разметка:

```bash
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split train --only-empty
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split val --only-empty
python scripts/semi_auto_annotate.py --dataset data/kz_plate --split test --only-empty
```

Управление:
- `A` принять bbox
- `E` нарисовать/исправить
- `S` пропустить
- `Q` выход

Ручная разметка (fallback):

```bash
python scripts/annotate_bboxes.py --dataset data/kz_plate --split train
python scripts/annotate_bboxes.py --dataset data/kz_plate --split val
python scripts/annotate_bboxes.py --dataset data/kz_plate --split test
```

Формат строки:
`0 x_center y_center width height` (нормализовано в [0..1]).

## 4) Обучение детектора

```bash
python scripts/train_detector.py \
  --data data/kz_plate/data.yaml \
  --model models/yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

Выход:
- `runs/detect/plate_kz/weights/best.pt`

## 5) Инференс (одно изображение → текст)

```bash
python scripts/infer_plate.py \
  --config configs/default.yaml \
  --image data/cars/img_1.jpg \
  --detector runs/detect/plate_kz/weights/best.pt \
  --save-vis outputs/img_1_result.jpg
```

Если путь к весам неверный — `infer_plate.py` автоматически подберёт последний `runs/**/best.pt`.

## 6) Batch‑инференс (папка → CSV)

```bash
python scripts/batch_infer.py \
  --config configs/default.yaml \
  --input-dir data/new_images \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/new_images_csv/predictions.csv \
  --vis-dir outputs/new_images_vis
```

`predictions.csv` включает:
- `plate_valid` (валиден ли формат + регион)
- `plate_format` (детектированный формат)
- `plate_type` (NEW/OLD/DIPLOMATIC/TRANSIT/TRAILER/INVALID)
- `region_valid`
- `region_code`
- `region_name`
- `region_scheme` (`new`/`old`)
- `postprocess_score`
- `normalization_steps`

## 7) Оценка OCR (Exact Accuracy + CER)

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --labels-csv data/kz_plate/ocr_labels_template.csv \
  --split test \
  --images-root data/kz_plate/images/test \
  --detector runs/detect/plate_kz/weights/best.pt \
  --output-csv outputs/test_ocr_results.csv \
  --report-path outputs/test_ocr_report.txt
```

Отчёт включает топ‑пары несовпадений по символам.
Важно: `--images-root` должен указывать на папку конкретного сплита
(`data/kz_plate/images/test`, `data/kz_plate/images/val` или `data/kz_plate/images/train`),
потому что `evaluate.py` ищет картинки по имени файла внутри этого каталога.

## 8) Экспорт модели (mobile/offline)

```bash
python scripts/export_model.py \
  --config configs/default.yaml \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format onnx \
  --imgsz 640
```

TFLite:

```bash
python scripts/export_model.py \
  --config configs/default.yaml \
  --weights runs/detect/plate_kz/weights/best.pt \
  --format tflite \
  --imgsz 640
```

## 9) Тесты

```bash
pytest -q
```

Опциональный интеграционный тест (веса + PaddleOCR + Ultralytics):

```bash
RUN_INTEGRATION=1 pytest -q tests/test_integration_pipeline.py
```

Smoke‑тест (без строгих проверок):

```bash
RUN_SMOKE=1 pytest -q tests/test_integration_pipeline.py
```

## 10) Project Quality Helpers

- `scripts/postprocess.py` — нормализация/валидация.
- Большинство скриптов поддерживают `--config configs/default.yaml`.
- `Makefile`:
  - `make test`
  - `make eval`
  - `make batch`
  - `make export-onnx`
  - `make lint`
  - `make format`
- `.pre-commit-config.yaml` + `pyproject.toml` — линт/формат.
- `configs/default.yaml` — дефолтные пути.
- `demo/RUN_DEMO.md` — шаги демо.

Известные ограничения:
- сильный наклон или motion blur,
- очень маленький номер в кадре,
- ночные/контровые сцены, грязь,
- редкие/нестандартные форматы.

Рекомендации:
- расширить датасет сложными сценами,
- добавить аугментации blur/glare/low‑res,
- при необходимости дообучить OCR под домен KZ.

## 11) Финальный чек‑лист (EN)

1. Обучение детектора:
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
  --images-root data/kz_plate/images/test \
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
  --images-root data/kz_plate/images/test \
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

## 13) Структура проекта

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
│   │   ├── batch_20260218_220803
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
*** End Patch
