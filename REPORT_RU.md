# Финальный отчёт: Распознавание автомобильных номеров РК (локально, open source)

## 1. Цель задания

Реализовать локальный open-source пайплайн, который:
1. Использует датасет из 100 изображений автомобилей.
2. Подготавливает/размечает данные для распознавания номеров.
3. Обучает модель на этих данных.
4. На Python выполняет распознавание номера и перевод в текст.

Все этапы выполнены локально на компьютере, с заделом на офлайн/mobile интеграцию.

## 2. Данные проекта

- Исходные изображения: `Cars/` (100 изображений, казахстанские номера).
- Структура датасета:
  - `data/kz_plate/images/{train,val,test}`
  - `data/kz_plate/labels/{train,val,test}`
  - `data/kz_plate/data.yaml`
  - `data/kz_plate/ocr_labels_template.csv` (истинные тексты номеров).

Текущий split:
- train: 70
- val: 20
- test: 10

## 3. Что реализовано

### 3.1 Подготовка датасета
- Скрипт: `scripts/prepare_dataset.py`
- Создаёт структуру в формате YOLO и шаблон CSV для OCR.

### 3.2 Разметка bbox номеров
- Полуавтоматическая разметка: `scripts/semi_auto_annotate.py`
- Ручной fallback: `scripts/annotate_bboxes.py`
- Размечены все 100 изображений.

### 3.3 Обучение детектора
- Скрипт: `scripts/train_detector.py`
- Модель: YOLOv8n (Ultralytics).
- Веса: `runs/detect/.../weights/best.pt`.

### 3.4 OCR + постобработка
- Скрипт: `scripts/infer_plate.py`
- OCR:
  - PaddleOCR (основной)
  - Tesseract (резервный)
- Реализовано:
  - нормализация OCR-текста,
  - очистка `KZ` префикса/суффикса,
  - исправление типовых OCR-ошибок по позициям номера,
  - валидация формата и кода региона,
  - расчёт postprocess score.

### 3.5 Batch и оценка
- Пакетный инференс: `scripts/batch_infer.py`
- Оценка качества OCR: `scripts/evaluate.py` (Exact Accuracy + CER)
- Экспорт под mobile/offline: `scripts/export_model.py`
- Тесты:
  - `tests/test_postprocess.py`
  - `tests/test_validation_rules.py`
  - `tests/test_batch_contract.py`

## 4. Метрики

### 4.1 Качество OCR на test (10 изображений)
Команда:
```bash
python3 scripts/evaluate.py --split test --images-root Cars --detector runs/detect/runs/plate_kz/weights/best.pt --ocr-backend paddle --output-csv outputs/test_ocr_results.csv --report-path outputs/test_ocr_report.txt
```

Результат:
- Exact Accuracy: **1.0000 (10/10)**
- CER: **0.0000 (0/80)**

Файлы:
- `outputs/test_ocr_report.txt`
- `outputs/test_ocr_results.csv`

### 4.2 Проверка на новых изображениях
Папка: `new_images/` (5 изображений)

Команда:
```bash
python3 scripts/batch_infer.py --input-dir new_images --detector runs/detect/runs/plate_kz/weights/best.pt --ocr-backend paddle --output-csv outputs/new_images_csv/predictions.csv --vis-dir outputs/new_images_vis
```

Результаты:
- `outputs/new_images_csv/predictions.csv`
- `outputs/new_images_vis/` (визуализации bbox + текст)

## 5. Соответствие требованиям ТЗ

1. Open source:
- Ultralytics YOLOv8, PaddleOCR/Tesseract, OpenCV, NumPy, Pillow.

2. Локальная реализация:
- обучение, инференс, оценка, тесты запущены локально.

3. Задел на mobile/offline:
- есть экспорт в ONNX/TFLite,
- есть офлайн-конфиг через `.env.example`,
- логика нормализации/валидации готова для переноса в мобильный модуль.

## 6. Mobile/Offline шаги

Экспорт в ONNX:
```bash
pip install onnx onnxruntime
python3 scripts/export_model.py --weights runs/detect/runs/plate_kz/weights/best.pt --format onnx --imgsz 640
```

Экспорт в TFLite:
```bash
python3 scripts/export_model.py --weights runs/detect/runs/plate_kz/weights/best.pt --format tflite --imgsz 640
```

Для полностью офлайн OCR:
- использовать локальные пути моделей через env:
  - `PADDLEOCR_TEXT_DET_MODEL_DIR`
  - `PADDLEOCR_TEXT_REC_MODEL_DIR`

## 7. Воспроизводимость

- Основные зависимости: `requirements.txt`
- Зафиксированные версии: `requirements-lock.txt`
- Шаблон env: `.env.example`
- Тесты:
```bash
pytest -q
```

## 8. Итог

Задание выполнено end-to-end:
- датасет подготовлен и размечен,
- модель обучена,
- распознавание номера и перевод в текст реализованы на Python,
- есть пакетная проверка и метрики,
- подготовлен путь к офлайн/mobile внедрению.
