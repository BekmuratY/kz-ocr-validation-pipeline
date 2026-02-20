# Финальный отчёт: Распознавание автомобильных номеров РК (локально, open source)

## 1. Цель задания

Реализовать локальный open-source пайплайн, который:
1. Использует датасет из 100 изображений автомобилей.
2. Подготавливает/размечает данные для распознавания номеров.
3. Обучает модель на этих данных.
4. На Python выполняет распознавание номера и перевод в текст.

Все этапы выполнены локально на компьютере, с заделом на офлайн/mobile интеграцию.

## 2. Данные проекта

- Исходные изображения: `data/cars/` (100 изображений, казахстанские номера).
- Структура датасета:
  - `data/kz_plate/images/{train,val,test}`
  - `data/kz_plate/labels/{train,val,test}`
  - `data/kz_plate/data.yaml`
  - `data/kz_plate/ocr_labels_template.csv` (истинные тексты номеров).

Текущий split:
- train: 70
- val: 20
- test: 10

## 3. Быстрый запуск (5 минут)

1. Проверка качества на test:
```bash
python3 scripts/evaluate.py --split test --images-root data/cars --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/test_ocr_results.csv --report-path outputs/test_ocr_report.txt
```

2. Пакетная проверка новых изображений:
```bash
python3 scripts/batch_infer.py --input-dir data/new_images --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/new_images_csv/predictions.csv --vis-dir outputs/new_images_vis
```

3. Экспорт для mobile/offline:
```bash
python3 scripts/export_model.py --weights runs/detect/plate_kz/weights/best.pt --format onnx --imgsz 640
```

## 4. Что реализовано

### 4.1 Подготовка датасета
- Скрипт: `scripts/prepare_dataset.py`
- Создаёт структуру в формате YOLO и шаблон CSV для OCR.

### 4.2 Разметка bbox номеров
- Полуавтоматическая разметка: `scripts/semi_auto_annotate.py`
- Ручной fallback: `scripts/annotate_bboxes.py`
- Размечены все 100 изображений.

### 4.3 Обучение детектора
- Скрипт: `scripts/train_detector.py`
- Модель: YOLOv8n (Ultralytics).
- Веса: `runs/detect/.../weights/best.pt`.

### 4.4 OCR + постобработка
- Скрипт: `scripts/infer_plate.py`
- OCR:
  - PaddleOCR (основной)
- Реализовано:
  - нормализация OCR-текста,
  - очистка `KZ` префикса/суффикса,
  - исправление типовых OCR-ошибок по позициям номера,
  - валидация формата и кода региона,
  - расчёт postprocess score.
Дополнительно:
- `plate_type` (NEW_STANDARD / OLD_STANDARD / DIPLOMATIC / TRANSIT / TRAILER / INVALID).

### 4.5 Batch и оценка
- Пакетный инференс: `scripts/batch_infer.py`
- Оценка качества OCR: `scripts/evaluate.py` (Exact Accuracy + CER)
- Экспорт под mobile/offline: `scripts/export_model.py`
- Тесты:
  - `tests/test_postprocess.py`
  - `tests/test_validation_rules.py`
  - `tests/test_batch_contract.py`

## 5. Метрики

### 5.1 Качество OCR на test (10 изображений)
Команда:
```bash
python3 scripts/evaluate.py --split test --images-root data/cars --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/test_ocr_results.csv --report-path outputs/test_ocr_report.txt
```

Дата запуска: 2026-02-18  
Использованные веса: `runs/detect/plate_kz/weights/best.pt`

Результат:
- Exact Accuracy: **1.0000 (10/10)**
- CER: **0.0000 (0/80)**

Файлы:
- `outputs/test_ocr_report.txt`
- `outputs/test_ocr_results.csv`

### 5.2 Проверка на новых изображениях
Папка: `data/new_images/` (5 изображений)

Команда:
```bash
python3 scripts/batch_infer.py --input-dir data/new_images --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/new_images_csv/predictions.csv --vis-dir outputs/new_images_vis
```

Дата запуска: 2026-02-18  
Использованные веса: `runs/detect/plate_kz/weights/best.pt`

Результаты:
- `outputs/new_images_csv/predictions.csv`
- `outputs/new_images_vis/` (визуализации bbox + текст)

Ключевые поля CSV:
- `plate_format`, `plate_type`, `region_code`, `region_name`, `region_scheme`.

## 6. Соответствие требованиям ТЗ

1. Open source:
- Ultralytics YOLOv8, PaddleOCR, OpenCV, NumPy, Pillow.

2. Локальная реализация:
- обучение, инференс, оценка, тесты запущены локально.

3. Задел на mobile/offline:
- есть экспорт в ONNX/TFLite,
- есть офлайн-конфиг через `.env.example`,
- для полностью офлайн OCR используются переменные
  `PADDLEOCR_TEXT_DET_MODEL_DIR` и `PADDLEOCR_TEXT_REC_MODEL_DIR`
  (при задании несуществующих путей выполнение завершится с ошибкой),
- логика нормализации/валидации готова для переноса в мобильный модуль.

## 7. Финальные артефакты

| Артефакт | Путь |
|---|---|
| Веса PyTorch | `runs/detect/plate_kz/weights/best.pt` |
| Веса ONNX | `runs/detect/plate_kz/weights/best.onnx` |
| Отчёт test OCR | `outputs/test_ocr_report.txt` |
| Таблица test OCR | `outputs/test_ocr_results.csv` |
| Предсказания новых фото | `outputs/new_images_csv/predictions.csv` |
| Визуализации новых фото | `outputs/new_images_vis/` |

## 8. Ограничения

Возможные проблемные условия:
- сильный наклон номера и motion blur,
- очень низкое разрешение номера в кадре,
- ночные/контровые сцены, блики и загрязнение номера,
- нестандартные/редкие форматы номеров.

## 9. План улучшений

1. Увеличить датасет (особенно ночь/дождь/низкое качество).
2. Добавить отдельный детектор качества кадра (флаг “переснять фото”).
3. Дообучить OCR на узком домене KZ-номеров.
4. Добавить ONNX Runtime benchmark (latency/FPS) для mobile-профиля.
5. Поддержать расширенный набор форматов номеров РК через отдельные regex-профили.

## 10. Воспроизводимость

- Основные зависимости: `requirements.txt`
- Зафиксированные версии: `requirements-lock.txt`
- Шаблон env: `.env.example`
- Тесты:
```bash
pytest -q
```

## 11. Итог

Задание выполнено end-to-end:
- датасет подготовлен и размечен,
- модель обучена,
- распознавание номера и перевод в текст реализованы на Python,
- есть пакетная проверка и метрики,
- подготовлен путь к офлайн/mobile внедрению.
