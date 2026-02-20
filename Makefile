PYTHON ?= python3

.PHONY: test eval batch export-onnx format lint ui

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) scripts/evaluate.py --split test --images-root data/kz_plate/images/test --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/test_ocr_results.csv --report-path outputs/test_ocr_report.txt

batch:
	$(PYTHON) scripts/batch_infer.py --input-dir examples --detector runs/detect/plate_kz/weights/best.pt --output-csv outputs/examples_csv/predictions.csv --vis-dir outputs/examples_vis

export-onnx:
	$(PYTHON) scripts/export_model.py --weights runs/detect/plate_kz/weights/best.pt --format onnx --imgsz 640

ui:
	streamlit run app.py

format:
	$(PYTHON) -m black scripts tests

lint:
	$(PYTHON) -m ruff check scripts tests
