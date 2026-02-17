# Demo Run

## 1) Evaluate test set

```bash
make eval
```

Output files:
- `outputs/test_ocr_report.txt`
- `outputs/test_ocr_results.csv`

## 2) Run batch inference on new images

Put images into `new_images/`, then run:

```bash
make batch
```

Output files:
- `outputs/new_images_csv/predictions.csv`
- `outputs/new_images_vis/`

## 3) Export ONNX for deployment

```bash
make export-onnx
```

Output file:
- `runs/detect/runs/plate_kz/weights/best.onnx`
