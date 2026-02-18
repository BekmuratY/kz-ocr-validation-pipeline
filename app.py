from __future__ import annotations

import csv
import io
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from scripts.batch_infer import BATCH_FIELDNAMES, inference_to_csv_row
from scripts.infer_plate import get_detector, get_paddle_ocr, run_plate_inference


def _save_upload_to_tmp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def _safe_image(image_obj) -> None:
    try:
        st.image(image_obj, width="stretch")
        return
    except TypeError:
        pass
    try:
        st.image(image_obj, use_column_width=True)
        return
    except TypeError:
        st.image(image_obj)


def _csv_bytes(rows: list[dict[str, object]], fieldnames: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


@st.cache_resource
def _warm_resources(detector_path: str, ocr_backend: str) -> bool:
    # Warm heavy resources once per app process.
    get_detector(detector_path)
    if ocr_backend == "paddle":
        get_paddle_ocr()
    return True


def _conf_schedule(base_conf: float, auto_retry: bool) -> list[float]:
    if not auto_retry:
        return [base_conf]
    vals = [base_conf, max(0.07, round(base_conf * 0.65, 2)), 0.05]
    uniq: list[float] = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return uniq


def _run_with_auto_conf(
    image_path: Path,
    detector: Path,
    base_conf: float,
    ocr_backend: str,
    tesseract_lang: str,
    save_vis: Path,
    auto_retry: bool,
) -> tuple[dict[str, object], float, int]:
    attempts = _conf_schedule(base_conf, auto_retry)
    last: dict[str, object] | None = None
    chosen_conf = attempts[0]
    for c in attempts:
        chosen_conf = c
        res = run_plate_inference(
            image_path=image_path,
            detector_path=detector,
            conf=c,
            ocr_backend=ocr_backend,
            tesseract_lang=tesseract_lang,
            save_vis=save_vis,
        )
        last = res
        if res.get("status") == "ok":
            return res, c, len(attempts)
    return last or {"status": "error"}, chosen_conf, len(attempts)


def _save_ui_run(single_row: dict[str, object] | None, batch_rows: list[dict[str, object]] | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/runs") / f"ui_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if single_row is not None:
        single_csv = run_dir / "single_result.csv"
        single_csv.write_bytes(_csv_bytes([single_row], list(single_row.keys())))
        vis = Path(str(single_row.get("visualization", "")))
        if vis.exists():
            shutil.copy2(vis, run_dir / vis.name)

    if batch_rows:
        batch_csv = run_dir / "batch_results.csv"
        batch_csv.write_bytes(_csv_bytes(batch_rows, BATCH_FIELDNAMES))
        vis_dir = run_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        for r in batch_rows:
            vis = Path(str(r.get("visualization", "")))
            if vis.exists():
                shutil.copy2(vis, vis_dir / vis.name)

    return run_dir


def main() -> None:
    st.set_page_config(page_title="KZ Plate Recognition", layout="wide")
    st.title("KZ Plate Recognition")
    st.caption("Local offline-friendly demo using YOLO + OCR.")

    with st.sidebar:
        st.header("Settings")
        model_mode = st.selectbox("Detector runtime", options=["PyTorch (.pt)", "ONNX (.onnx)"])
        default_detector = (
            "runs/detect/runs/plate_kz/weights/best.onnx"
            if "ONNX" in model_mode
            else "runs/detect/runs/plate_kz/weights/best.pt"
        )
        detector = st.text_input("Detector weights", value=default_detector)
        conf = st.slider("Detection confidence", min_value=0.01, max_value=0.50, value=0.20, step=0.01)
        auto_conf_retry = st.checkbox("Auto lower confidence if no plate", value=True)
        ocr_backend = st.selectbox("OCR backend", options=["paddle", "tesseract"], index=0)
        tesseract_lang = st.text_input("Tesseract lang", value="eng")

    detector_path = Path(detector)
    _warm_resources(str(detector_path), ocr_backend)

    tab_single, tab_batch = st.tabs(["Single image", "Batch images"])

    with tab_single:
        one = st.file_uploader(
            "Upload one image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
        )
        if one is not None and st.button("Run on single image", type="primary"):
            image_path = _save_upload_to_tmp(one)
            vis_path = Path(tempfile.gettempdir()) / f"vis_{one.name}"
            t0 = time.perf_counter()
            with st.spinner("Running recognition..."):
                result, used_conf, attempts = _run_with_auto_conf(
                    image_path=image_path,
                    detector=detector_path,
                    base_conf=conf,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    save_vis=vis_path,
                    auto_retry=auto_conf_retry,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original")
                _safe_image(one)
            with c2:
                st.subheader("Detection + OCR")
                if vis_path.exists():
                    _safe_image(str(vis_path))
                else:
                    st.info("No visualization available")

            st.subheader("Result")
            st.json(
                {
                    "plate_text": result.get("plate_text"),
                    "confidence": result.get("confidence"),
                    "used_conf": used_conf,
                    "attempts": attempts,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "plate_valid": result.get("plate_valid"),
                    "plate_format": result.get("plate_format"),
                    "region_valid": result.get("region_valid"),
                    "status": result.get("status"),
                }
            )

            if vis_path.exists():
                st.download_button(
                    label="Download visualization",
                    data=vis_path.read_bytes(),
                    file_name=f"result_{one.name}",
                    mime="image/jpeg",
                )

            single_row = {
                "image": one.name,
                "pred_text": result.get("plate_text", ""),
                "confidence": result.get("confidence", 0.0),
                "plate_valid": result.get("plate_valid", False),
                "plate_format": result.get("plate_format", "unknown"),
                "region_valid": result.get("region_valid", False),
                "postprocess_score": result.get("postprocess_score", 0),
                "normalization_steps": ",".join(result.get("normalization_steps", [])),
                "status": result.get("status", "error"),
                "visualization": str(vis_path),
            }
            st.download_button(
                label="Download single result CSV",
                data=_csv_bytes([single_row], BATCH_FIELDNAMES),
                file_name="single_result.csv",
                mime="text/csv",
            )

            run_dir = _save_ui_run(single_row=single_row, batch_rows=None)
            st.caption(f"Saved UI run log: {run_dir}")

    with tab_batch:
        files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        if files and st.button("Run batch", type="primary"):
            rows: list[dict[str, str]] = []
            progress = st.progress(0.0)
            t0 = time.perf_counter()
            for idx, f in enumerate(files, start=1):
                image_path = _save_upload_to_tmp(f)
                vis_path = Path(tempfile.gettempdir()) / f"vis_{f.name}"
                result, _, _ = _run_with_auto_conf(
                    image_path=image_path,
                    detector=detector_path,
                    base_conf=conf,
                    ocr_backend=ocr_backend,
                    tesseract_lang=tesseract_lang,
                    save_vis=vis_path,
                    auto_retry=auto_conf_retry,
                )
                rows.append(inference_to_csv_row(f.name, result, vis_path))
                progress.progress(idx / len(files))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            st.subheader("Batch results")
            st.caption(f"Processed {len(files)} images in {elapsed_ms:.2f} ms")
            st.dataframe(rows, use_container_width=True)

            st.download_button(
                label="Download batch CSV",
                data=_csv_bytes(rows, BATCH_FIELDNAMES),
                file_name="batch_results.csv",
                mime="text/csv",
            )

            run_dir = _save_ui_run(single_row=None, batch_rows=rows)
            st.caption(f"Saved UI run log: {run_dir}")


if __name__ == "__main__":
    main()
