"""Run the plate-recognition app with Streamlit, or use the training CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from plate_core import (
    ROOT,
    DB_PATH,
    MODEL_PATH,
    database,
    normalize_plate,
    load_models,
    OnnxTextReader,
    load_ocr,
    load_japanese_ocr,
    load_korean_ocr,
    korean_plate_from_rows,
    japanese_plate_from_rows,
    read_plate_text,
    load_global_alpr,
    recognize_global,
    country_name,
    recognize,
    save_detection,
)


def run_ui():
    from dashboard import render_dashboard
    render_dashboard()


def train(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="runs/plate", name="detector", patience=20)
    best = Path("runs/plate/detector/weights/best.pt")
    print(f"Сургалт дууслаа. PLATE_MODEL={best.resolve()} гэж тохируулна уу.")


def main():
    parser = argparse.ArgumentParser(description="Mongolian vehicle plate detection and OCR")
    sub = parser.add_subparsers(dest="command")
    p_train = sub.add_parser("train", help="YOLO plate detector сургах")
    p_train.add_argument("--data", default="data/plates.yaml")
    p_train.add_argument("--model", default="yolov8n.pt", help="Урьдчилан сурсан YOLO weight")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        run_ui()


if __name__ == "__main__":
    main()
