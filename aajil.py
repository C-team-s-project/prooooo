"""International vehicle plate demo (FastALPR + multilingual ONNX OCR + SQLite).

Run the UI:  streamlit run aajil.py
Train:       python aajil.py train --data data/plates.yaml --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("PLATE_DB", ROOT / "plates.db"))
MODEL_PATH = Path(os.getenv("PLATE_MODEL", ROOT / "models" / "license_plate.pt"))


def database() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        plate_text TEXT NOT NULL,
        confidence REAL NOT NULL,
        source TEXT NOT NULL,
        image_path TEXT,
        country TEXT NOT NULL DEFAULT 'Unknown'
    )""")
    columns = {row[1] for row in con.execute("PRAGMA table_info(detections)")}
    if "country" not in columns:
        con.execute("ALTER TABLE detections ADD COLUMN country TEXT NOT NULL DEFAULT 'Unknown'")
    con.commit()
    return con


def normalize_plate(text: str) -> str:
    """Keep Cyrillic/Latin letters and digits; normalize whitespace and case."""
    return re.sub(r"[^A-ZА-ЯӨҮ0-9]", "", text.upper().replace("Ё", "Е"))


def load_models(model_path: str | Path):
    """Load detector and OCR lazily so the training CLI needs no Streamlit."""
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO weight not found: {model_path}. Train a plate detector or set PLATE_MODEL."
        )
    detector = YOLO(str(model_path))
    # Use a Cyrillic OCR model for local plate text.
    reader = load_ocr()
    return detector, reader


class OnnxTextReader:
    """Expose OCR rows using ONNX models without importing PyTorch DLLs."""

    def __init__(self, language: str):
        import rapidocr
        from rapidocr import RapidOCR, EngineType, LangRec, ModelType, OCRVersion

        self.language = language
        params = {
            "Global.model_root_dir": str(ROOT / "models" / "rapidocr"),
            "Global.use_cls": False,
            "Global.text_score": 0.40,
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH if language == "japan" else LangRec(language),
            "Rec.ocr_version": OCRVersion.PPOCRV6 if language == "japan" else OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.SMALL if language == "japan" else ModelType.MOBILE,
        }
        if language == "japan":
            # RapidOCR 3.10 ships this Japanese-capable model with its package.
            bundled = Path(rapidocr.__file__).resolve().parent / "models" / "PP-OCRv6_rec_small.onnx"
            if bundled.is_file():
                params["Rec.model_path"] = str(bundled)
        self.engine = RapidOCR(params=params)

    def readtext(self, image: Any, detail: int = 1, paragraph: bool = False):
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self.engine(bgr, use_det=True, use_cls=False)
        rows = []
        if result.boxes is not None and result.txts is not None and result.scores is not None:
            rows = [(box.tolist(), text, float(score))
                    for box, text, score in zip(result.boxes, result.txts, result.scores)]
        # Large plate characters are sometimes missed by document text detection.
        # Also recognize the plate as a line (the lower line for Japanese plates).
        h, w = bgr.shape[:2]
        y = int(h * 0.40) if self.language == "japan" else 0
        variants = [bgr[y:]]
        if self.language == "korean":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            variants.append(cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1])
        for variant in variants:
            line = self.engine(variant, use_det=False, use_cls=False)
            for text, score in zip(line.txts or (), line.scores or ()):
                if text and float(score) >= 0.40:
                    rows.append(([[0, y], [w, y], [w, h], [0, h]], text, float(score)))
        return rows


@lru_cache(maxsize=1)
def load_ocr():
    return OnnxTextReader("cyrillic")


@lru_cache(maxsize=1)
def load_japanese_ocr():
    """Read Japanese place names and kana that the global model cannot output."""
    return OnnxTextReader("japan")


@lru_cache(maxsize=1)
def load_korean_ocr():
    """Read Hangul registration characters absent from the global OCR alphabet."""
    return OnnxTextReader("korean")


def korean_plate_from_rows(rows: list):
    """Recognize ordinary South Korean plates: 2/3 digits, a usage syllable, 4 digits."""
    reliable = [row for row in rows if float(row[2]) >= 0.40]
    ordered = sorted(reliable, key=lambda row: (min(point[1] for point in row[0]), min(point[0] for point in row[0])))
    usage_letters = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호"
    candidates = [(row[1], float(row[2])) for row in reliable]
    if reliable:
        candidates.append(("".join(row[1] for row in ordered), sum(float(row[2]) for row in reliable) / len(reliable)))
    for text, confidence in sorted(candidates, key=lambda item: item[1], reverse=True):
        text = re.sub(r"[\s·.\-]", "", unicodedata.normalize("NFKC", text))
        match = re.search(r"(?<![0-9])([0-9]{2,3})([" + usage_letters + r"])([0-9]{4})(?![0-9])", text)
        if match:
            return "".join(match.groups()), confidence
    return None


def japanese_plate_from_rows(rows: list):
    """Require a Japanese place/class line and a kana/serial line as evidence."""
    reliable = [row for row in rows if float(row[2]) >= 0.40]
    ordered = sorted(reliable, key=lambda row: (min(point[1] for point in row[0]), min(point[0] for point in row[0])))
    text = unicodedata.normalize("NFKC", " ".join(row[1] for row in ordered))
    heading = re.search(r"([\u4e00-\u9fff々]{1,6})\s*([0-9A-Z]{2,3})(?![0-9A-Z])", text)
    serial = re.search(r"([\u3041-\u3096\u30a1-\u30fa])\s*([0-9・.]{1,4}(?:\s*[-－]\s*[0-9]{1,2})?)(?![0-9])", text)
    standalone_kana = next((row[1].strip() for row in reliable
                            if re.fullmatch(r"[\u3041-\u3096\u30a1-\u30fa]", row[1].strip())), None)
    numeric_rows = [unicodedata.normalize("NFKC", row[1]).strip() for row in ordered
                    if re.fullmatch(r"[0-9・.]{1,4}(?:[-－][0-9]{1,2})?", unicodedata.normalize("NFKC", row[1]).strip())]
    if not heading or (not serial and not (standalone_kana and numeric_rows)):
        return None
    serial_text = re.sub(r"\s+", "", serial.group(2)) if serial else numeric_rows[-1]
    kana = standalone_kana or serial.group(1)
    plate = f"{heading.group(1)} {heading.group(2)} {kana} {serial_text}"
    confidence = sum(float(row[2]) for row in reliable) / len(reliable)
    return plate, confidence


def read_plate_text(image: Any, reader: Any):
    """Read a full image when no plate detector is installed; return OCR rows and a plate guess."""
    import cv2

    ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = reader.readtext(ocr_image, detail=1, paragraph=False)
    raw = "".join(row[1] for row in rows)
    compact = normalize_plate(raw)
    # Mongolian plates commonly have four digits followed by two or three Cyrillic letters.
    # OCR may mistake the small MGL country mark between them for Latin letters.
    # Confirm MGL separately before interpreting Latin lookalikes as Cyrillic.
    match = re.search(r"(?<![0-9])([0-9]{4})(?:MGL|MGU)?([A-ZА-ЯЁӨҮ]{2,3})(?![A-ZА-ЯЁӨҮ])", compact)
    if match:
        confs = [float(row[2]) for row in rows]
        letters = match.group(2).translate(str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ"))
        return rows, match.group(1) + letters, sum(confs) / len(confs) if confs else 0.0
    return rows, "", 0.0


def load_global_alpr():
    """Load local ONNX models for global plate detection, OCR, and region prediction."""
    from fast_plate_ocr.inference import hub as ocr_hub
    from open_image_models.detection.core import hub as detector_hub

    # Keep model caches in the project; some Windows Python installations deny access to HOME.
    ocr_hub.MODEL_CACHE_DIR = ROOT / "models" / "fast-plate-ocr"
    detector_hub.MODEL_CACHE_DIR = ROOT / "models" / "open-image-models"
    from fast_alpr import ALPR

    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v2-global-model",
        ocr_device="cpu",
    )


def recognize_global(image: Any, alpr: Any):
    """Return detected plate crops, text, scores, and predicted country/region."""
    import cv2

    results = []
    h, w = image.shape[:2]
    for item in alpr.predict(image):
        ocr = item.ocr
        box = item.detection.bounding_box
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = ocr.confidence if ocr is not None else 0.0
        if isinstance(confidence, list):
            confidence = sum(float(value) for value in confidence) / len(confidence) if confidence else 0.0
        crop = image[y1:y2, x1:x2]
        region = ocr.region if ocr is not None else None
        region_score = ocr.region_confidence if ocr is not None else None
        plate = ocr.text if ocr is not None else ""
        # Unsupported countries can receive very high scores for a supported country.
        # Check Japanese script independently of the global classifier's score.
        japanese_rows = load_japanese_ocr().readtext(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=1, paragraph=False,
        )
        korean_rows = load_korean_ocr().readtext(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=1, paragraph=False,
        )
        japanese = japanese_plate_from_rows(japanese_rows)
        korean = korean_plate_from_rows(korean_rows)
        if japanese and korean:
            region, region_score = "Unknown", None
        elif japanese:
            plate, confidence = japanese
            region, region_score = "Japan", None
        elif korean:
            plate, confidence = korean
            region, region_score = "South Korea", None
        elif re.search(r"[\u3041-\u3096\u30a1-\u30fa\u4e00-\u9fff\uac00-\ud7a3]", "".join(
            row[1] for row in japanese_rows + korean_rows if float(row[2]) >= 0.60
        )):
            # Asian script without enough layout evidence is not proof of Japan.
            region, region_score = "Unknown", None
        if not japanese and not korean and (re.fullmatch(r"[0-9]{4}[A-Z]{2,3}", plate) or region_score is None or float(region_score) < 0.90):
            rows, mongolian_plate, mongolian_score = read_plate_text(crop, load_ocr())
            has_mgl_mark = any(normalize_plate(row[1]) in {"MGL", "MGU"} for row in rows)
            if has_mgl_mark and mongolian_plate:
                region = "Mongolia"
                region_score = None  # A visible-marker rule has no calibrated model probability.
                plate = mongolian_plate
                confidence = mongolian_score
        if plate:
            results.append((crop, plate, float(confidence), region, region_score))
    return results


def country_name(region: str | None) -> str:
    """Convert a global OCR region label into a readable country name when available."""
    if not region or str(region).lower() == "unknown":
        return "Тодорхойгүй"
    mongolian_names = {
        "MONGOLIA": "Монгол", "JAPAN": "Япон", "BELARUS": "Беларусь", "UNITED STATES": "АНУ",
        "SOUTH KOREA": "Өмнөд Солонгос", "KOREA, REPUBLIC OF": "Өмнөд Солонгос",
        "MNG": "Монгол", "USA": "АНУ", "GBR": "Их Британи", "DEU": "Герман",
        "RUS": "Орос", "CHN": "Хятад", "JPN": "Япон", "KOR": "Өмнөд Солонгос",
        "FRA": "Франц", "IND": "Энэтхэг", "AUS": "Австрали", "CAN": "Канад",
        "BRA": "Бразил", "ARG": "Аргентин", "ESP": "Испани", "ITA": "Итали",
        "POL": "Польш", "UKR": "Украин", "KAZ": "Казахстан", "KGZ": "Киргизстан",
    }
    code = str(region).upper()
    if code in mongolian_names:
        return mongolian_names[code]
    try:
        import pycountry

        country = (pycountry.countries.get(alpha_3=code)
                   or pycountry.countries.get(alpha_2=code)
                   or pycountry.countries.get(name=str(region)))
        if country:
            return mongolian_names.get(country.alpha_3, country.name)
    except ImportError:
        pass
    return str(region)


def recognize(image: Any, detector: Any, reader: Any, threshold: float = 0.25):
    """Return (plate crop, OCR text, combined confidence) for the best box."""
    import cv2

    result = detector.predict(image, conf=threshold, verbose=False)[0]
    candidates = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Upscale small plates to improve OCR readability.
        scale = max(1.0, 320 / max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Text readers accept RGB arrays, while OpenCV images are BGR.
        ocr_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ocr = reader.readtext(ocr_image, detail=1, paragraph=False)
        raw = "".join(row[1] for row in ocr)
        plate = normalize_plate(raw)
        det_conf = float(box.conf[0])
        ocr_conf = sum(float(row[2]) for row in ocr) / len(ocr) if ocr else 0.0
        candidates.append((crop, plate, det_conf * ocr_conf))
    return max(candidates, key=lambda item: item[2]) if candidates else (None, "", 0.0)


def save_detection(plate: str, confidence: float, source: str, image_path: str = "", country: str = "Unknown"):
    with database() as con:
        con.execute("INSERT INTO detections(created_at,plate_text,confidence,source,image_path,country) VALUES(?,?,?,?,?,?)",
                    (datetime.now().astimezone().isoformat(timespec="seconds"), plate,
                     confidence, source, image_path, country))


def run_ui():
    import cv2
    import numpy as np
    import pandas as pd
    import streamlit as st
    from PIL import Image

    st.set_page_config(page_title="Автомашины дугаар танигч", page_icon="🚘", layout="wide")
    st.title("🚘 Автомашины дугаар танигч")
    st.caption("Олон улсын дугаар илрүүлэлт · OCR · улсын бүсийн таамаг · SQLite бүртгэл")
    st.caption("Япон · Өмнөд Солонгос · АНУ · Монгол · Герман болон бусад дэмжигдсэн улс")
    st.info("Улсын нэр нь автомат таамаг юм. Global загвар 65 улсын ангилалтай; Япон, Өмнөд Солонгос, Монгол дугаарыг нэмэлт OCR-оор шалгана. Дэмжигдээгүй улсын дугаарыг өөр улстай андуурах боломжтой.")
    upload = st.file_uploader("Зураг эсвэл видео сонгох", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"])

    @st.cache_resource(show_spinner="Олон улсын дугаар таних загварыг ачаалж байна…")
    def international_alpr():
        return load_global_alpr()

    if upload:
        is_video = Path(upload.name).suffix.lower() in {".mp4", ".avi", ".mov"}
        if not is_video:
            pil = Image.open(upload).convert("RGB")
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            st.image(pil, caption=upload.name, use_container_width=True)
            if st.button("Дугаар таних", type="primary"):
                try:
                    with st.spinner("Дугаар болон улсын шинжийг уншиж байна…"):
                        results = recognize_global(image, international_alpr())
                    if not results:
                        st.warning("Дугаар олдсонгүй. Дугаарыг тод, том харагдах өнцгөөс авсан зураг оруулна уу.")
                    for crop, plate, score, region, region_score in results:
                        name = country_name(region)
                        region_detail = f" · улсын загварын оноо {region_score:.1%}" if region_score is not None else ""
                        st.info(f"Улсын таамаг: **{name}**{region_detail} · Дугаар: **{plate}** · OCR оноо {score:.1%}")
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Илрүүлсэн дугаар", width=420)
                        save_detection(plate, score, upload.name, country=region or "Unknown")
                except Exception as exc:
                    st.error(f"Загвар ачаалах/танихад алдаа гарлаа: {exc}")
        else:
            st.info("Видео танихад кадрыг жигд алгасна. Дугаар олдсон кадр бүр SQLite-д хадгалагдана.")
            step = st.slider("Кадр алгасах тоо", 1, 60, 10)
            if st.button("Видео боловсруулах", type="primary"):
                tmp = ROOT / "uploads"
                tmp.mkdir(exist_ok=True)
                path = tmp / Path(upload.name).name
                path.write_bytes(upload.getbuffer())
                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                bar = st.progress(0)
                result_area = st.empty()
                found = 0
                try:
                    alpr = international_alpr()
                    i = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if i % step == 0:
                            for _, plate, score, region, region_score in recognize_global(frame, alpr):
                                save_detection(plate, score, upload.name, country=region or "Unknown")
                                found += 1
                                result_area.write(f"Кадр {i}: **{country_name(region)}** · **{plate}** ({score:.0%})")
                        i += 1
                        if i % 10 == 0:
                            bar.progress(min(i / total, 1.0))
                    cap.release()
                    bar.progress(1.0)
                    st.success(f"Дууслаа. {found} танилт бүртгэв.")
                except Exception as exc:
                    cap.release()
                    st.error(f"Видео боловсруулахад алдаа: {exc}")

    st.subheader("Бүртгэлийн сан")
    with database() as con:
        rows = con.execute("SELECT id,created_at,plate_text,country,confidence,source FROM detections ORDER BY id DESC LIMIT 500").fetchall()
    st.dataframe(pd.DataFrame(rows, columns=["ID", "Огноо", "Дугаар", "Улс/бүс", "Итгэлцэл", "Эх сурвалж"]),
                 use_container_width=True, hide_index=True)


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
