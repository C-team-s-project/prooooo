"""Plate recognition and SQLite storage, shared by the UI and training entrypoint.

Run the UI:  streamlit run aajil.py
Train:       python aajil.py train --data data/plates.yaml --model yolov8n.pt
"""
from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("PLATE_DB", ROOT / "plates.db"))
MODEL_PATH = Path(os.getenv("PLATE_MODEL", ROOT / "models" / "license_plate.pt"))
APP_TIMEZONE = timezone(timedelta(hours=8))


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
        from rapidocr.ch_ppocr_rec import TextRecOutput
        from rapidocr.utils.output import RapidOCROutput

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Keep original pixels as one hypothesis; enhancement cannot recover
        # missing detail and must never replace the original observation.
        if self.language == "cyrillic":
            enhanced = enhance_plate(bgr)
        else:
            enhanced = None
        result = cast(RapidOCROutput, self.engine(bgr, use_det=True, use_cls=False))
        rows = []
        if result.boxes is not None and result.txts is not None and result.scores is not None:
            rows = [(box.tolist(), text, float(score))
                    for box, text, score in zip(result.boxes, result.txts, result.scores)]
        # Large plate characters are sometimes missed by document text detection.
        # Also recognize the plate as a line (the lower line for Japanese plates).
        h, w = bgr.shape[:2]
        y = int(h * 0.40) if self.language == "japan" else 0
        variants = [(bgr[y:], y, h)]
        if self.language == "japan":
            # Kana is much smaller than the serial digits. Without a margin the
            # line recognizer can drop it even when the digits are very clear.
            # Retain the original line and retry with a replicated border.
            padding = max(4, round(min(h - y, w) * 0.10))
            variants.append((cv2.copyMakeBorder(bgr[y:], padding, padding, padding, padding, cv2.BORDER_REPLICATE), y, h))
        if self.language == "korean":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            variants.append((cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1], 0, h))
        if enhanced is not None:
            variants.append((enhanced, 0, h))
        if self.language == "cyrillic" and h / w >= .50:
            top, bottom = max(1, int(h * .53)), int(h * .48)
            for source in (bgr, enhanced):
                variants.extend([(source[:top], 0, top), (source[bottom:], bottom, h)])
        for variant, line_y1, line_y2 in variants:
            line = cast(TextRecOutput, self.engine(variant, use_det=False, use_cls=False))
            for text, score in zip(line.txts or (), line.scores or ()):
                if text and float(score) >= 0.40:
                    rows.append(([[0, line_y1], [w, line_y1], [w, line_y2], [0, line_y2]], text, float(score)))
        return rows


def enhance_plate(image: Any):
    """Improve local contrast without inventing characters or changing size."""
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


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
    if not heading:
        return None
    if serial is not None:
        serial_text = re.sub(r"\s+", "", serial.group(2))
        kana = standalone_kana or serial.group(1)
    elif standalone_kana and numeric_rows:
        serial_text, kana = numeric_rows[-1], standalone_kana
    else:
        return None
    plate = f"{heading.group(1)} {heading.group(2)} {kana} {serial_text}"
    confidence = sum(float(row[2]) for row in reliable) / len(reliable)
    return plate, confidence


COUNTRY_MARKS = {"MGL": "Mongolia", "MNG": "Mongolia", "MGU": "Mongolia",
                 "JPN": "Japan", "USA": "United States", "KOR": "South Korea", "DEU": "Germany"}


def country_marks_from_rows(rows: list) -> set[str]:
    """Read separate country labels; never match a code inside a registration."""
    marks = set()
    lookalikes = str.maketrans("МГЛАЅ", "MGLAS")
    for _, text, score in rows:
        token = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).upper()).translate(lookalikes)
        if float(score) >= .75 and token in COUNTRY_MARKS:
            marks.add(COUNTRY_MARKS[token])
    return marks


def mongolian_plate_from_rows(rows: list, fallback_text: str = "", fallback_score: float = 0.0):
    """Join number/letter rows without treating the separate MGL oval as text."""
    reliable = [(normalize_plate(unicodedata.normalize("NFKC", text)), float(score))
                for _, text, score in rows if float(score) >= .40]
    reliable = [(text, score) for text, score in reliable if text not in COUNTRY_MARKS]
    translation = str.maketrans("ABCEHKMOPTXYDL06", "АВСЕНКМОРТХУДЛОБ")

    def letters(text):
        converted = text.translate(translation)
        return converted if re.fullmatch(r"[А-ЯӨҮ]{3}", converted) else None

    candidates = []
    for text, score in reliable:
        # Some line recognizers include the oval between/after the main text.
        for marker in ("MGL", "MNG", "MGU"):
            text = text.replace(marker, "")
        match = re.fullmatch(r"([0-9]{4})([A-ZА-ЯӨҮ06]{3})", text)
        if match and (suffix := letters(match.group(2))):
            candidates.append((match.group(1) + suffix, score))
    if candidates:
        return max(candidates, key=lambda candidate: candidate[1])
    numbers = [(text, score) for text, score in reliable if re.fullmatch(r"[0-9]{4}", text)]
    suffixes = [(suffix, score) for text, score in reliable
                if len(text) == 3 and (suffix := letters(text))]
    # Only use the global model's digits as a fallback. Its Latin letter guesses
    # are precisely what can confuse Mongolian plates with UK registrations.
    if not numbers and country_marks_from_rows(rows) == {"Mongolia"}:
        match = re.fullmatch(r"([0-9]{4})[A-ZА-ЯӨҮ]{3}", normalize_plate(fallback_text))
        if match:
            numbers.append((match.group(1), fallback_score))
    if numbers and suffixes:
        number, number_score = max(numbers, key=lambda candidate: candidate[1])
        suffix, suffix_score = max(suffixes, key=lambda candidate: candidate[1])
        return number + suffix, (number_score + suffix_score) / 2
    return None


def has_mongolian_script(rows: list) -> bool:
    """Require a four-digit/three-letter layout and actual Cyrillic evidence."""
    if not mongolian_plate_from_rows(rows):
        return False
    for _, text, score in rows:
        compact = normalize_plate(unicodedata.normalize("NFKC", text))
        match = re.fullmatch(r"(?:[0-9]{4})?([A-ZА-ЯӨҮ]{3})", compact)
        if match and float(score) >= .65 and len(re.findall(r"[А-ЯӨҮ]", match.group(1))) >= 2:
            return True
    return False


def read_plate_text(image: Any, reader: Any):
    import cv2

    rows = reader.readtext(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detail=1, paragraph=False)
    parsed = mongolian_plate_from_rows(rows)
    return (rows, parsed[0], parsed[1]) if parsed else (rows, "", 0.0)


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
        scale = max(1.0, min(4.0, 360 / max(crop.shape[:2])))
        ocr_crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
        # Unsupported countries can receive very high scores for a supported country.
        # Check Japanese script independently of the global classifier's score.
        japanese_rows = load_japanese_ocr().readtext(
            cv2.cvtColor(ocr_crop, cv2.COLOR_BGR2RGB), detail=1, paragraph=False,
        )
        korean_rows = load_korean_ocr().readtext(
            cv2.cvtColor(ocr_crop, cv2.COLOR_BGR2RGB), detail=1, paragraph=False,
        )
        cyrillic_rows, mongolian_plate, mongolian_score = read_plate_text(ocr_crop, load_ocr())
        if scale > 1:
            native_rows, _, _ = read_plate_text(crop, load_ocr())
            cyrillic_rows += native_rows
        japanese = japanese_plate_from_rows(japanese_rows)
        korean = korean_plate_from_rows(korean_rows)
        evidence = country_marks_from_rows(japanese_rows + korean_rows + cyrillic_rows)
        if japanese:
            evidence.add("Japan")
        if korean:
            evidence.add("South Korea")
        if has_mongolian_script(cyrillic_rows):
            evidence.add("Mongolia")
        if len(evidence) > 1:
            region, region_score = "Unknown", None
        elif evidence:
            region, region_score = next(iter(evidence)), None
            if region == "Japan" and japanese:
                plate, confidence = japanese
            elif region == "South Korea" and korean:
                plate, confidence = korean
            elif region == "Mongolia":
                parsed = mongolian_plate_from_rows(cyrillic_rows, plate, float(confidence))
                if parsed:
                    plate, confidence = parsed
                elif mongolian_plate:
                    plate, confidence = mongolian_plate, mongolian_score
        elif re.search(r"[\u3041-\u3096\u30a1-\u30fa\u4e00-\u9fff\uac00-\ud7a3]", "".join(
            row[1] for row in japanese_rows + korean_rows if float(row[2]) >= 0.60
        )):
            # Asian script without enough layout evidence is not proof of Japan.
            region, region_score = "Unknown", None
        elif any(float(score) >= .65 and len(re.findall(r"[А-ЯӨҮ]", text.upper())) >= 2
                 for _, text, score in cyrillic_rows):
            # Cyrillic evidence contradicts a Latin-only global guess, but
            # unreadable digits are insufficient to assert a Mongolian plate.
            region, region_score = "Unknown", None
        elif region_score is not None and float(region_score) < .75:
            region, region_score = "Unknown", None
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
        "MGL": "Монгол", "MNG": "Монгол", "USA": "АНУ", "GBR": "Их Британи", "DEU": "Герман",
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


def country_code(region: str | None) -> str:
    """Use the plate mark MGL for Mongolia and ISO alpha-3 labels elsewhere."""
    if not region or str(region).lower() == "unknown":
        return "—"
    code = str(region).strip().upper()
    aliases = {"MGL": "MGL", "MNG": "MGL", "MONGOLIA": "MGL", "MN": "MGL",
               "SOUTH KOREA": "KOR", "JAPAN": "JPN", "UNITED STATES": "USA", "GERMANY": "DEU"}
    if code in aliases:
        return aliases[code]
    try:
        import pycountry

        country = (pycountry.countries.get(alpha_3=code) or pycountry.countries.get(alpha_2=code)
                   or pycountry.countries.get(name=str(region)))
        if country:
            return country.alpha_3
    except ImportError:
        pass
    return code if code in COUNTRY_MARKS else "—"


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


def save_detection(plate: str, confidence: float, source: str, image_path: str = "", country: str = "Unknown") -> int:
    con = database()
    try:
        cursor = con.execute(
            "INSERT INTO detections(created_at,plate_text,confidence,source,image_path,country) VALUES(?,?,?,?,?,?)",
            (datetime.now(APP_TIMEZONE).isoformat(timespec="seconds"), plate, confidence, source, image_path, country),
        )
        con.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Бүртгэлийн дугаар үүссэнгүй.")
        return cursor.lastrowid
    finally:
        con.close()

