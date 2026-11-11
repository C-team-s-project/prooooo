"""Mongolian plate OCR, used alongside the international recognition pipeline."""
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any
from plate_core import ROOT, normalize_plate

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
        return [row for group in self.read_groups(image) for row in group]

    def read_groups(self, image: Any):
        """Keep text detection and whole-line recognition as separate alternatives."""
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self.engine(bgr, use_det=True, use_cls=False)
        rows = []
        if result.boxes is not None and result.txts is not None and result.scores is not None:
            rows = [(box.tolist(), text, float(score))
                    for box, text, score in zip(result.boxes, result.txts, result.scores)]
        groups = [rows]
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
                    groups.append([([[0, y], [w, y], [w, h], [0, h]], text, float(score))])
        if self.language == "cyrillic" and h / w >= 0.45:
            # Square plates have digits above letters. Recognizing the entire
            # crop as one text line squashes both lines together.
            split = h // 2
            split_rows = []
            for top, bottom in ((0, split), (split, h)):
                line = self.engine(bgr[top:bottom], use_det=False, use_cls=False)
                for text, score in zip(line.txts or (), line.scores or ()):
                    if text and float(score) >= 0.40:
                        split_rows.append(([[0, top], [w, top], [w, bottom], [0, bottom]], text, float(score)))
            groups.append(split_rows)
        return groups


@lru_cache(maxsize=1)
def load_ocr():
    return OnnxTextReader("cyrillic")


def mongolian_candidates(rows: list) -> dict[str, float]:
    """Parse ordinary 4-digit/3-letter plates without guessing missing characters."""
    reliable = [row for row in rows if float(row[2]) >= 0.60]
    candidates = [(row[1], float(row[2])) for row in reliable]
    # Join only disjoint boxes on the same text line. Whole-line OCR must never
    # be concatenated with the individual boxes covering those same characters.
    ordered = sorted(reliable, key=lambda row: min(p[0] for p in row[0]))
    # Also assemble two aligned lines: exactly four digits above three letters.
    # Include same-line fragments assembled above, e.g. "06" + "13".
    segments = [(normalize_plate(unicodedata.normalize("NFKC", row[1])), float(row[2]), row[0])
                for row in reliable]
    for start in range(len(ordered)):
        group = [ordered[start]]
        for row in ordered[start + 1:]:
            prev, box = group[-1][0], row[0]
            overlap = min(max(p[1] for p in prev), max(p[1] for p in box)) - max(min(p[1] for p in prev), min(p[1] for p in box))
            height = min(max(p[1] for p in prev) - min(p[1] for p in prev), max(p[1] for p in box) - min(p[1] for p in box))
            if min(p[0] for p in box) < max(p[0] for p in prev) or overlap < height * 0.5:
                continue
            group.append(row)
            candidates.append(("".join(item[1] for item in group), min(float(item[2]) for item in group)))
            points = [p for item in group for p in item[0]]
            left, right = min(p[0] for p in points), max(p[0] for p in points)
            top, bottom = min(p[1] for p in points), max(p[1] for p in points)
            segments.append((normalize_plate("".join(item[1] for item in group)), min(float(item[2]) for item in group),
                             [[left, top], [right, top], [right, bottom], [left, bottom]]))
    for digits, digit_score, upper in segments:
        if not re.fullmatch(r"[0-9]{4}", digits):
            continue
        for letters, letter_score, lower in segments:
            if not re.fullmatch(r"[А-ЯӨҮABCEHKMOPTXY]{3}", letters):
                continue
            upper_bottom, lower_top = max(p[1] for p in upper), min(p[1] for p in lower)
            heights = [max(p[1] for p in box) - min(p[1] for p in box) for box in (upper, lower)]
            widths = [max(p[0] for p in box) - min(p[0] for p in box) for box in (upper, lower)]
            horizontal_overlap = min(max(p[0] for p in upper), max(p[0] for p in lower)) - max(min(p[0] for p in upper), min(p[0] for p in lower))
            if (0 <= lower_top - upper_bottom <= max(heights) * 1.5
                    and horizontal_overlap >= min(widths) * 0.5):
                candidates.append((digits + letters, min(digit_score, letter_score)))
    found = {}
    for text, score in candidates:
        compact = normalize_plate(unicodedata.normalize("NFKC", text))
        match = re.fullmatch(r"(?:MGL|MGU)?([0-9]{4})(?:MGL|MGU)?([А-ЯӨҮABCEHKMOPTXY]{3})(?:MGL|MGU)?", compact)
        if match:
            letters = match.group(2).translate(str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ"))
            plate = match.group(1) + letters
            found[plate] = max(found.get(plate, 0.0), score)
    return found


def read_plate_text(image: Any, reader: Any):
    """Compare three views; score zero means an unconfirmed suggestion."""
    import cv2

    scale = max(1.0, 320 / image.shape[1])
    base = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else image
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    views = [base, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]
    all_rows, votes = [], {}
    for view in views:
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        groups = reader.read_groups(rgb) if hasattr(reader, "read_groups") else [reader.readtext(rgb, detail=1, paragraph=False)]
        view_candidates = {}
        for group in groups:
            all_rows.extend(group)
            for plate, score in mongolian_candidates(group).items():
                view_candidates[plate] = max(view_candidates.get(plate, 0.0), score)
        for plate, score in view_candidates.items():
            votes.setdefault(plate, []).append(score)
    if not votes:
        return all_rows, "", 0.0
    plate = max(votes, key=lambda value: (len(votes[value]), min(votes[value])))
    scores = votes[plate]
    # Preprocessing agreement is a consistency check, not independent proof.
    confirmed = len(votes) == 1 and len(scores) >= 2
    return all_rows, plate, min(scores) if confirmed else 0.0


def recognize_global(image: Any, alpr: Any, mode: str = "mongolia"):
    """Return detected plate crops, text, scores, and predicted country/region."""
    import cv2

    if mode != "mongolia":
        raise ValueError("Use plate_core for international recognition")
    results = []
    h, w = image.shape[:2]
    for item in alpr.predict(image):
        ocr = item.ocr
        box = item.detection.bounding_box
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = confidence_from_ocr(ocr.confidence) if ocr is not None else 0.0
        crop = image[y1:y2, x1:x2]
        if mode == "mongolia":
            _, plate, confidence = read_plate_text(crop, load_ocr())
            if not plate:
                # A tight detection box can cut the first/last character.
                pad_x, pad_y = max(2, int((x2 - x1) * 0.06)), max(2, int((y2 - y1) * 0.08))
                expanded = image[max(0, y1 - pad_y):min(h, y2 + pad_y),
                                 max(0, x1 - pad_x):min(w, x2 + pad_x)]
                _, plate, confidence = read_plate_text(expanded, load_ocr())
                if plate:
                    crop = expanded
            if not plate and ocr is not None:
                # ALPR already ran its own OCR. Reuse it only when the text
                # satisfies the selected Mongolian format; never invent digits.
                fallback = mongolian_candidates([
                    ([[0, 0], [x2 - x1, 0], [x2 - x1, y2 - y1], [0, y2 - y1]],
                     ocr.text, float(confidence_from_ocr(ocr.confidence))),
                ])
                if fallback:
                    plate, confidence = max(fallback.items(), key=lambda item: item[1])
            # Explicit user-selected format; no tiny MGL mark is required.
            results.append((crop, plate, confidence, "Mongolia" if plate else None, None))
            continue
    return results


def confidence_from_ocr(value: Any) -> float:
    """Handle scalar and per-character model scores, including missing values."""
    import math

    if value is None:
        return 0.0
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        return 0.0
    scores = [float(score) for score in values]
    if not all(math.isfinite(score) for score in scores):
        return 0.0
    return max(0.0, min(1.0, sum(scores) / len(scores)))


