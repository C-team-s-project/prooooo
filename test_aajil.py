"""Regression checks; run: python -m unittest test_aajil -v."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import mongolian_ocr as aajil


def row(text, score=0.95, x=0, right=200):
    return ([[x, 0], [right, 0], [right, 40], [x, 40]], text, score)


class Reader:
    def __init__(self, views):
        self.views = iter(views)

    def read_groups(self, image):
        return next(self.views)


class MongolianRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((50, 220, 3), dtype=np.uint8)

    def test_full_line_and_split_boxes_do_not_duplicate_plate(self):
        rows = [row("9638", x=0, right=80), row("MGL", x=82, right=110),
                row("УЕН", x=120, right=200), row("9638УЕН")]
        self.assertEqual(aajil.mongolian_candidates(rows), {"9638УЕН": 0.95})

    def test_strict_format_preserves_leading_zero_and_converts_lookalikes(self):
        self.assertEqual(aajil.mongolian_candidates([row("００１２-УBX")]), {"0012УВХ": 0.95})
        for text in ("123УЕН", "12345УЕН", "1234УЕ", "1234УЕНА", "1234QWE", "1234УЕ8"):
            with self.subTest(text=text):
                self.assertEqual(aajil.mongolian_candidates([row(text)]), {})

    def test_separate_text_lines_cannot_form_plate(self):
        letters = ([[90, 50], [200, 50], [200, 90], [90, 90]], "УЕН", 0.95)
        self.assertEqual(aajil.mongolian_candidates([row("9638", right=80), letters]), {})

    def test_aligned_two_line_plate_preserves_leading_zero(self):
        digits = row("0613", right=200)
        letters = ([[20, 50], [180, 50], [180, 90], [20, 90]], "УНИ", 0.92)
        self.assertEqual(aajil.mongolian_candidates([letters, digits]), {"0613УНИ": 0.92})

    def test_two_line_plate_with_split_digit_boxes(self):
        letters = ([[20, 50], [180, 50], [180, 90], [20, 90]], "УНИ", 0.92)
        rows = [row("06", right=80), row("13", x=90, right=200), letters]
        self.assertEqual(aajil.mongolian_candidates(rows), {"0613УНИ": 0.92})

    def test_letters_above_digits_are_not_joined(self):
        digits = ([[0, 50], [200, 50], [200, 90], [0, 90]], "0613", 0.95)
        self.assertEqual(aajil.mongolian_candidates([row("УНИ"), digits]), {})

    def test_square_crop_ocr_reads_both_halves(self):
        reader = aajil.OnnxTextReader.__new__(aajil.OnnxTextReader)
        reader.language = "cyrillic"
        responses = iter([
            SimpleNamespace(boxes=None, txts=None, scores=None),
            SimpleNamespace(txts=[], scores=[]),
            SimpleNamespace(txts=["0613"], scores=[0.95]),
            SimpleNamespace(txts=["УНИ"], scores=[0.92]),
        ])
        shapes = []
        def engine(image, **kwargs):
            shapes.append(image.shape[:2])
            return next(responses)
        reader.engine = engine
        groups = reader.read_groups(np.zeros((100, 160, 3), dtype=np.uint8))
        self.assertEqual(shapes[-2:], [(50, 160), (50, 160)])
        self.assertEqual(aajil.mongolian_candidates(groups[-1]), {"0613УНИ": 0.92})

    def test_marker_not_required_and_two_views_agree(self):
        reader = Reader([[[row("9638УЕН")]], [[row("9638УЕН", 0.91)]], [[]]])
        _, plate, score = aajil.read_plate_text(self.image, reader)
        self.assertEqual(plate, "9638УЕН")
        self.assertAlmostEqual(score, 0.91)

    def test_alternative_ocr_passes_in_one_view_are_only_one_vote(self):
        reader = Reader([[[row("9638УЕН")], [row("9638УЕН")]], [[]], [[]]])
        _, plate, score = aajil.read_plate_text(self.image, reader)
        self.assertEqual(plate, "9638УЕН")
        self.assertEqual(score, 0.0)

    def test_conflicting_high_scores_remain_unconfirmed(self):
        reader = Reader([[[row("9638УЕН")]], [[row("9638УЕН")]], [[row("9639УЕН", 0.99)]]])
        _, _, score = aajil.read_plate_text(self.image, reader)
        self.assertEqual(score, 0.0)

    def test_mongolian_mode_skips_foreign_readers_and_global_text(self):
        item = SimpleNamespace(ocr=SimpleNamespace(confidence=0.99, text="WRONG"),
                               detection=SimpleNamespace(bounding_box=SimpleNamespace(x1=0, y1=0, x2=220, y2=50)))
        alpr = SimpleNamespace(predict=lambda image: [item])
        reader = Reader([[[row("9638УЕН")]]] * 3)
        with patch.object(aajil, "load_ocr", return_value=reader), \
                patch.object(aajil, "load_japanese_ocr", side_effect=AssertionError("foreign OCR loaded"), create=True), \
                patch.object(aajil, "load_korean_ocr", side_effect=AssertionError("foreign OCR loaded"), create=True):
            result = aajil.recognize_global(self.image, alpr, mode="mongolia")
        self.assertEqual(result[0][1:], ("9638УЕН", 0.95, "Mongolia", None))

    def test_failed_local_ocr_reuses_valid_global_text(self):
        item = SimpleNamespace(ocr=SimpleNamespace(confidence=np.array([0.94, 0.96]), text="0012YHX"),
                               detection=SimpleNamespace(bounding_box=SimpleNamespace(x1=10, y1=10, x2=200, y2=40)))
        alpr = SimpleNamespace(predict=lambda image: [item])
        with patch.object(aajil, "load_ocr", return_value=object()), \
                patch.object(aajil, "read_plate_text", return_value=([], "", 0.0)) as read:
            result = aajil.recognize_global(self.image, alpr, mode="mongolia")
        self.assertEqual(read.call_count, 2)
        self.assertEqual(result[0][1], "0012УНХ")
        self.assertAlmostEqual(result[0][2], 0.95)

    def test_expanded_crop_retry_and_edge_clamping(self):
        item = SimpleNamespace(ocr=None, detection=SimpleNamespace(
            bounding_box=SimpleNamespace(x1=0, y1=0, x2=210, y2=45)))
        alpr = SimpleNamespace(predict=lambda image: [item])
        with patch.object(aajil, "load_ocr", return_value=object()), \
                patch.object(aajil, "read_plate_text", side_effect=[([], "", 0.0), ([], "0613УНИ", 0.9)]) as read:
            result = aajil.recognize_global(self.image, alpr, mode="mongolia")
        self.assertEqual(read.call_args_list[1].args[0].shape, (48, 220, 3))
        self.assertEqual(result[0][1], "0613УНИ")

    def test_missing_and_nonfinite_confidence(self):
        for value in (None, [], float("nan"), np.array([float("inf")])):
            self.assertEqual(aajil.confidence_from_ocr(value), 0.0)


if __name__ == "__main__":
    unittest.main()
