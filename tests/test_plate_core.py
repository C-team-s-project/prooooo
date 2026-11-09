"""Country decisions must use script evidence, even against a high global score."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

import plate_core as core


def row(text, score=.98, y=0):
    return ([[0, y], [380, y], [380, y + 60], [0, y + 60]], text, score)


class CountryEvidenceTests(TestCase):
    def predict(self, japanese_rows=(), korean_rows=(), cyrillic_rows=(), plate="4750", region="United States", region_score=.994):
        item = SimpleNamespace(
            detection=SimpleNamespace(bounding_box=SimpleNamespace(x1=0, y1=0, x2=384, y2=204)),
            ocr=SimpleNamespace(text=plate, confidence=.999, region=region, region_confidence=region_score),
        )
        alpr = SimpleNamespace(predict=Mock(return_value=[item]))
        parsed = core.mongolian_plate_from_rows(list(cyrillic_rows))
        with patch.object(core, "load_japanese_ocr", return_value=SimpleNamespace(readtext=Mock(return_value=list(japanese_rows)))), \
                patch.object(core, "load_korean_ocr", return_value=SimpleNamespace(readtext=Mock(return_value=list(korean_rows)))), \
                patch.object(core, "load_ocr", return_value=object()), \
                patch.object(core, "read_plate_text", return_value=(list(cyrillic_rows), parsed[0] if parsed else "", parsed[1] if parsed else 0)):
            return core.recognize_global(np.zeros((204, 384, 3), dtype=np.uint8), alpr)[0]

    def test_recovered_kana_overrides_incorrect_high_usa_score(self):
        # Original photo: detector read the kana as Latin n and global OCR said
        # USA. The padded lower-line retry recovers the actual Japanese text.
        rows = [row("倉敷580"), row("n", .958, 94), row("47-50", .999, 81), row("わ47-50", .964, 81)]
        _, plate, _, region, region_score = self.predict(rows)
        self.assertEqual(plate, "倉敷 580 わ 47-50")
        self.assertEqual(region, "Japan")
        self.assertEqual(core.country_name(region), "Япон")
        self.assertIsNone(region_score)

    def test_digits_alone_do_not_prove_japan(self):
        self.assertIsNone(core.japanese_plate_from_rows([row("47-50")]))
        self.assertIsNone(core.japanese_plate_from_rows([row("わ47-50")]))

    def test_heading_without_kana_remains_unknown(self):
        result = self.predict([row("倉敷580"), row("47-50", y=90)])
        self.assertEqual(result[3], "Unknown")

    def test_conflicting_japanese_and_korean_evidence_remains_unknown(self):
        result = self.predict([row("倉敷580"), row("わ47-50", y=90)], [row("12가3456")])
        self.assertEqual(result[3], "Unknown")

    def test_ordinary_latin_ocr_keeps_global_country(self):
        result = self.predict([row("4750")], [row("4750")])
        self.assertEqual(result[3], "United States")

    def test_mgl_overrides_high_uk_score_and_recovers_two_line_letters(self):
        result = self.predict(japanese_rows=[row("MGL")],
                              cyrillic_rows=[row("Y6P", .602), row("MGL", .95), row("17422", .75)],
                              plate="1742YEP", region="United Kingdom", region_score=.988)
        self.assertEqual(result[1], "1742УБР")
        self.assertEqual(result[3], "Mongolia")
        self.assertIsNone(result[4])

    def test_country_marker_still_works_when_serial_is_unreadable(self):
        result = self.predict(japanese_rows=[row("MGL")], plate="I742YEP", region="United Kingdom")
        self.assertEqual(result[3], "Mongolia")
        self.assertEqual(result[1], "I742YEP")

    def test_country_code_labels_are_distinct(self):
        for region, expected in (("MGL", "MGL"), ("Mongolia", "MGL"), ("Japan", "JPN"),
                                 ("United States", "USA"), ("United Kingdom", "GBR"),
                                 ("South Korea", "KOR"), ("Germany", "DEU")):
            with self.subTest(region=region):
                self.assertEqual(core.country_code(region), expected)
        self.assertEqual(core.country_name("MGL"), "Монгол")

    def test_jpn_and_usa_markers_override_incorrect_global_country(self):
        for marker, expected in (("JPN", "Japan"), ("USA", "United States")):
            with self.subTest(marker=marker):
                result = self.predict(japanese_rows=[row(marker)], region="United Kingdom")
                self.assertEqual(result[3], expected)

    def test_marker_cannot_be_a_substring_or_unreliable_ocr(self):
        self.assertFalse(core.country_marks_from_rows([row("ABMGL123"), row("MGL", .5)]))

    def test_conflicting_country_markers_remain_unknown(self):
        self.assertEqual(self.predict(japanese_rows=[row("MGL"), row("USA")])[3], "Unknown")

    def test_four_digits_three_latin_letters_do_not_force_mongolia(self):
        result = self.predict(cyrillic_rows=[row("1742ABC")], plate="1742ABC", region="United Kingdom")
        self.assertEqual(result[3], "United Kingdom")

    def test_native_cyrillic_layout_is_used_when_marker_is_too_small(self):
        result = self.predict(cyrillic_rows=[row("8804ДОУ", .675)], plate="8804DOY", region="Armenia", region_score=.38)
        self.assertEqual(result[1], "8804ДОУ")
        self.assertEqual(result[3], "Mongolia")

    def test_weak_global_guess_is_not_presented_as_a_known_country(self):
        self.assertEqual(self.predict(region="Spain", region_score=.60)[3], "Unknown")

    def test_mgl_not_appended_to_split_registration(self):
        parsed = core.mongolian_plate_from_rows([row("17 42"), row("УБР", y=80), row("MGL", y=80)])
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0], "1742УБР")
