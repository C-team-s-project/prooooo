from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import plate_core as core


class DarkPlateTests(TestCase):
    def test_contrast_preserves_dimensions_and_input(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[10:60, 30:80] = 25
        before = image.copy()
        result = core.enhance_plate(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        np.testing.assert_array_equal(image, before)

    def test_split_lines_keep_vertical_coordinates(self):
        from rapidocr.ch_ppocr_rec import TextRecOutput
        from rapidocr.utils.output import RapidOCROutput

        reader = core.OnnxTextReader.__new__(core.OnnxTextReader)
        reader.language = 'cyrillic'
        def engine(image, use_det, use_cls):
            if use_det:
                return SimpleNamespace(boxes=None, txts=None, scores=None)
            return SimpleNamespace(txts=['0613'], scores=[.9])
        reader.engine = engine
        rows = reader.readtext(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertTrue(any(box[0][1] == 38 for box, _, _ in rows))
        self.assertTrue(any(box[2][1] == 42 for box, _, _ in rows))

    def test_unreadable_cyrillic_does_not_keep_uk_guess(self):
        from test_plate_core import CountryEvidenceTests, row
        result = CountryEvidenceTests().predict(
            cyrillic_rows=[row('УНИ', .8)], plate='DG13YHW',
            region='United Kingdom', region_score=.99)
        self.assertEqual(result[3], 'Unknown')
