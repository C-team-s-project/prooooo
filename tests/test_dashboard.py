"""UI regressions: clean history, upload/save, reruns, and navigation."""
from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image
from streamlit.testing.v1 import AppTest

import dashboard
import plate_core


class DashboardTests(unittest.TestCase):
    def setUp(self):
        argv_patch = patch("sys.argv", ["aajil.py"])
        argv_patch.start()
        self.addCleanup(argv_patch.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = Path(self.directory.name) / "history.db"
        self.db_patch = patch.object(plate_core, "DB_PATH", self.db_path)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.source = "aedDBLEQZ7JFFgkjK5_" + "unreadable_filename_" * 12 + ".jpg"
        plate_core.save_detection("9638УЕН", .946, self.source, country="Mongolia")
        plate_core.save_detection("3SAM123", .999, self.source, country="United States")

    def app(self):
        return AppTest.from_file(str(plate_core.ROOT / "aajil.py"), default_timeout=20).run()

    def count(self):
        with closing(sqlite3.connect(self.db_path)) as con:
            return con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    def image_bytes(self):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 160), "white").save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_history_hides_filenames_and_escapes_values(self):
        stats, rows = dashboard.dashboard_data()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["today"], 2)
        text = dashboard.history_markdown(rows)
        self.assertNotIn(self.source, text)
        self.assertNotIn("Эх сурвалж", text)
        self.assertIn("Бүртгэгдсэн", text)
        self.assertIn("Монгол", text)
        self.assertIn("Монгол · MGL", text)
        self.assertIn("АНУ · USA", text)
        self.assertNotIn(self.source, dashboard.history_csv(rows).decode("utf-8-sig"))
        unsafe = [(1, "2026-11-09T12:00:00+08:00", "<script>a|b</script>", "Unknown", .5, self.source)]
        rendered = dashboard.history_markdown(unsafe)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn(r"a\|b", rendered)

    def test_search_country_filter_and_help_navigation(self):
        at = self.app()
        self.assertFalse(at.exception)
        at.button(key="open_history").click().run()
        self.assertFalse(at.exception)
        at.text_input(key="history_search").set_value("9638").run()
        tables = [node.value for node in at.markdown if "| Дугаар |" in node.value]
        self.assertEqual(len(tables), 1)
        self.assertIn("9638УЕН", tables[0])
        self.assertNotIn("3SAM123", tables[0])
        at.selectbox(key="history_country").select("АНУ").run()
        self.assertTrue(any("Тохирох бүртгэл" in node.value for node in at.info))
        at.text_input(key="history_search").set_value("").run()
        self.assertTrue(any("3SAM123" in node.value for node in at.markdown))
        at.radio(key="workspace").set_value("Тусламж").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Хэрхэн ашиглах вэ?")

    def test_upload_save_and_rerun_do_not_duplicate_registration(self):
        at = self.app()
        at.file_uploader[0].set_value((self.source, self.image_bytes(), "image/jpeg")).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.file_uploader), 0)
        self.assertFalse(any(self.source in str(node.value) for node in at.caption))
        result = [(np.zeros((70, 250, 3), dtype=np.uint8), "1234УБА", .98, "Mongolia", None)]
        with patch.object(dashboard, "international_alpr", return_value=object()), \
                patch.object(dashboard, "recognize_global", return_value=result):
            at.button(key="recognize_media").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(self.count(), 3)
        self.assertTrue(any("бүртгэгдлээ" in node.value for node in at.success))
        self.assertTrue(any(node.value == "1234УБА" for node in at.subheader))
        self.assertTrue(any(":blue-badge[MGL]" in node.value for node in at.markdown))
        at.run()
        self.assertEqual(self.count(), 3)
        self.assertTrue(at.success)
        at.button(key="replace_media").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.file_uploader), 1)
        self.assertFalse(at.success)

    def test_invalid_image_is_handled(self):
        at = self.app()
        at.file_uploader[0].set_value((self.source, b"not an image", "image/jpeg")).run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertTrue(at.button(key="recognize_media").disabled)
        self.assertEqual(self.count(), 2)

    def test_no_detection_does_not_claim_saved(self):
        at = self.app()
        at.file_uploader[0].set_value((self.source, self.image_bytes(), "image/jpeg")).run()
        with patch.object(dashboard, "international_alpr", return_value=object()), \
                patch.object(dashboard, "recognize_global", return_value=[]):
            at.button(key="recognize_media").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.warning)
        self.assertFalse(at.success)
        self.assertEqual(self.count(), 2)

    def test_history_pagination_clamps_after_search(self):
        for i in range(15):
            plate_core.save_detection(f"B{i:04}", .90, self.source, country="Germany")
        at = self.app()
        at.radio(key="workspace").set_value("Бүртгэлийн түүх").run()
        at.number_input(key="history_page").set_value(2).run()
        at.text_input(key="history_search").set_value("9638").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.number_input(key="history_page").value, 1)

    def test_video_saves_sampled_frame_and_removes_temporary_upload(self):
        path = Path(self.directory.name) / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (320, 160))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(3):
                writer.write(np.zeros((160, 320, 3), dtype=np.uint8))
        finally:
            writer.release()
        before = set((plate_core.ROOT / "uploads").glob("*"))
        at = self.app()
        at.file_uploader[0].set_value(("long_video_name.avi", path.read_bytes(), "video/x-msvideo")).run()
        result = [(np.zeros((70, 250, 3), dtype=np.uint8), "B1234", .95, "Germany", .9)]
        with patch.object(dashboard, "international_alpr", return_value=object()), \
                patch.object(dashboard, "recognize_global", return_value=result) as recognize:
            at.button(key="recognize_media").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(recognize.call_count, 1)
        self.assertEqual(self.count(), 3)
        self.assertTrue(at.success)
        self.assertEqual(set((plate_core.ROOT / "uploads").glob("*")), before)


if __name__ == "__main__":
    unittest.main()
