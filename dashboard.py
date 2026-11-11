"""PlateScope's Streamlit workspace. Tables deliberately avoid pandas."""
from __future__ import annotations

import csv
import io
import re
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from plate_core import APP_TIMEZONE, ROOT, country_code, country_name, database, load_global_alpr, recognize_global, save_detection
from mongolian_ocr import recognize_global as recognize_mongolian


def dashboard_data():
    con = database()
    try:
        today = datetime.now(APP_TIMEZONE).date().isoformat()
        total, today_count, average = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(date(created_at, '+8 hours') = ?), 0), AVG(confidence) FROM detections",
            (today,),
        ).fetchone()
        countries = {country_name(row[0]) for row in con.execute("SELECT DISTINCT country FROM detections")}
        countries.discard("Тодорхойгүй")
        rows = con.execute(
            "SELECT id,created_at,plate_text,country,confidence,source FROM detections ORDER BY id DESC LIMIT 500"
        ).fetchall()
        return {"total": total, "today": today_count, "countries": len(countries), "average": average}, rows
    finally:
        con.close()


def display_date(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=APP_TIMEZONE)
        return moment.astimezone(APP_TIMEZONE).strftime("%d.%m.%Y · %H:%M")
    except (TypeError, ValueError):
        return str(value)


def markdown_cell(value: Any) -> str:
    text = escape(str(value) if value is not None else "").replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", text)


def history_markdown(rows: list[tuple[Any, ...]]) -> str:
    """Show concise registration details, keeping filenames internal."""
    lines = ["| Дугаар | Улс | Бүртгэсэн огноо | OCR оноо | Төлөв |", "| :--- | :--- | :--- | ---: | :--- |"]
    for _, created_at, plate, country, confidence, _source in rows:
        cells = (plate, f"{country_name(country)} · {country_code(country)}", display_date(created_at), f"{confidence:.1%}")
        lines.append("| " + " | ".join(markdown_cell(value) for value in cells) + " | :green-badge[Бүртгэгдсэн] |")
    return "\n".join(lines)


def history_csv(rows: list[tuple[Any, ...]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("ID", "Огноо", "Дугаар", "Улс", "Улсын код", "OCR оноо", "Төлөв"))
    for record_id, created_at, plate, country, confidence, _source in rows:
        writer.writerow((record_id, display_date(created_at), plate, country_name(country), country_code(country), f"{confidence:.1%}", "Бүртгэгдсэн"))
    return output.getvalue().encode("utf-8-sig")


def filter_history(rows: list[tuple[Any, ...]], query: str, country: str, order: str):
    term = re.sub(r"\s+", "", query).casefold()
    filtered = [row for row in rows
                if (not term or term in re.sub(r"\s+", "", row[2]).casefold())
                and (country == "Бүх улс" or country_name(row[3]) == country)]
    if order == "Хуучин нь эхэнд":
        filtered.reverse()
    elif order == "OCR оноо өндөр":
        filtered.sort(key=lambda row: row[4], reverse=True)
    return filtered


def clear_media():
    st.session_state["media"] = None
    st.session_state["recognition_results"] = []
    st.session_state["recognition_count"] = 0
    st.session_state["recognition_done"] = False
    st.session_state["recognition_error"] = None


def open_history():
    st.session_state["workspace"] = "Бүртгэлийн түүх"


@st.cache_resource(show_spinner="Таних загваруудыг бэлдэж байна…", max_entries=1)
def international_alpr():
    return load_global_alpr()


def process_media(media: dict[str, Any], preview: Any, step: int, mode: str = "international"):
    """Save on explicit recognition only; subsequent UI reruns reuse the results."""
    saved: list[dict[str, Any]] = []
    found = 0
    video_path = None
    cap = None
    st.session_state["recognition_error"] = None
    st.session_state["recognition_done"] = True

    def register(frame_results):
        nonlocal found
        for crop, plate, score, region, region_score in frame_results:
            if not plate:
                continue
            record_id = save_detection(plate, score, media["source"], country=region or "Unknown")
            saved.append({"crop": crop, "plate": plate, "score": score, "region": region,
                          "region_score": region_score, "id": record_id})
            del saved[:-10]
            found += 1

    try:
        with st.spinner("Дугаар болон улсын шинжийг уншиж байна…"):
            alpr = international_alpr()
            recognize_frame = recognize_mongolian if mode == "mongolia" else recognize_global
            if media["suffix"] in {".mp4", ".avi", ".mov"}:
                directory = ROOT / "uploads"
                directory.mkdir(exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=directory, suffix=media["suffix"], delete=False) as temporary:
                    temporary.write(media["bytes"])
                    video_path = Path(temporary.name)
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError("Видеог нээж чадсангүй.")
                total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                progress = st.progress(0.0, text="Видео боловсруулж байна…")
                index = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index % step == 0:
                        register(recognize_frame(frame, alpr))
                    index += 1
                    if index % step == 0:
                        progress.progress(min(index / total, 1.0), text=f"{found} танилт бүртгэгдлээ")
                progress.progress(1.0)
            else:
                register(recognize_frame(cv2.cvtColor(np.array(preview), cv2.COLOR_RGB2BGR), alpr))
    except Exception as exc:
        st.session_state["recognition_error"] = str(exc)
    finally:
        if cap is not None:
            cap.release()
        if video_path is not None:
            video_path.unlink(missing_ok=True)
    st.session_state["recognition_results"] = saved
    st.session_state["recognition_count"] = found
    if found:
        st.session_state["saved_notice"] = f"{found} танилт амжилттай бүртгэгдлээ."
    st.rerun()


def render_source_panel():
    with st.container(border=True):
        st.caption("01 / ЭХ ФАЙЛ")
        st.subheader("Зураг эсвэл видео", icon=":material/add_photo_alternate:")
        selection = st.radio("Таних дугаар", ["Монгол дугаар", "Гадаад дугаар"], horizontal=True, key="recognition_mode")
        mode = "mongolia" if selection == "Монгол дугаар" else "international"
        media = st.session_state["media"]
        if media is None:
            st.caption("Файлаа доорх хэсэгт чирж оруулах эсвэл сонгоно уу.")
            upload = st.file_uploader("Таних зураг эсвэл видео", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
                                      key="media_uploader", label_visibility="collapsed")
            if upload is not None:
                clear_media()
                st.session_state["media"] = {"bytes": upload.getvalue(), "source": upload.name,
                                              "suffix": Path(upload.name).suffix.lower()}
                st.rerun()
            st.caption("Зураг: JPG, PNG, BMP · Видео: MP4, AVI, MOV")
            st.space("small")
            st.markdown("**Нэг файл. Гурван үр дүн.**")
            st.caption("Улсын дугаар → Улсын таамаг → Хадгалагдсан бүртгэл")
            st.button("Дугаар таних", type="primary", icon=":material/document_scanner:", disabled=True, width="stretch")
            return

        is_video = media["suffix"] in {".mp4", ".avi", ".mov"}
        preview = None
        valid = False
        try:
            if is_video:
                st.video(media["bytes"], alt="Оруулсан автомашины видео")
            else:
                preview = ImageOps.exif_transpose(Image.open(io.BytesIO(media["bytes"]))).convert("RGB")
                st.image(preview, width="stretch", alt="Танихад сонгосон автомашины зураг")
            details = f"{media['suffix'].lstrip('.').upper()} · {len(media['bytes']) / 1024 / 1024:.2f} MB"
            if preview is not None:
                details += f" · {preview.width} × {preview.height}"
            st.caption(details)
            valid = True
        except Exception:
            st.error("Файлыг нээж чадсангүй. Өөр зураг эсвэл видео сонгоно уу.")
        st.button("Өөр файл сонгох", icon=":material/swap_horiz:", on_click=clear_media, key="replace_media")
        step = st.slider("Видео шалгах алхам", 1, 60, 10, key="video_step", help="Хэдэн кадр тутамд дугаар хайхыг сонгоно.") if is_video else 10
        if st.button("Видео таних" if is_video else "Дугаар таних", type="primary", width="stretch",
                     icon=":material/document_scanner:", key="recognize_media", disabled=not valid):
            process_media(media, preview, step, mode=mode)


def render_results_panel():
    with st.container(border=True):
        st.caption("02 / ТАНИЛТЫН ҮР ДҮН")
        st.subheader("Танилтын мэдээлэл", icon=":material/fact_check:")
        results = st.session_state["recognition_results"]
        error = st.session_state["recognition_error"]
        if error:
            st.error("Файлыг бүрэн боловсруулж чадсангүй.", icon=":material/error_outline:")
            with st.expander("Алдааны дэлгэрэнгүй"):
                st.text(error)
        if results:
            st.success(f"{st.session_state['recognition_count']} танилт бүртгэгдлээ.", icon=":material/check_circle:")
            for result in results:
                with st.container(border=True):
                    st.caption("УЛСЫН ДУГААР")
                    st.subheader(result["plate"], anchor=False)
                    st.markdown(f":material/public: **{markdown_cell(country_name(result['region']))}** :blue-badge[{country_code(result['region'])}]")
                    st.image(cv2.cvtColor(result["crop"], cv2.COLOR_BGR2RGB), width="stretch", alt="Илрүүлсэн улсын дугаар")
                    st.progress(min(1.0, max(0.0, result["score"])), text=f"OCR оноо · {result['score']:.1%}")
                    st.caption(f"Бүртгэл #{result['id']:05d} · Амжилттай хадгалагдсан")
            if st.session_state["recognition_count"] > len(results):
                st.caption("Энд сүүлийн 10 танилтыг харуулж байна. Бүх танилт бүртгэлийн түүхэнд хадгалагдсан.")
        elif st.session_state["recognition_done"] and not error:
            st.warning("Дугаар олдсонгүй. Илүү тод, ойроос авсан зураг сонгоод дахин оролдоорой.")
        elif not error:
            st.space("medium")
            st.markdown("# :material/center_focus_strong:")
            st.markdown("**Үр дүн энд харагдана**")
            st.caption("01  Зураг эсвэл видеогоо сонгоно.")
            st.caption("02  Дугаар таних товчийг дарна.")
            st.caption("03  Дугаар, улсын таамаг бүртгэгдэнэ.")
            st.badge("Файл хүлээж байна", color="gray", icon=":material/hourglass_empty:")


def render_history(rows):
    search, country_filter, sort = st.columns([2, 1, 1])
    with search:
        query = st.text_input("Дугаар хайх", placeholder="Жишээ нь: 9638", icon=":material/search:", key="history_search") or ""
    with country_filter:
        selected_country = st.selectbox("Улсаар шүүх", ["Бүх улс"] + sorted({country_name(row[3]) for row in rows}), key="history_country")
    with sort:
        order = st.selectbox("Эрэмбэ", ["Шинэ нь эхэнд", "Хуучин нь эхэнд", "OCR оноо өндөр"], key="history_order")
    filtered = filter_history(rows, query, selected_country or "Бүх улс", order or "Шинэ нь эхэнд")
    count_area, download_area = st.columns([3, 1], vertical_alignment="center")
    with count_area:
        st.caption(f"Сүүлийн 500 бүртгэлээс {len(filtered)} үр дүн олдлоо.")
    with download_area:
        st.download_button("CSV татах", history_csv(filtered), file_name="plates-history.csv", mime="text/csv",
                           icon=":material/download:", disabled=not filtered, width="stretch", on_click="ignore")
    if filtered:
        size_col, page_col = st.columns([1, 1])
        with size_col:
            per_page = st.selectbox("Нэг хуудсанд", [10, 25, 50], key="history_page_size") or 10
        pages = max(1, (len(filtered) + per_page - 1) // per_page)
        st.session_state["history_page"] = min(int(st.session_state.get("history_page", 1)), pages)
        with page_col:
            number = int(st.number_input("Хуудас", min_value=1, max_value=pages, step=1, key="history_page"))
        with st.container(border=True):
            st.markdown(history_markdown(filtered[(number - 1) * per_page:number * per_page]))
        st.caption(f"{number} / {pages} хуудас")
    else:
        st.info("Тохирох бүртгэл олдсонгүй. Хайлтын үг эсвэл улсын шүүлтүүрээ өөрчилнө үү.", icon=":material/search_off:")


def render_help():
    steps = st.columns(3, gap="medium")
    for column, title, description in zip(steps, ("01 · Файлаа сонго", "02 · Дугаарыг таниул", "03 · Бүртгэлээ ашигла"),
                                          ("Дугаар нь тод харагдах зураг эсвэл видеогоо оруулна.",
                                           "Дугаар таних товчийг дараад боловсруулж дуустал хүлээнэ.",
                                           "Үр дүн автоматаар хадгалагдана. Түүхээс хайж, CSV хэлбэрээр татаж болно.")):
        with column, st.container(border=True):
            st.subheader(title)
            st.caption(description)
    with st.container(border=True):
        st.subheader("Илүү сайн зураг, илүү сайн танилт", icon=":material/tips_and_updates:")
        st.markdown("- Дугаарыг бүтнээр нь, урдаас харагдахаар авна.\n- Хэт хол, хөдөлгөөнтэй, хурц тусгалтай зургаас зайлсхий.\n- Видео урт байх тусам боловсруулах хугацаа нэмэгдэнэ.")
    with st.expander("Ямар улсын дугаар дэмжих вэ?", icon=":material/public:"):
        st.markdown("**Япон, Өмнөд Солонгос, АНУ, Монгол, Герман** болон global загварын бусад улсын дугаарыг таамаглана.")
        st.caption("Бүх улсын бүх формат, хуучин болон тусгай дугаарыг алдаагүй танина гэсэн баталгаа байхгүй.")
    with st.expander("OCR оноо юуг илэрхийлэх вэ?", icon=":material/analytics:"):
        st.caption("Текст унших загварын дотоод оноо юм. Өндөр оноотой байсан ч үсэг, цифр эсвэл улсыг андуурч болно.")


def render_dashboard():
    st.set_page_config(page_title="PlateScope · Улсын дугаар таних", page_icon=":material/document_scanner:",
                       layout="wide", initial_sidebar_state="expanded")
    defaults: dict[str, Any] = {"workspace": "Таних самбар", "media": None, "recognition_results": [],
                                "recognition_count": 0, "recognition_done": False, "recognition_error": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    stats, rows = dashboard_data()
    with st.sidebar:
        st.markdown("## :material/document_scanner: PlateScope")
        st.caption("Дугаар бүрийн ард — тодорхой мэдээлэл.")
        st.space("medium")
        labels = {"Таних самбар": ":material/center_focus_strong:  Таних самбар",
                  "Бүртгэлийн түүх": ":material/receipt_long:  Бүртгэлийн түүх", "Тусламж": ":material/help_outline:  Тусламж"}
        st.radio("Ажлын хэсэг", list(labels), key="workspace", format_func=lambda value: labels[value], label_visibility="collapsed")
        st.space("large")
        st.caption("ОНЦЛОХ УЛСУУД")
        st.markdown("🇲🇳 Монгол\n\n🇯🇵 Япон\n\n🇰🇷 Өмнөд Солонгос\n\n🇺🇸 АНУ\n\n🇩🇪 Герман")
        st.caption("Бусад дэмжигдсэн улсын дугаарыг мөн таамаглана.")
        st.space("large")
        with st.container(border=True):
            st.markdown(":material/lightbulb: **Илүү сайн танихын тулд**")
            st.caption("Дугаар бүтнээрээ, тод харагдах, урдаас авсан зураг сонгоорой.")
        st.caption("PlateScope · Танилт ба бүртгэл")
    page = st.session_state["workspace"]
    if notice := st.session_state.pop("saved_notice", None):
        st.toast(notice, icon=":material/check_circle:")
    with st.container(horizontal=True, vertical_alignment="center"):
        with st.container(width="stretch"):
            st.caption("PLATESCOPE / АЖЛЫН ТАЛБАР")
            st.title({"Таних самбар": "Улсын дугаар таних", "Бүртгэлийн түүх": "Бүртгэлийн түүх", "Тусламж": "Хэрхэн ашиглах вэ?"}[page])
            st.caption("Зураг оруулаад дугаар, улсын таамгийг нэг дор хараарай." if page == "Таних самбар"
                       else "Танилтын мэдээллээ хайж, шүүж, хэрэгтэй үедээ татаж аваарай." if page == "Бүртгэлийн түүх"
                       else "Хэдхэн алхмаар зураг, видеогоо бүртгэл болгоно.")
        st.badge("Ашиглахад бэлэн", icon=":material/check_circle:", color="green")
    if page != "Тусламж":
        with st.grid(4, min_column_width=170, gap="medium"):
            st.metric("Нийт бүртгэл", f"{stats['total']:,}", icon=":material/database:", border=True)
            st.metric("Өнөөдрийн танилт", f"{stats['today']:,}", icon=":material/today:", border=True)
            st.metric("Бүртгэлтэй улс", stats["countries"], icon=":material/public:", border=True)
            st.metric("Дундаж OCR оноо", f"{stats['average']:.1%}" if stats["average"] is not None else "—",
                      icon=":material/analytics:", border=True, help="OCR загварын оноо. Бодит нарийвчлалын баталгаа биш.")
        st.space("small")
    if page == "Таних самбар":
        with st.grid(2, min_column_width=320, gap="large"):
            render_source_panel()
            render_results_panel()
        st.space("small")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            with st.container(width="stretch"):
                st.subheader("Сүүлийн бүртгэлүүд", icon=":material/history:")
                st.caption("Таны хамгийн сүүлд хадгалсан танилтууд.")
            st.button("Бүх бүртгэл", icon=":material/arrow_forward:", on_click=open_history, key="open_history")
        with st.container(border=True):
            if rows:
                st.markdown(history_markdown(rows[:5]))
            else:
                st.caption("Эхний зургаа таниулсны дараа бүртгэлүүд энд харагдана.")
    elif page == "Бүртгэлийн түүх":
        render_history(rows)
    else:
        render_help()
