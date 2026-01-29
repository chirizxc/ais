import operator
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfgen.canvas import Canvas
from starlette.staticfiles import StaticFiles
from ultralytics import YOLO

from ais import db
from ais.config import config

model = YOLO(config.yolo.path)

STATIC_DIR = Path("static")
ANNOTATED_DIR = STATIC_DIR / "annotated"
ORIGINAL_DIR = STATIC_DIR / "original"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        msg = "Could not decode the image"
        raise ValueError(msg)
    return img


def _encode_jpg(bgr_img: np.ndarray) -> bytes:
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        92,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE),
        1,
        int(cv2.IMWRITE_JPEG_OPTIMIZE),
        1,
    ]
    ok, jpg = cv2.imencode(".jpg", bgr_img, params)
    if not ok:
        msg = "Could not encode image as jpg"
        raise ValueError(msg)
    return jpg.tobytes()


def predict_details(image_bytes: bytes, model: YOLO) -> tuple:
    img = _decode_image(image_bytes)

    results = model.predict(img, conf=0.6, verbose=False)
    r0 = results[0]

    cows = 0
    names = model.names

    counts = {}
    confs = []

    if r0.boxes is not None:
        for box in r0.boxes:
            cls_id = int(box.cls)
            cls_name = names.get(cls_id, str(cls_id))
            counts[cls_name] = counts.get(cls_name, 0) + 1
            confs.append(float(box.conf))
            if cls_name == "cow":
                cows += 1

    annotated_bgr = r0.plot()
    annotated_jpg = _encode_jpg(annotated_bgr)

    return cows, annotated_jpg, counts, confs


def _draw_image_fit(  # noqa: PLR0917, PLR0913
    c: Canvas,
    img_path: Path,
    x: float,
    y: float,
    max_w: float,
    max_h: float,
) -> None:
    reader = ImageReader(str(img_path))
    iw, ih = reader.getSize()

    scale = min(max_w / iw, max_h / ih)
    w = iw * scale
    h = ih * scale

    dx = x + (max_w - w) / 2
    dy = y + (max_h - h) / 2

    c.setLineWidth(0.5)
    c.rect(x, y, max_w, max_h)

    c.drawImage(reader, dx, dy, width=w, height=h, preserveAspectRatio=True, anchor="c")


def _draw_bar_chart(  # noqa: PLR0913, PLR0914, PLR0917
    c: Canvas,
    title: str,
    items: list,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y + h + 10, title)

    c.setLineWidth(0.5)
    c.rect(x, y, w, h)

    if not items:
        c.setFont("Helvetica", 10)
        c.drawString(x + 8, y + h / 2, "No data")
        return

    max_v = max(v for _, v in items) or 1

    left_pad = 70
    right_pad = 10
    bottom_pad = 18
    top_pad = 10

    chart_x = x + left_pad
    chart_y = y + bottom_pad
    chart_w = w - left_pad - right_pad
    chart_h = h - bottom_pad - top_pad

    c.setFont("Helvetica", 8)
    steps = 4
    for i in range(steps + 1):
        yy = chart_y + (chart_h * i / steps)
        c.setLineWidth(0.25)
        c.line(chart_x, yy, chart_x + chart_w, yy)
        val = round(max_v * i / steps)
        c.drawRightString(chart_x - 6, yy - 3, str(val))

    n = len(items)
    gap = 10
    bar_w = max(10, (chart_w - gap * (n - 1)) / n)

    for i, (name, val) in enumerate(items):
        bh = (val / max_v) * chart_h
        bx = chart_x + i * (bar_w + gap)
        by = chart_y

        c.setLineWidth(0.5)
        c.rect(bx, by, bar_w, bh, fill=1)

        label = (name[:10] + "…") if len(name) > 10 else name  # noqa: PLR2004
        c.setFont("Helvetica", 8)
        c.drawCentredString(bx + bar_w / 2, y + 4, label)

        c.setFont("Helvetica", 8)
        c.drawCentredString(bx + bar_w / 2, by + bh + 3, str(val))


def _make_pdf_bytes(  # noqa: PLR0914, PLR0917, PLR0913
    record_id: int,
    cows: int,
    ts: str,
    original_path: Path,
    annotated_path: Path,
    counts: dict,
    confs: list,
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    margin = 36
    content_w = page_w - 2 * margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - margin, f"Cows report #{record_id}")

    c.setFont("Helvetica", 11)
    c.drawString(margin, page_h - margin - 18, f"Timestamp: {ts}")
    c.drawString(margin, page_h - margin - 34, f"Cows counted: {cows}")

    img_top = page_h - margin - 60
    img_box_h = 260
    gap = 16
    box_w = (content_w - gap) / 2

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, img_top + 10, "Original")
    c.drawString(margin + box_w + gap, img_top + 10, "Annotated")

    _draw_image_fit(c, original_path, margin, img_top - img_box_h, box_w, img_box_h)
    _draw_image_fit(
        c, annotated_path, margin + box_w + gap, img_top - img_box_h, box_w, img_box_h,
    )

    y = img_top - img_box_h - 26
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "YOLO summary")
    y -= 14

    total_dets = sum(counts.values()) if counts else 0
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Detections: {total_dets}")
    y -= 12
    c.drawString(
        margin, y, f"Avg confidence: {avg_conf:.3f}" if confs else "Avg confidence: N/A",
    )
    y -= 16

    top_items = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[:8]
    if top_items:
        c.setFont("Helvetica", 10)
        for name, val in top_items:
            c.drawString(margin, y, f"- {name}: {val}")
            y -= 12
    else:
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, "No detections")
        y -= 12

    chart_h = 150
    chart_y = max(margin, y - chart_h - 10)

    items = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[:6]
    _draw_bar_chart(
        c,
        "Class counts chart",
        items,
        margin,
        chart_y,
        content_w,
        chart_h,
    )

    c.showPage()
    c.save()
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await db.close_pool()


app = FastAPI(title="Cows Counter", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Cows</title>
</head>
<body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px;">
  <h1>Загрузить фото</h1>

  <form action="/upload" method="post" enctype="multipart/form-data" style="margin: 24px 0;">
    <input type="file" name="file" accept="image/*" required />
    <button type="submit" style="margin-left: 8px;">Посчитать коров</button>
  </form>

  <p><a href="/history">Посмотреть историю</a></p>
</body>
</html>
"""  # noqa: E501


@app.post("/upload", response_class=HTMLResponse)
async def upload(file: Annotated[UploadFile, File()]) -> str:
    image_bytes = await file.read()

    try:
        cows, annotated_jpg, _, _ = predict_details(image_bytes, model)
    except Exception as exc:
        return f"""
<!doctype html>
<html lang="ru">
<head><meta charset="utf-8" /><title>Ошибка</title></head>
<body style="font-family: system-ui; max-width: 720px; margin: 40px auto; padding: 0 16px;">
  <h1>Ошибка обработки</h1>
  <pre>{type(exc).__name__}: {exc}</pre>
  <p><a href="/">Назад</a></p>
</body>
</html>
"""  # noqa: E501

    filename = f"{int(time.time() * 1000)}.jpg"

    orig_rel = f"original/{filename}"
    ann_rel = f"annotated/{filename}"

    (ORIGINAL_DIR / filename).write_bytes(image_bytes)
    (ANNOTATED_DIR / filename).write_bytes(annotated_jpg)

    record_id = await db.add(cows, orig_rel, ann_rel)

    image_url = f"/static/{ann_rel}"
    pdf_url = f"/report/{record_id}"

    return f"""
<!doctype html>
<html lang="ru">
<head><meta charset="utf-8" /><title>Результат</title></head>
<body style="font-family: system-ui; max-width: 720px; margin: 40px auto; padding: 0 16px;">
  <h1>Готово</h1>
  <p>Найдено коров: <strong>{cows}</strong></p>

  <div style="margin: 16px 0;">
    <img src="{image_url}" alt="annotated" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;" />
  </div>

  <p>
    <a href="/">Загрузить ещё</a> ·
    <a href="/history">История</a> ·
    <a href="{pdf_url}">PDF-отчёт</a>
  </p>
</body>
</html>
"""  # noqa: E501


@app.get("/history", response_class=HTMLResponse)
async def history(limit: int = 50) -> str:
    rows = await db.get_history(limit=limit)

    if rows:
        body = "\n".join(
            f"""
            <tr>
              <td>{rid}</td>
              <td>{cows}</td>
              <td>{ts}</td>
              <td>{f"<img src='/static/{ann_path}' style='max-width:220px;border:1px solid #ddd;border-radius:6px;' />" if ann_path else ""}</td>
              <td>{f"<a href='/report/{rid}'>PDF</a>"}</td>
            </tr>
            """  # noqa: E501
            for rid, cows, orig_path, ann_path, ts in rows
        )
    else:
        body = "<tr><td colspan='5'>Пока пусто</td></tr>"

    return f"""
<!doctype html>
<html lang="ru">
<head><meta charset="utf-8" /><title>История</title></head>
<body style="font-family: system-ui; max-width: 980px; margin: 40px auto; padding: 0 16px;">
  <h1>История</h1>
  <p><a href="/">← Назад</a></p>

  <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr><th>ID</th><th>Cows</th><th>Timestamp</th><th>Annotated</th><th>Report</th></tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""  # noqa: E501


@app.get("/report/{record_id}")
async def report(record_id: int) -> Response:
    row = await db.get_one(record_id)
    if row is None:
        return Response("Not found", status_code=404)

    rid, cows, orig_rel, ann_rel, ts = row

    orig_path = STATIC_DIR / orig_rel
    ann_path = STATIC_DIR / ann_rel

    if not orig_path.exists() or not ann_path.exists():
        return Response("Images missing", status_code=500)

    orig_bytes = orig_path.read_bytes()
    try:
        _, _, counts, confs = predict_details(orig_bytes, model)
    except Exception as exc:
        return Response(f"YOLO error: {type(exc).__name__}: {exc}", status_code=500)

    pdf_bytes = _make_pdf_bytes(rid, cows, ts, orig_path, ann_path, counts, confs)

    headers = {
        "Content-Disposition": f'inline; filename="report_{rid}.pdf"',
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
