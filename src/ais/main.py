import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from ultralytics import YOLO

from ais import db
from ais.config import config

model = YOLO(config.yolo.path)

STATIC_DIR: Path = Path("static")
ANNOTATED_DIR: Path = STATIC_DIR / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def count_cows(image_bytes: bytes, model: YOLO) -> tuple[int, bytes]:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        msg = "Could not decode the image"
        raise ValueError(msg)

    results = model.predict(img, conf=0.6, verbose=False)
    r0 = results[0]

    cows = 0
    names = model.names

    if r0.boxes is not None:
        for box in r0.boxes:
            cls_id: int = int(box.cls)
            if names.get(cls_id) == "cow":
                cows += 1

    annotated_bgr = r0.plot()

    ok, jpg = cv2.imencode(".jpg", annotated_bgr)
    if not ok:
        msg = "Could not encode annotated image"
        raise ValueError(msg)

    return cows, jpg.tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await db.close_pool()


app = FastAPI(
    title="Cows Counter",
    lifespan=lifespan,
)

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
        cows, annotated_jpg = count_cows(image_bytes, model)
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
    rel_path = f"annotated/{filename}"
    full_path = ANNOTATED_DIR / filename
    full_path.write_bytes(annotated_jpg)

    await db.add(cows, rel_path)

    image_url: str = f"/static/{rel_path}"

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

  <p><a href="/">Загрузить ещё</a> · <a href="/history">История</a></p>
</body>
</html>
"""  # noqa: E501


@app.get("/history", response_class=HTMLResponse)
async def history(limit: int = 50) -> str:
    rows = await db.get_history(limit=limit)

    if rows:
        body: str = "\n".join(
            f"""
            <tr>
              <td>{rid}</td>
              <td>{cows}</td>
              <td>{ts}</td>
              <td>{f"<img src='/static/{path}' style='max-width:220px;border:1px solid #ddd;border-radius:6px;' />" if path else ""}</td>
            </tr>
            """  # noqa: E501
            for rid, cows, path, ts in rows
        )
    else:
        body = "<tr><td colspan='4'>Пока пусто</td></tr>"

    return f"""
<!doctype html>
<html lang="ru">
<head><meta charset="utf-8" /><title>История</title></head>
<body style="font-family: system-ui; max-width: 980px; margin: 40px auto; padding: 0 16px;">
  <h1>История</h1>
  <p><a href="/">← Назад</a></p>

  <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr><th>ID</th><th>Cows</th><th>Timestamp</th><th>Image</th></tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""  # noqa: E501
