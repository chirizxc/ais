1. Скачайте [эту*](https://www.kaggle.com/code/stpeteishii/cows-yolov8-train-and-predict/output?select=yolov8x.pt) YOLO
   модель.

2. Положите ее в папку `models/`.

3. Создайте и активируйте виртуальное окружение:

```bash
# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
py -m venv .venv
.venv\scripts\activate
```

4. Установите зависимости и само приложение.

```bash
# uv
uv pip install .

# pip
pip install .
```

5. Создайте базу данных.

```bash
python scripts/init_db.py
```

6. Поменяйте название конфига с `example.config.toml` на `config.toml` и заполните его своими данными.

7. Запустите приложение.
```bash
uvicorn ais.main:app --reload
```

8. Откройте http://127.0.0.1:8000/ в браузере.