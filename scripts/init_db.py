import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT_SQL = ROOT / ".config" / "init.sql"
DB = ROOT / Path("db.sqlite3")


def main() -> None:
    conn = sqlite3.connect(DB)
    with INIT_SQL.open("r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    print("DB initialized")


if __name__ == "__main__":
    main()
