CREATE TABLE IF NOT EXISTS history
(
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cows           INTEGER NOT NULL,
    original_path  TEXT    NOT NULL,
    annotated_path TEXT    NOT NULL,
    timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP
)