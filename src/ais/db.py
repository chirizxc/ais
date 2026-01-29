from collections.abc import Awaitable, Callable
from typing import cast

import aiosqlite
from aiosqlitepool.client import SQLiteConnectionPool
from aiosqlitepool.protocols import Connection as PoolConnection

from ais.config import config


async def sqlite_connection() -> aiosqlite.Connection:
    path = config.db.path
    conn = await aiosqlite.connect(path)
    await conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = 10000;
        PRAGMA temp_store = MEMORY;
        PRAGMA foreign_keys = ON;
        PRAGMA mmap_size = 268435456;
        PRAGMA busy_timeout = 5000;
    """)
    return conn


pool = SQLiteConnectionPool(
    connection_factory=cast(
        Callable[[], Awaitable[PoolConnection]],
        cast(object, sqlite_connection),
    ),
)


async def add(cows: int, original_path: str, annotated_path: str) -> int:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO history (cows, original_path, annotated_path) VALUES (?, ?, ?)",
            (cows, original_path, annotated_path),
        )
        await conn.commit()
        return int(cur.lastrowid)


async def get_one(record_id: int) -> tuple | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, cows, original_path, annotated_path, timestamp "
            "FROM history WHERE id = ?",
            (record_id,),
        )
        return await cur.fetchone()


async def get_history(limit: int = 100) -> list:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, cows, original_path, annotated_path, timestamp "
            "FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return await cur.fetchall()


async def close_pool() -> None:
    await pool.close()
