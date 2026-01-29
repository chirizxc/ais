from dataclasses import dataclass
from pathlib import Path

import toml_rs
from adaptix import DebugTrail, Retort

from ais.common import init_only


@dataclass(**init_only, frozen=True)
class SQLiteConfig:
    path: str


@dataclass(**init_only, frozen=True)
class YOLOConfig:
    path: str


@dataclass(**init_only, frozen=True)
class Config:
    db: SQLiteConfig
    yolo: YOLOConfig


_retort = Retort(
    debug_trail=DebugTrail.DISABLE,
)


def load_config() -> Config:
    load_toml = toml_rs.loads(
        Path(".config/config.toml").read_text("utf-8"),
        toml_version="1.1.0",
    )
    return _retort.load(load_toml, Config)


config = load_config()
