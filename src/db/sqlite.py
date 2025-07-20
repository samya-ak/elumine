"""SQLite database setup and access using sqlite-utils."""

from pathlib import Path
from sqlite_utils import Database
from src.config import config_manager


def get_db_path() -> Path:
    config = config_manager.config
    return Path(config.db_path) / "elumine_metadata.db"

# Singleton DB instance
_db_instance = None

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(str(get_db_path()))
    return _db_instance
