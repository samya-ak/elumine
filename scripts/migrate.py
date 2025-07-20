"""Run all SQL migrations in src/db/migrations using sqlite-utils."""

import os
from pathlib import Path
from sqlite_utils import Database
from src.config import config_manager
from src.db.sqlite import get_db_path

MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "db" / "migrations"

def run_migrations():
    db_path = get_db_path()
    db = Database(str(db_path))
    for migration_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        with open(migration_file, "r") as f:
            sql = f.read()
            db.conn.executescript(sql)
    print(f"Migrations applied to {db_path}")

if __name__ == "__main__":
    run_migrations()
