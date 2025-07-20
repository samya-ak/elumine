# Makefile for Elumine project

.PHONY: migrate sqlite sqlite-tables sqlite-columns sqlite-data

# Get database path
DB_PATH := $(shell python -c 'from src.db.sqlite import get_db_path; print(str(get_db_path()))')

# Run all database migrations
migrate:
	python scripts/migrate.py

# Open SQLite shell for the Elumine metadata database
sqlite:
	sqlite3 $(DB_PATH)

# List all tables in the SQLite database
sqlite-tables:
	echo ".tables" | sqlite3 $(DB_PATH)

# Show columns for a table (usage: make sqlite-columns TABLE=artifacts)
sqlite-columns:
	echo ".schema $(TABLE)" | sqlite3 $(DB_PATH)

# Show all data from a table (usage: make sqlite-data TABLE=artifacts)
sqlite-data:
	echo "SELECT * FROM $(TABLE);" | sqlite3 -header -column $(DB_PATH)
