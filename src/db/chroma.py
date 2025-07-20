"""ChromaDB path utilities for LangChain integration."""

from pathlib import Path
from src.config import config_manager

def get_chroma_path() -> Path:
    """Get the path where ChromaDB data will be stored."""
    config = config_manager.config
    return Path(config.db_path) / "chroma_db"
