"""ChromaDB database setup and access."""

import chromadb
from pathlib import Path
from chromadb.config import Settings
from chromadb import Collection
from src.config import config_manager

# Singleton ChromaDB client and collection
_chroma_client = None
_chroma_collection = None

def get_chroma_path() -> Path:
    config = config_manager.config
    return Path(config.db_path) / "chroma_db"

def get_chroma_client() -> chromadb.PersistentClient:
    """Get the ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        chroma_path = get_chroma_path()
        _chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(allow_reset=True)
        )
    return _chroma_client

def get_chroma_collection() -> Collection:
    """Get the ChromaDB collection for Elumine artifacts."""
    global _chroma_collection
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = client.get_or_create_collection("elumine_artifacts")
    return _chroma_collection
