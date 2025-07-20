"""Configuration management for Elumine."""

from pathlib import Path
from typing import Optional
import json
from pydantic import BaseModel, Field


class ElumineConfig(BaseModel):
    """Configuration model for Elumine."""

    transcriptions_path: Path = Field(
        default_factory=lambda: Path.home() / "elumine" / "transcriptions",
        description="Path where transcription files will be saved"
    )
    artifacts_path: Path = Field(
        default_factory=lambda: Path.home() / "elumine" / "artifacts",
        description="Path where original uploaded files will be stored"
    )
    vectordb_path: Path = Field(
        default_factory=lambda: Path.home() / "elumine" / "vectordb",
        description="Path where vector database will be stored"
    )
    whisper_model: str = Field(
        default="base",
        description="Whisper model size (tiny, base, small, medium, large)"
    )
    chunk_size: int = Field(
        default=1000,
        description="Text chunk size for vector embeddings"
    )

    class Config:
        """Pydantic config."""
        validate_assignment = True


class ConfigManager:
    """Manages Elumine configuration."""

    CONFIG_DIR = Path.home() / ".config" / "elumine"
    CONFIG_FILE = CONFIG_DIR / "config.json"

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> ElumineConfig:
        """Load configuration from file or create default."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string paths back to Path objects
                    for key in ['transcriptions_path', 'artifacts_path', 'vectordb_path']:
                        if key in data:
                            data[key] = Path(data[key])
                    return ElumineConfig(**data)
            except Exception:
                # If config is corrupted, use defaults
                pass

        return ElumineConfig()

    def save_config(self):
        """Save current configuration to file."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for JSON serialization
        config_dict = self.config.model_dump()
        for key in ['transcriptions_path', 'artifacts_path', 'vectordb_path']:
            config_dict[key] = str(config_dict[key])

        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def update_config(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()

    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        self.config.transcriptions_path.mkdir(parents=True, exist_ok=True)
        self.config.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.config.vectordb_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config_manager = ConfigManager()
