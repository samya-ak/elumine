"""Configuration command handlers."""

from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import config_manager, ElumineConfig

console = Console()


def handle_config(
    transcriptions_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    whisper_model: Optional[str] = None,
    whisper_device: Optional[str] = None,
    whisper_compute_type: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    show: bool = False,
    reset: bool = False
) -> None:
    """Handle configuration updates and display."""

    if reset:
        config_manager.config = ElumineConfig()
        config_manager.save_config()
        console.print("[green]✅ Configuration reset to defaults[/green]")
        return

    # Update configuration if any options provided
    updates = {}

    if transcriptions_path:
        updates['transcriptions_path'] = transcriptions_path.expanduser().absolute()

    if db_path:
        updates['db_path'] = db_path.expanduser().absolute()

    if whisper_model:
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if whisper_model not in valid_models:
            console.print(f"[red]Error:[/red] Invalid model. Choose from: {', '.join(valid_models)}")
            raise ValueError(f"Invalid model: {whisper_model}")
        updates['whisper_model'] = whisper_model

    if whisper_device:
        valid_devices = ["cpu", "cuda", "auto"]
        if whisper_device not in valid_devices:
            console.print(f"[red]Error:[/red] Invalid device. Choose from: {', '.join(valid_devices)}")
            raise ValueError(f"Invalid device: {whisper_device}")
        updates['whisper_device'] = whisper_device

    if whisper_compute_type:
        valid_compute_types = ["int8", "int16", "float16", "float32"]
        if whisper_compute_type not in valid_compute_types:
            console.print(f"[red]Error:[/red] Invalid compute type. Choose from: {', '.join(valid_compute_types)}")
            raise ValueError(f"Invalid compute type: {whisper_compute_type}")
        updates['whisper_compute_type'] = whisper_compute_type

    if openai_api_key:
        updates['openai_api_key'] = openai_api_key

    if updates:
        config_manager.update_config(**updates)
        console.print("[green]✅ Configuration updated[/green]")

        # Create directories if they don't exist
        config_manager.ensure_directories_exist()
        console.print("[dim]📁 Created necessary directories[/dim]")

    # Show current configuration (default behavior or when --show is used)
    if show or not updates:
        display_config(config_manager.config)


def display_config(config: ElumineConfig) -> None:
    """Display current configuration in a table."""
    table = Table(title="🔧 Elumine Configuration", box=box.ROUNDED)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    table.add_row(
        "Transcriptions Path",
        str(config.transcriptions_path),
        "Where transcription files are saved"
    )
    table.add_row(
        "DB Path",
        str(config.db_path),
        "Where database is stored"
    )
    table.add_row(
        "Whisper Model",
        config.whisper_model,
        "Speech-to-text model size"
    )
    table.add_row(
        "Whisper Device",
        config.whisper_device,
        "Device to run Whisper on"
    )
    table.add_row(
        "Whisper Compute Type",
        config.whisper_compute_type,
        "Compute precision for Whisper"
    )
    table.add_row(
        "Chunk Size",
        str(config.chunk_size),
        "Text chunk size for embeddings"
    )
    table.add_row(
        "OpenAI API Key",
        "***" if config.openai_api_key else "Not set",
        "API key for OpenAI services"
    )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]💡 Use [bold]elumine config --help[/bold] to see configuration options[/dim]")
