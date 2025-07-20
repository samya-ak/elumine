"""CLI command implementations."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from pathlib import Path
from typing import Optional

console = Console()


def display_welcome():
    """Display the welcome screen."""
    welcome_text = Text()
    welcome_text.append("✨ Welcome to ", style="bold blue")
    welcome_text.append("Elumine", style="bold magenta")
    welcome_text.append(" ✨", style="bold blue")

    description = Text()
    description.append("Transform your audio and video files into searchable knowledge\n", style="dim")
    description.append("• Transcribe audio/video files\n", style="green")
    description.append("• Ask questions about your content\n", style="green")
    description.append("• Generate summaries and notes\n", style="green")
    description.append("• Search through your transcriptions", style="green")

    panel = Panel(
        description,
        title=welcome_text,
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    console.print()
    console.print(panel)
    console.print()


def upload_command(
    file_path: Path = typer.Argument(..., help="Path to audio/video file"),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the artifact")
):
    """📤 Upload and transcribe an audio/video file."""
    console.print(f"[green]Processing file:[/green] {file_path}")

    if not file_path.exists():
        console.print(f"[red]Error:[/red] File {file_path} not found")
        raise typer.Exit(1)

    # TODO: Implement transcription logic
    console.print("[yellow]🚧 Transcription feature coming soon![/yellow]")


def query_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to query"),
    question: str = typer.Argument(..., help="Question to ask about the artifact")
):
    """❓ Ask a question about a specific artifact."""
    console.print(f"[blue]Querying artifact '{artifact_id}':[/blue] {question}")

    # TODO: Implement RAG query logic
    console.print("[yellow]🚧 Query feature coming soon![/yellow]")


def list_command():
    """📋 List all processed artifacts."""
    console.print("[blue]Available artifacts:[/blue]")

    # TODO: List from database
    console.print("[dim]No artifacts found. Use [bold]elumine upload[/bold] to add some![/dim]")


def summarize_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to summarize")
):
    """📝 Generate summary of an artifact."""
    console.print(f"[green]Generating summary for artifact:[/green] {artifact_id}")

    # TODO: Generate summary
    console.print("[yellow]🚧 Summarization feature coming soon![/yellow]")


def notes_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to create notes from")
):
    """📓 Create structured notes from an artifact."""
    console.print(f"[green]Creating notes for artifact:[/green] {artifact_id}")

    # TODO: Generate notes
    console.print("[yellow]🚧 Notes feature coming soon![/yellow]")


def config_command(
    transcriptions_path: Optional[Path] = typer.Option(
        None, "--transcriptions-path", "-t",
        help="Set path for transcription files"
    ),
    artifacts_path: Optional[Path] = typer.Option(
        None, "--artifacts-path", "-a",
        help="Set path for original uploaded files"
    ),
    vectordb_path: Optional[Path] = typer.Option(
        None, "--vectordb-path", "-v",
        help="Set path for vector database"
    ),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model", "-m",
        help="Set Whisper model (tiny, base, small, medium, large)"
    ),
    show: bool = typer.Option(
        False, "--show", "-s",
        help="Show current configuration"
    ),
    reset: bool = typer.Option(
        False, "--reset", "-r",
        help="Reset configuration to defaults"
    )
):
    """⚙️ Configure Elumine settings."""
    from src.config import config_manager

    if reset:
        from src.config import ElumineConfig
        config_manager.config = ElumineConfig()
        config_manager.save_config()
        console.print("[green]✅ Configuration reset to defaults[/green]")
        return

    # Update configuration if any options provided
    updates = {}
    if transcriptions_path:
        updates['transcriptions_path'] = transcriptions_path.expanduser().absolute()
    if artifacts_path:
        updates['artifacts_path'] = artifacts_path.expanduser().absolute()
    if vectordb_path:
        updates['vectordb_path'] = vectordb_path.expanduser().absolute()
    if whisper_model:
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if whisper_model not in valid_models:
            console.print(f"[red]Error:[/red] Invalid model. Choose from: {', '.join(valid_models)}")
            raise typer.Exit(1)
        updates['whisper_model'] = whisper_model

    if updates:
        config_manager.update_config(**updates)
        console.print("[green]✅ Configuration updated[/green]")

        # Create directories if they don't exist
        config_manager.ensure_directories_exist()
        console.print("[dim]📁 Created necessary directories[/dim]")

    # Show current configuration (default behavior or when --show is used)
    if show or not updates:
        _display_config(config_manager.config)


def _display_config(config):
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
        "Artifacts Path",
        str(config.artifacts_path),
        "Where original files are stored"
    )
    table.add_row(
        "Vector DB Path",
        str(config.vectordb_path),
        "Where vector database is stored"
    )
    table.add_row(
        "Whisper Model",
        config.whisper_model,
        "Speech-to-text model size"
    )
    table.add_row(
        "Chunk Size",
        str(config.chunk_size),
        "Text chunk size for embeddings"
    )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]💡 Use [bold]elumine config --help[/bold] to see configuration options[/dim]")
