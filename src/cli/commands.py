"""CLI command implementations."""

import typer
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from pathlib import Path
from typing import Optional

console = Console()


def _save_transcription(result: dict, transcriptions_path: Path) -> tuple[Path, Path]:
    """Save transcription result to JSON and text files."""
    # Ensure transcriptions directory exists
    transcriptions_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in result['name'] if c.isalnum() or c in (' ', '-', '_')).strip()
    base_filename = f"{safe_name}_{timestamp}"

    json_file = transcriptions_path / f"{base_filename}.json"
    text_file = transcriptions_path / f"{base_filename}.txt"

    # Prepare data to save
    transcription_data = {
        "name": result['name'],
        "type": result['type'],
        "source": result['source'],
        "timestamp": datetime.now().isoformat(),
        "language": result.get('language'),
        "language_probability": result.get('language_probability'),
        "duration": result.get('duration'),
        "full_text": result['full_text'],
        "segments": result.get('segments', [])
    }

    # Save to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)

    # Save to plain text file
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result['full_text'])

    return json_file, text_file


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


def transcribe_command(
    input_path: str = typer.Argument(..., help="Path to audio/video file or YouTube URL"),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the artifact")
):
    """🎤 Transcribe an audio/video file or YouTube video."""
    from src.config import config_manager
    from src.core.transcription import TranscriptionService

    try:
        # Load configuration
        config = config_manager.config

        # Initialize transcription service
        transcription_service = TranscriptionService(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type
        )

        console.print(f"[green]Processing:[/green] {input_path}")

        # Transcribe the input
        result = transcription_service.transcribe_input(input_path, name)

        # Display results
        console.print(f"[green]✅ Successfully transcribed:[/green] {result['name']}")
        console.print(f"[blue]Type:[/blue] {result['type']}")
        console.print(f"[blue]Language:[/blue] {result.get('language', 'N/A')}")
        if result.get('duration'):
            minutes = int(result['duration'] // 60)
            seconds = int(result['duration'] % 60)
            console.print(f"[blue]Duration:[/blue] {minutes}:{seconds:02d}")

        # Show preview of transcription
        preview_text = result['full_text'][:200]
        if len(result['full_text']) > 200:
            preview_text += "..."

        console.print(f"[dim]Preview:[/dim] {preview_text}")

        # Save transcription to configured directory
        json_file, text_file = _save_transcription(result, config.transcriptions_path)
        console.print(f"[green]💾 Transcription saved to:[/green]")
        console.print(f"  [dim]JSON:[/dim] {json_file}")
        console.print(f"  [dim]Text:[/dim] {text_file}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


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
    console.print("[dim]No artifacts found. Use [bold]elumine transcribe[/bold] to add some![/dim]")


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
    vectordb_path: Optional[Path] = typer.Option(
        None, "--vectordb-path", "-v",
        help="Set path for vector database"
    ),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model", "-m",
        help="Set Whisper model (tiny, base, small, medium, large)"
    ),
    whisper_device: Optional[str] = typer.Option(
        None, "--whisper-device", "-d",
        help="Set Whisper device (cpu, cuda, auto)"
    ),
    whisper_compute_type: Optional[str] = typer.Option(
        None, "--whisper-compute-type", "-c",
        help="Set Whisper compute type (int8, int16, float16, float32)"
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
    if vectordb_path:
        updates['vectordb_path'] = vectordb_path.expanduser().absolute()
    if whisper_model:
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if whisper_model not in valid_models:
            console.print(f"[red]Error:[/red] Invalid model. Choose from: {', '.join(valid_models)}")
            raise typer.Exit(1)
        updates['whisper_model'] = whisper_model
    if whisper_device:
        valid_devices = ["cpu", "cuda", "auto"]
        if whisper_device not in valid_devices:
            console.print(f"[red]Error:[/red] Invalid device. Choose from: {', '.join(valid_devices)}")
            raise typer.Exit(1)
        updates['whisper_device'] = whisper_device
    if whisper_compute_type:
        valid_compute_types = ["int8", "int16", "float16", "float32"]
        if whisper_compute_type not in valid_compute_types:
            console.print(f"[red]Error:[/red] Invalid compute type. Choose from: {', '.join(valid_compute_types)}")
            raise typer.Exit(1)
        updates['whisper_compute_type'] = whisper_compute_type

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

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]💡 Use [bold]elumine config --help[/bold] to see configuration options[/dim]")
