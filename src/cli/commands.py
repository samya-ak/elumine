"""CLI command implementations."""

import typer
from pathlib import Path
from typing import Optional

from .handlers.ui import display_welcome
from .handlers.transcription import handle_transcribe
from .handlers.config import handle_config
from .handlers.rag import handle_query, handle_list, handle_summarize, handle_notes


def transcribe_command(
    input_path: str = typer.Argument(..., help="Path to audio/video file or YouTube URL"),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the artifact")
):
    """🎤 Transcribe an audio/video file or YouTube video."""
    try:
        handle_transcribe(input_path, name)
    except (FileNotFoundError, ValueError, Exception) as e:
        raise typer.Exit(1)


def query_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to query"),
    question: str = typer.Argument(..., help="Question to ask about the artifact")
):
    """❓ Ask a question about a specific artifact."""
    handle_query(artifact_id, question)


def list_command():
    """📋 List all processed artifacts."""
    handle_list()


def summarize_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to summarize")
):
    """📝 Generate summary of an artifact."""
    handle_summarize(artifact_id)


def notes_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to create notes from")
):
    """📓 Create structured notes from an artifact."""
    handle_notes(artifact_id)


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
    try:
        handle_config(
            transcriptions_path=transcriptions_path,
            vectordb_path=vectordb_path,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            show=show,
            reset=reset
        )
    except ValueError:
        raise typer.Exit(1)
