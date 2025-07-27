"""CLI command implementations."""

import typer
from pathlib import Path
from typing import Optional

from .handlers.ui import display_welcome
from .handlers.transcription import handle_transcribe
from .handlers.config import handle_config
from .handlers.rag import handle_query, handle_list, handle_summarize, handle_notes
from .handlers.ingest import handle_ingest


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
    artifact_id: str = typer.Argument(..., help="ID of the artifact to summarize"),
    save: Optional[Path] = typer.Option(None, "--save", help="Directory path to save the summary as a markdown file")
):
    """📝 Generate summary of an artifact."""
    handle_summarize(artifact_id, save)


def notes_command(
    artifact_id: str = typer.Argument(..., help="ID of the artifact to create notes from"),
    save: Optional[Path] = typer.Option(None, "--save", help="Directory path to save the notes as a markdown file")
):
    """📓 Create structured notes from an artifact."""
    handle_notes(artifact_id, save)


def config_command(
    transcriptions_path: Optional[Path] = typer.Option(
        None, "--transcriptions-path", "-t",
        help="Set path for transcription files"
    ),
    db_path: Optional[Path] = typer.Option(
        None, "--db-path", "-db",
        help="Set path for Elumine databases (ChromaDB)"
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
    openai_api_key: Optional[str] = typer.Option(
        None, "--openai-api-key", "-k",
        help="Set OpenAI API key for embeddings and completions"
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model", "-l",
        help="Set OpenAI model for chat completions (e.g., gpt-3.5-turbo, gpt-4)"
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
            db_path=db_path,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            openai_api_key=openai_api_key,
            llm_model=llm_model,
            show=show,
            reset=reset
        )
    except ValueError:
        raise typer.Exit(1)

def ingest_command(
    medias: list[str] = typer.Argument(..., help="Paths to up to 5 media files (audio, video, text) or YouTube URLs to ingest"),
    batch_name: str = typer.Option(None, "--batch-name", "-b", help="Optional batch name for this ingest"),
):
    """📥 Ingest up to 5 media files or YouTube URLs, transcribe if needed, store in ChromaDB."""
    try:
        if len(medias) > 5:
            typer.echo("❌ You can ingest up to 5 media items at a time.")
            raise typer.Exit(1)

        handle_ingest(medias, batch_name)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)
