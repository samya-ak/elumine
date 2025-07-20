"""CLI application setup and configuration."""

import typer
from rich.console import Console
from .commands import (
    display_welcome,
    transcribe_command,
    query_command,
    list_command,
    summarize_command,
    notes_command,
    config_command,
)

console = Console()

app = typer.Typer(
    name="elumine",
    help="🎵 Audio/Video transcription and RAG CLI tool",
    add_completion=False,
    rich_markup_mode="rich"
)

# Register commands
app.command("transcribe", help="🎤 Transcribe audio/video files or YouTube videos")(transcribe_command)
app.command("query", help="❓ Ask a question about a specific artifact")(query_command)
app.command("list", help="📋 List all processed artifacts")(list_command)
app.command("summarize", help="📝 Generate summary of an artifact")(summarize_command)
app.command("notes", help="📓 Create structured notes from an artifact")(notes_command)
app.command("config", help="⚙️ Configure Elumine settings")(config_command)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
):
    """🎵 Elumine - Audio/Video transcription and RAG CLI tool."""
    from src import __version__

    if version:
        console.print(f"Elumine version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        display_welcome()
