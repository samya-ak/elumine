"""UI/Display handlers for the CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def display_welcome() -> None:
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
    console.print("[dim]Use [bold]elumine --help[/bold] to see available commands[/dim]")
