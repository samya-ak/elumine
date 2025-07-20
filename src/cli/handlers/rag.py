"""RAG (Retrieval Augmented Generation) command handlers."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown
from src.db.sqlite import get_db
from src.services.langchain_service import get_langchain_service

console = Console()

def handle_query(artifact_id: str, question: str) -> None:
    """Handle querying artifacts with questions."""
    try:
        db = get_db()
        langchain_service = get_langchain_service()

        # Get artifact from database
        artifacts = list(db["artifacts"].rows_where("id = ?", [artifact_id]))
        if not artifacts:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        artifact = artifacts[0]
        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[blue]Querying artifact '{artifact['filename']}':[/blue] {question}")

        # Search for relevant documents
        results = langchain_service.search_by_filename(question, artifact["filename"], k=5)

        if not results:
            console.print("[yellow]No relevant information found for your question.[/yellow]")
            return

        # Generate answer using RAG
        answer = langchain_service.generate_answer(question, results)

        # Display the answer
        panel = Panel(
            Markdown(answer),
            title="💡 Answer",
            border_style="green",
            box=box.ROUNDED
        )
        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

def handle_list() -> None:
    """Handle listing all processed artifacts."""
    try:
        db = get_db()

        # Get all artifacts
        artifacts = list(db["artifacts"].rows)

        if not artifacts:
            console.print("[dim]No artifacts found. Use [bold]elumine ingest[/bold] to add some![/dim]")
            return

        # Create table
        table = Table(title="📋 Available Artifacts", box=box.ROUNDED)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Filename", style="green")
        table.add_column("Type", style="blue", width=10)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Chunks", style="magenta", width=8)
        table.add_column("Batch", style="dim", width=8)

        for artifact in artifacts:
            status_style = "green" if artifact["status"] == "ingested" else "red"
            table.add_row(
                str(artifact["id"]),
                artifact["filename"],
                artifact["filetype"],
                f"[{status_style}]{artifact['status']}[/{status_style}]",
                str(artifact.get("chunk_count", 0)),
                str(artifact["batch_id"])
            )

        console.print()
        console.print(table)
        console.print()
        console.print("[dim]💡 Use [bold]elumine query <artifact-id> \"your question\"[/bold] to ask questions about an artifact[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

def handle_summarize(artifact_id: str) -> None:
    """Handle generating summaries of artifacts."""
    try:
        db = get_db()
        langchain_service = get_langchain_service()

        # Get artifact from database
        artifacts = list(db["artifacts"].rows_where("id = ?", [artifact_id]))
        if not artifacts:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        artifact = artifacts[0]
        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[green]Generating summary for:[/green] {artifact['filename']}")

        # Get all documents for this artifact
        results = langchain_service.search_by_filename("", artifact["filename"], k=50)

        if not results:
            console.print("[yellow]No content found for summarization.[/yellow]")
            return

        # Generate summary
        with console.status("Generating summary..."):
            summary = langchain_service.generate_summary(results)

        # Display the summary
        panel = Panel(
            Markdown(summary),
            title=f"📝 Summary: {artifact['filename']}",
            border_style="blue",
            box=box.ROUNDED
        )
        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

def handle_notes(artifact_id: str) -> None:
    """Handle creating structured notes from artifacts."""
    try:
        db = get_db()
        langchain_service = get_langchain_service()

        # Get artifact from database
        artifacts = list(db["artifacts"].rows_where("id = ?", [artifact_id]))
        if not artifacts:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        artifact = artifacts[0]
        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[green]Creating structured notes for:[/green] {artifact['filename']}")

        # Get all documents for this artifact
        results = langchain_service.search_by_filename("", artifact["filename"], k=50)

        if not results:
            console.print("[yellow]No content found for note generation.[/yellow]")
            return

        # Generate structured notes
        with console.status("Creating structured notes..."):
            notes = langchain_service.generate_notes(results)

        # Display the notes
        panel = Panel(
            Markdown(notes),
            title=f"📓 Notes: {artifact['filename']}",
            border_style="magenta",
            box=box.ROUNDED
        )
        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
