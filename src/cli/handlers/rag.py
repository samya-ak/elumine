"""RAG (Retrieval Augmented Generation) command handlers."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown
from src.services.chroma_service import get_chroma_service

console = Console()

def handle_query(artifact_id: str, question: str) -> None:
    """Handle querying artifacts with questions."""
    try:
        chroma_service = get_chroma_service()

        # Get artifact from ChromaDB
        artifact = chroma_service.get_artifact_by_id(artifact_id)
        if not artifact:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[blue]Querying artifact '{artifact['filename']}':[/blue] {question}")

        # Search for relevant documents within this artifact
        results = chroma_service.search_by_artifact(question, artifact_id, k=5)

        if not results:
            console.print("[yellow]No relevant information found for your question.[/yellow]")
            return

        # Generate answer using RAG
        answer = chroma_service.generate_answer(question, results)

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
        chroma_service = get_chroma_service()

        # Get all artifacts
        artifacts = chroma_service.list_artifacts()

        if not artifacts:
            console.print("[dim]No artifacts found. Use [bold]elumine ingest[/bold] to add some![/dim]")
            return

        # Create table
        table = Table(title="📋 Available Artifacts", box=box.ROUNDED)
        table.add_column("Artifact ID", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Type", style="blue", width=12)
        table.add_column("Source Type", style="yellow", width=12)
        table.add_column("Chunks", style="magenta", width=8)
        table.add_column("Batch", style="dim", width=8)

        for artifact in artifacts:
            status_style = "green" if artifact["status"] == "ingested" else "red"
            table.add_row(
                artifact["artifact_id"],
                artifact["filename"],
                artifact.get("filetype", ""),
                artifact.get("source_type", ""),
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
        chroma_service = get_chroma_service()

        # Get artifact from ChromaDB
        artifact = chroma_service.get_artifact_by_id(artifact_id)
        if not artifact:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[green]Generating summary for:[/green] {artifact['filename']}")

        # Get all documents for this artifact
        results = chroma_service.get_all_chunks_for_artifact(artifact_id)

        if not results:
            console.print("[yellow]No content found for summarization.[/yellow]")
            return

        # Generate summary
        with console.status("Generating summary..."):
            summary = chroma_service.generate_summary(results)

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
        chroma_service = get_chroma_service()

        # Get artifact from ChromaDB
        artifact = chroma_service.get_artifact_by_id(artifact_id)
        if not artifact:
            console.print(f"[red]Error:[/red] Artifact with ID {artifact_id} not found")
            return

        if artifact["status"] != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested")
            return

        console.print(f"[green]Creating structured notes for:[/green] {artifact['filename']}")

        # Get all documents for this artifact
        results = chroma_service.get_all_chunks_for_artifact(artifact_id)

        if not results:
            console.print("[yellow]No content found for note generation.[/yellow]")
            return

        # Generate structured notes
        with console.status("Creating structured notes..."):
            notes = chroma_service.generate_notes(results)

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
