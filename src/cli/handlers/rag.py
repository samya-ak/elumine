"""RAG (Retrieval Augmented Generation) command handlers."""

from pathlib import Path
from datetime import datetime
from typing import Optional
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

        console.print(f"[blue]Querying artifact '{artifact_id}':[/blue] {question}")

        # Search for relevant documents within this artifact
        # This will return None if artifact doesn't exist or has no content
        results = chroma_service.search_by_artifact(question, artifact_id, k=5)

        if not results:
            console.print(f"[yellow]No relevant information found for artifact {artifact_id}. "
                         "Either the artifact doesn't exist or no content matches your question.[/yellow]")
            return

        # Get artifact metadata from the first result (all chunks share core metadata)
        artifact_metadata = results[0].metadata
        filename = artifact_metadata.get('filename', artifact_id)
        status = artifact_metadata.get('status', 'unknown')

        if status != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested (status: {status})")
            return

        console.print(f"[blue]Found content in '{filename}'[/blue]")

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

def handle_summarize(artifact_id: str, save_dir: Optional[Path] = None) -> None:
    """Handle generating summaries of artifacts."""
    try:
        chroma_service = get_chroma_service()

        # Get all documents for this artifact
        results = chroma_service.get_all_chunks_for_artifact(artifact_id)

        if not results:
            console.print(f"[yellow]No content found for artifact {artifact_id}. "
                         "Either the artifact doesn't exist or has no content.[/yellow]")
            return

        # Get artifact metadata from the first result (all chunks share core metadata)
        artifact_metadata = results[0].metadata
        filename = artifact_metadata.get('filename', artifact_id)
        status = artifact_metadata.get('status', 'unknown')

        if status != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested (status: {status})")
            return

        console.print(f"[green]Generating summary for:[/green] {filename}")

        # Generate summary
        with console.status("Generating summary..."):
            summary = chroma_service.generate_summary(results)

        # Save to file if save_dir is provided
        if save_dir:
            if not save_dir.exists():
                console.print(f"[red]Error:[/red] Directory {save_dir} does not exist")
                return
            if not save_dir.is_dir():
                console.print(f"[red]Error:[/red] {save_dir} is not a directory")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{artifact_metadata['filename']}_{timestamp}.md"
            file_path = save_dir / filename

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Summary: {artifact_metadata['filename']}\n\n")
                    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Artifact ID:** {artifact_id}\n")
                    f.write(f"**Source:** {artifact_metadata.get('source', 'N/A')}\n\n")
                    f.write("---\n\n")
                    f.write(summary)

                console.print(f"[green]Summary saved to:[/green] {file_path}")
            except Exception as e:
                console.print(f"[red]Error saving file:[/red] {e}")
                return

        # Display the summary
        panel = Panel(
            Markdown(summary),
            title=f"📝 Summary: {filename}",
            border_style="blue",
            box=box.ROUNDED
        )
        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

def handle_notes(artifact_id: str, save_dir: Optional[Path] = None) -> None:
    """Handle creating structured notes from artifacts."""
    try:
        chroma_service = get_chroma_service()

        # Get all documents for this artifact
        results = chroma_service.get_all_chunks_for_artifact(artifact_id)

        if not results:
            console.print(f"[yellow]No content found for artifact {artifact_id}. "
                         "Either the artifact doesn't exist or has no content.[/yellow]")
            return

        # Get artifact metadata from the first result (all chunks share core metadata)
        artifact_metadata = results[0].metadata
        filename = artifact_metadata.get('filename', artifact_id)
        status = artifact_metadata.get('status', 'unknown')

        if status != "ingested":
            console.print(f"[red]Error:[/red] Artifact {artifact_id} is not properly ingested (status: {status})")
            return

        console.print(f"[green]Creating structured notes for:[/green] {filename}")

        # Generate structured notes
        with console.status("Creating structured notes..."):
            notes = chroma_service.generate_notes(results)

        # Save to file if save_dir is provided
        if save_dir:
            if not save_dir.exists():
                console.print(f"[red]Error:[/red] Directory {save_dir} does not exist")
                return
            if not save_dir.is_dir():
                console.print(f"[red]Error:[/red] {save_dir} is not a directory")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"notes_{artifact_metadata['filename']}_{timestamp}.md"
            file_path = save_dir / filename

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Notes: {artifact_metadata['filename']}\n\n")
                    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Artifact ID:** {artifact_id}\n")
                    f.write(f"**Source:** {artifact_metadata.get('source', 'N/A')}\n\n")
                    f.write("---\n\n")
                    f.write(notes)

                console.print(f"[green]Notes saved to:[/green] {file_path}")
            except Exception as e:
                console.print(f"[red]Error saving file:[/red] {e}")
                return

        # Display the notes
        panel = Panel(
            Markdown(notes),
            title=f"📓 Notes: {filename}",
            border_style="magenta",
            box=box.ROUNDED
        )
        console.print()
        console.print(panel)
        console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
