"""Ingest command handler for Elumine."""


from pathlib import Path
from typing import List, Optional
from rich.console import Console
from .transcription import handle_transcribe
from src.db.sqlite import get_db
from src.services.langchain_service import get_langchain_service


console = Console()

def handle_ingest(
    files: List[str],
    batch_name: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Ingest up to 5 files (audio, video, or text), transcribe if needed, store metadata in SQLite and text in ChromaDB."""
    if len(files) > 5:
        console.print("[red]Error:[/red] You can ingest up to 5 files at a time.")
        raise ValueError("Too many files")

    db = get_db()
    langchain_service = get_langchain_service()

    # Create batch
    batch = {"name": batch_name}
    batch_id = db["batches"].insert(batch, pk="id")

    for file_path in files:
        file = Path(file_path)
        filetype = file.suffix.lower()
        status = "pending"
        chroma_ids = []
        error = None
        try:
            if filetype in [".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".flac"]:
                # Transcribe audio/video
                transcript = handle_transcribe(str(file), None, return_text=True)
                text = transcript if transcript else ""
            elif filetype in [".txt", ".md", ".rtf"]:
                text = file.read_text(encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {filetype}")

            if not text.strip():
                raise ValueError("No content extracted from file")

            # Create documents using LangChain
            base_metadata = {
                "filetype": filetype,
                "source": str(file.absolute())
            }

            documents = langchain_service.create_documents(
                text=text,
                metadata=base_metadata,
                filename=file.name,
                batch_id=batch_id
            )

            if verbose:
                console.print(f"[blue]Processing {file.name}: {len(documents)} chunks[/blue]")

            # Add documents to vectorstore with embeddings
            chroma_ids = langchain_service.add_documents(documents)

            status = "ingested"

        except Exception as e:
            error = str(e)
            status = "error"
            if verbose:
                console.print(f"[red]Error processing {file.name}: {error}[/red]")

        db["artifacts"].insert({
            "batch_id": batch_id,
            "filename": file.name,
            "filetype": filetype,
            "status": status,
            "chroma_ids": ",".join(chroma_ids) if chroma_ids else None,
            "chunk_count": len(chroma_ids),
            "error": error
        })

        if verbose and status == "ingested":
            console.print(f"[green]Ingested:[/green] {file.name} ({len(chroma_ids)} chunks)")

    if verbose:
        console.print(f"[bold green]Batch {batch_id} ingest complete.[/bold green]")
