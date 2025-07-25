"""Ingest command handler for Elumine."""

import re
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from .transcription import handle_transcribe
from src.db.sqlite import get_db
from src.services.langchain_service import get_langchain_service


console = Console()

def is_youtube_url(url: str) -> bool:
    """Check if the provided string is a YouTube URL."""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return bool(youtube_regex.match(url))

def handle_ingest(
    medias: List[str],
    batch_name: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Ingest up to 5 media files or YouTube URLs, transcribe if needed, store metadata in SQLite and text in ChromaDB."""
    if len(medias) > 5:
        console.print("[red]Error:[/red] You can ingest up to 5 media items at a time.")
        raise ValueError("Too many media items")

    db = get_db()
    langchain_service = get_langchain_service()

    # Create batch
    batch = {"name": batch_name}
    batch_result = db["batches"].insert(batch)
    batch_id = batch_result.last_pk

    for media_path in medias:
        chroma_ids = []
        error = None
        filename = ""
        filetype = ""
        status = "pending"

        try:
            # Check if it's a YouTube URL
            if is_youtube_url(media_path):
                # Handle YouTube URL
                filename = f"youtube_{media_path.split('v=')[-1][:11] if 'v=' in media_path else 'video'}"
                filetype = ".youtube"

                if verbose:
                    console.print(f"[blue]Processing YouTube URL:[/blue] {media_path}")

                # Transcribe YouTube video
                text = handle_transcribe(media_path, None, return_text=True)

                base_metadata = {
                    "filetype": filetype,
                    "source": media_path,
                    "type": "youtube_video"
                }

            else:
                # Handle local file
                file = Path(media_path)
                filename = file.name
                filetype = file.suffix.lower()

                if verbose:
                    console.print(f"[blue]Processing local file:[/blue] {file.name}")

                if filetype in [".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".flac", ".mkv", ".webm"]:
                    # Transcribe audio/video
                    text = handle_transcribe(str(file), None, return_text=True)
                    base_metadata = {
                        "filetype": filetype,
                        "source": str(file.absolute()),
                        "type": "local_media"
                    }
                elif filetype in [".txt", ".md", ".rtf"]:
                    # Read text file directly
                    text = file.read_text(encoding='utf-8')
                    base_metadata = {
                        "filetype": filetype,
                        "source": str(file.absolute()),
                        "type": "text_file"
                    }
                else:
                    raise ValueError(f"Unsupported file type: {filetype}")

            if not text or not text.strip():
                raise ValueError("No content extracted from media")

            # Create documents using LangChain
            documents = langchain_service.create_documents(
                text=text,
                metadata=base_metadata,
                filename=filename,
                batch_id=batch_id
            )

            if verbose:
                console.print(f"[blue]Processing {filename}: {len(documents)} chunks[/blue]")

            # Add documents to vectorstore with embeddings
            chroma_ids = langchain_service.add_documents(documents)

            status = "ingested"

        except Exception as e:
            error = str(e)
            status = "error"
            if verbose:
                console.print(f"[red]Error processing {filename or media_path}: {error}[/red]")

        db["artifacts"].insert({
            "batch_id": batch_id,
            "filename": filename or media_path,
            "filetype": filetype,
            "status": status,
            "chroma_ids": ",".join(chroma_ids) if chroma_ids else None,
            "chunk_count": len(chroma_ids),
            "error": error
        })

        if verbose and status == "ingested":
            console.print(f"[green]Ingested:[/green] {filename} ({len(chroma_ids)} chunks)")

    if verbose:
        console.print(f"[bold green]Batch {batch_id} ingest complete.[/bold green]")
