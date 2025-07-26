"""Ingest command handler for Elumine."""

import re
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from .transcription import handle_transcribe
from src.services.chroma_service import get_chroma_service


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
) -> None:
    """Ingest up to 5 media files or YouTube URLs, transcribe if needed, store everything in ChromaDB."""
    if len(medias) > 5:
        console.print("[red]Error:[/red] You can ingest up to 5 media items at a time.")
        raise ValueError("Too many media items")

    chroma_service = get_chroma_service()

    # Create batch
    batch_id = chroma_service.create_batch(batch_name)

    console.print(f"[blue]Created batch {batch_id}: {batch_name or f'batch_{batch_id}'}[/blue]")

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
                video_id = media_path.split('v=')[-1][:11] if 'v=' in media_path else 'video'
                filename = f"youtube_{video_id}"
                filetype = ".youtube"
                source_type = "youtube_video"

                console.print(f"[blue]Processing YouTube URL:[/blue] {media_path}")

                # Transcribe YouTube video
                text = handle_transcribe(media_path, None, return_text=True)

                # Create documents with comprehensive metadata
                documents = chroma_service.create_documents(
                    text=text,
                    filename=filename,
                    batch_id=batch_id,
                    source_type=source_type,
                    source=media_path,
                    filetype=filetype,
                    video_id=video_id
                )

            else:
                # Handle local file
                file = Path(media_path)

                # Check if file exists
                if not file.exists():
                    raise ValueError(f"File not found: {media_path}")

                filename = file.name
                filetype = file.suffix.lower()

                console.print(f"[blue]Processing local file:[/blue] {file.name}")

                if filetype in [".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".flac", ".mkv", ".webm"]:
                    # Transcribe audio/video
                    text = handle_transcribe(str(file), None, return_text=True)
                    source_type = "local_media"
                elif filetype in [".txt", ".md", ".rtf"]:
                    # Read text file directly
                    text = file.read_text(encoding='utf-8')
                    source_type = "text_file"
                else:
                    if not filetype:
                        raise ValueError(f"No file extension found for '{file.name}'. Please provide a valid file path.")
                    else:
                        raise ValueError(f"Unsupported file type: {filetype}. Supported types: audio/video (.mp3, .wav, .m4a, .mp4, .mov, .avi, .flac, .mkv, .webm) or text (.txt, .md, .rtf)")

                # Create documents with comprehensive metadata
                documents = chroma_service.create_documents(
                    text=text,
                    filename=filename,
                    batch_id=batch_id,
                    source_type=source_type,
                    source=str(file.absolute()),
                    filetype=filetype,
                    file_size=file.stat().st_size if file.exists() else 0
                )

            if not text or not text.strip():
                raise ValueError("No content extracted from media")

            console.print(f"[blue]Created {len(documents)} chunks for {filename}[/blue]")

            # Add documents to ChromaDB with embeddings
            chroma_ids = chroma_service.add_documents(documents)

            status = "ingested"

        except Exception as e:
            error = str(e)
            status = "error"

            console.print(f"[red]Error processing {filename or media_path}: {error}[/red]")

        if status == "ingested":
            console.print(f"[green]Ingested:[/green] {filename} ({len(chroma_ids)} chunks)")

    console.print(f"[bold green]Batch {batch_id} ingest complete.[/bold green]")
