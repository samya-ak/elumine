"""Transcription service using faster-whisper."""

import re
from pathlib import Path
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel
import yt_dlp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class TranscriptionService:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """Initialize the transcription service."""
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            console.print(f"[blue]Loading Whisper model ({self.model_size})...[/blue]")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def is_youtube_url(self, url: str) -> bool:
        """Check if the provided string is a YouTube URL."""
        youtube_regex = re.compile(
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(youtube_regex.match(url))

    def is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file that doesn't need transcription."""
        text_extensions = {'.txt', '.md', '.rtf', '.doc', '.docx', '.pdf'}
        return file_path.suffix.lower() in text_extensions

    def is_media_file(self, file_path: Path) -> bool:
        """Check if file is an audio or video file."""
        media_extensions = {
            # Audio
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma',
            # Video
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'
        }
        return file_path.suffix.lower() in media_extensions

    def download_youtube_video(self, url: str, output_path: Path) -> Path:
        """Download YouTube video and return the file path."""
        console.print(f"[blue]📥 Downloading from YouTube...[/blue]")

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best
            'outtmpl': str(output_path / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

            console.print(f"[green]📹 Title:[/green] {title}")
            if duration:
                console.print(f"[green]⏱️ Duration:[/green] {duration//60}:{duration%60:02d}")

            # Download the video
            ydl.download([url])

            # Find the downloaded file
            for file_path in output_path.glob(f"{title}.*"):
                if file_path.suffix.lower() in {'.mp4', '.webm', '.mkv', '.m4v'}:
                    return file_path

        raise Exception("Failed to download video")

    def transcribe_media(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe audio/video file using faster-whisper."""
        self._load_model()

        console.print(f"[blue]🎤 Transcribing media file...[/blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing audio...", total=None)

            # faster-whisper can handle video files directly
            segments, info = self.model.transcribe(str(file_path))

            # Collect all segments
            transcription_segments = []
            full_text = ""

            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                transcription_segments.append(segment_data)
                full_text += segment.text.strip() + " "

            progress.update(task, completed=True, description="✅ Transcription complete")

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "full_text": full_text.strip(),
            "segments": transcription_segments
        }

    def transcribe_input(self, input_path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe input (file path or YouTube URL) and return transcription."""

        # Check if it's a YouTube URL
        if self.is_youtube_url(input_path):
            # For YouTube, we'll need to download to a temp location
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Download video to temp directory
                video_path = self.download_youtube_video(input_path, temp_path)

                # Generate artifact name
                artifact_name = name or video_path.stem

                # Transcribe the downloaded video
                result = self.transcribe_media(video_path)
                result.update({
                    "type": "youtube_video",
                    "source": input_path,
                    "name": artifact_name
                })
                return result

        # Handle local file
        file_path = Path(input_path).expanduser().absolute()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        artifact_name = name or file_path.stem

        # Check if it's a text file (not supported for transcription)
        if self.is_text_file(file_path):
            raise ValueError(
                f"Text files don't need transcription. "
                f"File type '{file_path.suffix}' is not supported. "
                f"Please provide an audio or video file."
            )

        # Check if it's a media file
        elif self.is_media_file(file_path):
            # For local files, transcribe in place - no copying
            result = self.transcribe_media(file_path)
            result.update({
                "type": "local_media",
                "source": str(file_path),
                "name": artifact_name
            })
            return result

        else:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported formats: MP3, WAV, MP4, AVI, MOV, etc."
            )
