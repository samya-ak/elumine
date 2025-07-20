"""Transcription command handlers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from rich.console import Console

from src.config import config_manager
from src.core.transcription import TranscriptionService

console = Console()


def save_transcription(result: Dict[str, Any], transcriptions_path: Path) -> Tuple[Path, Path]:
    """Save transcription result to JSON and text files."""
    # Ensure transcriptions directory exists
    transcriptions_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in result['name'] if c.isalnum() or c in (' ', '-', '_')).strip()
    base_filename = f"{safe_name}_{timestamp}"

    json_file = transcriptions_path / f"{base_filename}.json"
    text_file = transcriptions_path / f"{base_filename}.txt"

    # Prepare data to save
    transcription_data = {
        "name": result['name'],
        "type": result['type'],
        "source": result['source'],
        "timestamp": datetime.now().isoformat(),
        "language": result.get('language'),
        "language_probability": result.get('language_probability'),
        "duration": result.get('duration'),
        "full_text": result['full_text'],
        "segments": result.get('segments', [])
    }

    # Save to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)

    # Save to plain text file
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result['full_text'])

    return json_file, text_file


def handle_transcribe(input_path: str, name: str = None) -> None:
    """Handle transcription of audio/video files or YouTube URLs."""
    try:
        # Load configuration
        config = config_manager.config

        # Initialize transcription service
        transcription_service = TranscriptionService(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type
        )

        console.print(f"[green]Processing:[/green] {input_path}")

        # Transcribe the input
        result = transcription_service.transcribe_input(input_path, name)

        # Display results
        console.print(f"[green]✅ Successfully transcribed:[/green] {result['name']}")
        console.print(f"[blue]Type:[/blue] {result['type']}")
        console.print(f"[blue]Language:[/blue] {result.get('language', 'N/A')}")
        if result.get('duration'):
            minutes = int(result['duration'] // 60)
            seconds = int(result['duration'] % 60)
            console.print(f"[blue]Duration:[/blue] {minutes}:{seconds:02d}")

        # Show preview of transcription
        preview_text = result['full_text'][:200]
        if len(result['full_text']) > 200:
            preview_text += "..."

        console.print(f"[dim]Preview:[/dim] {preview_text}")

        # Save transcription to configured directory
        json_file, text_file = save_transcription(result, config.transcriptions_path)
        console.print(f"[green]💾 Transcription saved to:[/green]")
        console.print(f"  [dim]JSON:[/dim] {json_file}")
        console.print(f"  [dim]Text:[/dim] {text_file}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
