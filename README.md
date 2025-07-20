# Elumine 🎵

> Transform your audio and video files into searchable knowledge

Elumine is a command-line tool that transcribes audio/video files and enables you to interact with the content through questions, summaries, and note generation using RAG (Retrieval-Augmented Generation).

## Features

- **📤 Upload & Transcribe**: Process audio and video files with high-quality transcription
- **❓ Smart Querying**: Ask specific questions about your transcribed content
- **📝 Summarization**: Generate concise summaries of your media files
- **📓 Note Generation**: Create structured notes from your content
- **🔍 Search**: Find and retrieve specific information across all your transcriptions

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd elumine

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# View welcome screen
uv run elumine

# Upload and transcribe a file
uv run elumine upload path/to/your/audio.mp3

# Ask questions about your content
uv run elumine query meeting-001 "What were the main action items?"

# Generate a summary
uv run elumine summarize meeting-001

# Create structured notes
uv run elumine notes meeting-001

# List all processed files
uv run elumine list
```

## Project Structure

```
elumine/
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── cli/                 # CLI commands
│   ├── core/                # Core functionality
│   │   ├── transcription.py # Whisper integration
│   │   ├── rag.py          # RAG implementation
│   │   └── storage.py      # Database operations
│   ├── models/              # Pydantic data models
│   └── utils/               # Helper functions
├── pyproject.toml
└── README.md
```

## Technologies Used

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package manager
- **[Typer](https://typer.tiangolo.com/)**: Modern CLI framework with rich formatting
- **[faster-whisper](https://github.com/guillaumekln/faster-whisper)**: Efficient speech-to-text transcription
- **[LangChain](https://langchain.readthedocs.io/)**: RAG and LLM integration
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[Rich](https://rich.readthedocs.io/)**: Beautiful terminal formatting

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run ruff check src/
```

## Roadmap

- [x] Basic CLI structure with welcome screen
- [ ] Audio/video transcription with faster-whisper
- [ ] RAG implementation with ChromaDB
- [ ] Question answering system
- [ ] Summarization and note generation
- [ ] Speaker diarization (future)
- [ ] OCR for video frames with diagrams (future)
- [ ] Web interface (future)

## License

MIT License - see LICENSE file for details.
