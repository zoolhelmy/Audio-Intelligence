# app/file_manager.py
import os
from pathlib import Path
from typing import List
 
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac', '.opus'}
 
def discover_audio_files(path: str) -> List[Path]:
    """Return a list of supported audio file paths.
    path may be a single file or a directory.
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [p]
        else:
            raise ValueError(f'Unsupported file format: {p.suffix}')
    elif p.is_dir():
        files = sorted([
            f for f in p.rglob('*')
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        if not files:
            raise FileNotFoundError(f'No supported audio files found in {path}')
        return files
    else:
        raise FileNotFoundError(f'Path does not exist: {path}')
 
def ensure_output_path(output_dir: str, original: Path, suffix: str) -> Path:
    """Construct an output path mirroring the input filename."""
    out = Path(output_dir) / (original.stem + suffix)
    os.makedirs(out.parent, exist_ok=True)
    return out
