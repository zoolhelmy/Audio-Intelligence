# app/transcriber.py
import gc
import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to launch Triton kernels",
    category=UserWarning,
    module="whisper.timing"
)

import whisper
import torch
import os
import tempfile
import logging
import datetime
from pathlib import Path
from pydub import AudioSegment
from typing import Callable, Optional

logger = logging.getLogger('transcriber')


class Transcriber:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        logger.info(f'Loading Whisper model: {cfg["model_size"]} on {self.device}')
        self.model = whisper.load_model(
            cfg['model_size'],
            device=self.device,
            download_root=cfg.get('model_dir', None)  # type: ignore
        )
        logger.info('Whisper model loaded successfully.')

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Memory management ─────────────────────────────────────────────────
    def unload(self):
        """Release Whisper model from GPU and CPU memory."""
        if self.model is None:
            logger.debug('unload() called but model already None — skipping.')
            return
        del self.model
        self.model = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info('Whisper model unloaded from memory.')

    # ── Audio helpers ──────────────────────────────────────────────────────
    def _convert_to_wav(self, audio_path: Path) -> str:
        """Convert any supported audio format to 16 kHz mono WAV."""
        logger.debug(f'Converting {audio_path.name} to 16kHz mono WAV')
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio.export(tmp.name, format='wav')
        return tmp.name

    def _fmt_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    # ── Core transcription ────────────────────────────────────────────────
    def transcribe(self, audio_path: Path, language: str = None,  # type: ignore
                   progress_callback: Optional[Callable] = None) -> dict:
        """
        Transcribe audio with optional progress reporting.
        progress_callback(fraction: float, message: str)

        NOTE: This instance must not have been unload()ed before calling.
              main.py creates a fresh Transcriber per file for this reason.
        """
        if self.model is None:
            raise RuntimeError(
                'Transcriber.model is None — create a new Transcriber instance '
                'instead of reusing one after unload().'
            )

        wav_path = self._convert_to_wav(audio_path)
        try:
            logger.info(f'Transcribing: {audio_path.name}')

            options = {
                'beam_size':       self.cfg.get('beam_size', 5),
                'word_timestamps': self.cfg.get('word_timestamps', True),
                'fp16':            self.cfg.get('fp16', True) and self.device == 'cuda',
                'verbose':         False,
            }
            if language:
                options['language'] = language

            # Get audio duration for progress percentage
            total_duration = None
            try:
                import soundfile as sf  # type: ignore
                total_duration = sf.info(wav_path).duration
            except Exception:
                pass

            if progress_callback:
                progress_callback(0.02, 'Loading audio...')

            # Run full transcription
            result = self.model.transcribe(wav_path, **options)  # type: ignore

            # Replay progress through completed segments
            if progress_callback:
                segments = result.get('segments', [])
                dur      = total_duration or (segments[-1]['end'] if segments else 1)
                for i, seg in enumerate(segments):
                    frac = min(seg['end'] / dur, 1.0)
                    progress_callback(
                        0.02 + frac * 0.98,
                        f'Transcribed {self._fmt_time(seg["end"])} / {self._fmt_time(dur)}'
                        f' — segment {i + 1}/{len(segments)}'
                    )
                if not segments:
                    progress_callback(1.0, 'Transcription complete.')

            logger.info(f'Transcription complete. Language: {result["language"]}')
            return result

        finally:
            os.unlink(wav_path)

    # ── Transcript formatter ──────────────────────────────────────────────
    def format_transcript(self, result: dict, audio_filename: str) -> str:
        """
        Format Whisper result into a structured report with:
        - Metadata header
        - Timestamps per segment
        - Heuristic speaker-turn detection (pause > 1.5s = new speaker)
        - Full plain-text section at the bottom
        """
        lines = []

        lines.append('=' * 70)
        lines.append('AUDIO TRANSCRIPTION REPORT')
        lines.append('=' * 70)
        lines.append(f'File            : {audio_filename}')
        lines.append(f'Generated       : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'Detected Language: {result.get("language", "unknown").upper()}')
        lines.append(f'Whisper Model   : {self.cfg.get("model_size", "unknown")}')

        segments = result.get('segments', [])
        if segments:
            total_sec = segments[-1]['end']
            h, rem = divmod(int(total_sec), 3600)
            m, s   = divmod(rem, 60)
            lines.append(f'Duration        : {h:02d}:{m:02d}:{s:02d}')
        lines.append(f'Total Segments  : {len(segments)}')
        lines.append('=' * 70)
        lines.append('')
        lines.append('TRANSCRIPT')
        lines.append('-' * 70)

        speaker_num = 1
        prev_end    = 0.0
        THRESHOLD   = 1.5  # seconds pause → new speaker

        def fmt(s: float) -> str:
            mi, se = divmod(int(s), 60)
            hr, mi = divmod(mi, 60)
            return f'{hr:02d}:{mi:02d}:{se:02d}'

        for seg in segments:
            start = seg['start']
            end   = seg['end']
            text  = seg['text'].strip()

            if start - prev_end > THRESHOLD and prev_end > 0:
                speaker_num += 1
                lines.append('')

            lines.append(f'[{fmt(start)} → {fmt(end)}]  SPEAKER {speaker_num:02d}')
            lines.append(f'  {text}')
            lines.append('')
            prev_end = end

        lines.append('=' * 70)
        lines.append('FULL TEXT (plain)')
        lines.append('=' * 70)
        lines.append(result.get('text', '').strip())

        return '\n'.join(lines)
