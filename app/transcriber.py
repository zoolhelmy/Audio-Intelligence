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
import logging
import datetime
from pathlib import Path
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
        """
        Fully release Whisper model from GPU and CPU memory.
        Use only when the model will not be needed again in this session.
        For temporary VRAM relief between pipeline stages, use offload_to_cpu()
        instead — it keeps weights in CPU RAM for fast GPU reload.
        """
        if self.model is None:
            logger.debug('unload() called but model already None — skipping.')
            return
        del self.model
        self.model = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info('Whisper model fully unloaded from memory.')

    def offload_to_cpu(self):
        """
        Move Whisper model tensors from GPU VRAM to CPU RAM.
        Frees VRAM for Ollama/NLLB without requiring a full disk reload next time.
        Reload to GPU with reload_to_gpu().
        """
        if self.model is None:
            logger.debug('offload_to_cpu() called but model is None — skipping.')
            return
        if self.device != 'cuda':
            return  # already on CPU, nothing to do
        current_device = str(next(self.model.parameters()).device)
        if current_device == 'cpu':
            logger.debug('Whisper already on CPU — skipping offload.')
            return
        self.model.to('cpu')
        torch.cuda.empty_cache()
        logger.info('Whisper offloaded to CPU RAM (VRAM freed).')

    def reload_to_gpu(self):
        """
        Move Whisper model tensors from CPU RAM back to GPU VRAM.
        Only valid after offload_to_cpu() — model must still be in CPU RAM.
        """
        if self.model is None:
            raise RuntimeError(
                'Cannot reload_to_gpu() — model is None. '
                'Create a new Transcriber instance.'
            )
        if self.device != 'cuda' or not torch.cuda.is_available():
            return  # running in CPU mode, nothing to reload
        current_device = str(next(self.model.parameters()).device)
        if current_device != 'cpu':
            logger.debug('Whisper already on GPU — skipping reload.')
            return
        self.model.to('cuda')
        torch.cuda.synchronize()
        logger.info('Whisper reloaded to GPU from CPU cache.')

    # ── Audio helpers ──────────────────────────────────────────────────────

    def _fmt_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    # ── Core transcription ────────────────────────────────────────────────

    def transcribe(self, audio_path: Path, language: str = None,  # type: ignore
                   progress_callback: Optional[Callable] = None) -> dict:
        """
        Transcribe audio with optional real-time progress reporting.
        progress_callback(fraction: float, message: str)

        Passes audio directly to Whisper — no intermediate WAV conversion.
        Whisper uses FFmpeg internally to handle all supported formats.

        NOTE: model must be loaded and on the correct device.
              If offloaded to CPU, call reload_to_gpu() first for GPU speed.
        """
        if self.model is None:
            raise RuntimeError(
                'Transcriber.model is None. '
                'Create a new Transcriber instance — do not reuse after unload().'
            )

        logger.info(f'Transcribing: {audio_path.name}')

        options = {
            'beam_size':       self.cfg.get('beam_size', 1),
            'word_timestamps': self.cfg.get('word_timestamps', True),
            'fp16':            self.cfg.get('fp16', True) and self.device == 'cuda',
            'verbose':         False,
        }
        if language:
            options['language'] = language

        # Get audio duration for accurate progress percentage
        total_duration = None
        try:
            import soundfile as sf  # type: ignore
            total_duration = sf.info(str(audio_path)).duration
        except Exception:
            pass  # falls back to segment-end-time estimate below

        if progress_callback:
            progress_callback(0.02, 'Loading audio...')

        # Pass source path directly — Whisper calls FFmpeg internally
        result = self.model.transcribe(str(audio_path), **options)  # type: ignore

        # Replay progress through completed segments for UI feedback
        if progress_callback:
            segments = result.get('segments', [])
            dur      = total_duration or (segments[-1]['end'] if segments else 1)  # type: ignore
            for i, seg in enumerate(segments):
                frac = min(seg['end'] / dur, 1.0)  # type: ignore
                progress_callback(
                    0.02 + frac * 0.98,
                    f'Transcribed {self._fmt_time(seg["end"])} / {self._fmt_time(dur)}'  # type: ignore
                    f' — segment {i + 1}/{len(segments)}'
                )
            if not segments:
                progress_callback(1.0, 'Transcription complete.')

        logger.info(f'Transcription complete. Language: {result["language"]}')
        return result

    # ── Transcript formatter ──────────────────────────────────────────────

    def format_transcript(self, result: dict, audio_filename: str) -> str:
        """
        Format Whisper result into a structured report:
        - Metadata header (file, language, model, duration, segments)
        - Timestamped segments with heuristic speaker-turn detection
          (pause > 1.5s between segments = new speaker)
        - Full plain-text section at the bottom for summarisation
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
        THRESHOLD   = 1.5  # seconds gap → new speaker turn

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
