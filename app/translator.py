# app/translator.py
import os
import re
import gc
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Callable, Optional

logger = logging.getLogger('translator')

# NLLB-200 BCP-47 language codes
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_CODES = {
    'en': 'eng_Latn',
    'ms': 'zsm_Latn',   # Bahasa Melayu (standard Malay)
    'zh': 'zho_Hans',   # Simplified Chinese
    'ar': 'arb_Arab',   # Modern Standard Arabic
    'fr': 'fra_Latn',   # French
    'de': 'deu_Latn',   # German
}


class Translator:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._cache     = {}  # loaded model stays here until unload()

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_nllb(self):
        """
        Lazy-load the NLLB-200 model on first translation call.
        Forced to CPU to preserve VRAM for Whisper and Ollama on RTX 4050.
        """
        if 'nllb' not in self._cache:
            path = os.path.normpath(
                os.path.join(self.models_dir, 'nllb-200-distilled-600M')
            )
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f'NLLB-200 model not found at: {path}\n'
                    f'Run scripts/download_nllb.py while online first.'
                )
            logger.info(f'Loading NLLB-200 on CPU (preserving VRAM for Whisper/Ollama)...')
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                path,
                device_map='cpu',
                low_cpu_mem_usage=True,
            )
            self._cache['nllb'] = (tok, mdl)
            logger.info('NLLB-200 loaded on CPU.')
        return self._cache['nllb']

    # ── Memory management ─────────────────────────────────────────────────

    def unload(self):
        """
        Release NLLB-200 from CPU RAM.
        Call after translation completes, before Ollama summarisation,
        to free ~2.4 GB RAM for the LLM inference step.
        """
        if 'nllb' in self._cache:
            del self._cache['nllb']
            self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info('NLLB-200 unloaded from CPU RAM.')

    # ── Translation ───────────────────────────────────────────────────────

    def translate(self, text: str, src: str, tgt: str,
                  chunk_size: int = 1024,
                  progress_callback: Optional[Callable] = None) -> str:
        """
        Translate plain text using NLLB-200.

        Uses token-aware sentence chunking:
        - Splits on sentence boundaries (. ! ?) rather than fixed character counts
        - Counts actual token IDs per sentence to stay within model context
        - Prevents mid-sentence truncation and maintains translation quality
          across languages with different token densities (e.g. Arabic vs English)

        chunk_size: maximum number of tokens per chunk (default 1024)
        progress_callback(fraction: float, message: str)

        IMPORTANT: Pass plain text only — never the formatted transcription report.
        """
        if not text or not text.strip():
            return ''

        src_code = NLLB_LANG_CODES.get(src)
        tgt_code = NLLB_LANG_CODES.get(tgt)
        if not src_code or not tgt_code:
            raise ValueError(
                f'Unsupported language pair: {src} → {tgt}\n'
                f'Supported codes: {list(NLLB_LANG_CODES.keys())}'
            )

        tok, mdl = self._load_nllb()
        tok.src_lang = src_code
        tgt_id = tok.convert_tokens_to_ids(tgt_code)

        # ── Token-aware sentence chunking ─────────────────────────────────
        # Split on sentence-ending punctuation + whitespace
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
            if s.strip()
        ]

        chunks, current, current_len = [], [], 0
        for sent in sentences:
            sent_ids = tok(sent, add_special_tokens=False).input_ids
            sent_len = len(sent_ids)
            if current_len + sent_len > chunk_size and current:
                chunks.append(' '.join(current))
                current, current_len = [sent], sent_len
            else:
                current.append(sent)
                current_len += sent_len
        if current:
            chunks.append(' '.join(current))

        total_chunks    = len(chunks)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    i / total_chunks,
                    f'Translating chunk {i + 1} of {total_chunks}...'
                )
            logger.debug(f'Translating chunk {i + 1}/{total_chunks}')

            inputs = tok(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=chunk_size,
            )
            # Inputs remain on CPU — model is CPU-bound
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_length=512,
                    num_beams=2,       # 2 beams: good quality/speed balance on CPU
                    early_stopping=True,
                )
            translated_parts.append(
                tok.batch_decode(outputs, skip_special_tokens=True)[0]
            )

        if progress_callback:
            progress_callback(
                1.0,
                f'Translation complete — {total_chunks} chunk(s) processed.'
            )

        result = ' '.join(translated_parts)
        logger.info(f'Translation complete: {len(text)} chars → {len(result)} chars')
        return result
