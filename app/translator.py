# app/translator.py
import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Callable, Optional

logger = logging.getLogger('translator')

# NLLB language codes for supported pairs
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
        self._cache = {}

    def _load_nllb(self):
        if 'nllb' not in self._cache:
            path = os.path.normpath(
                os.path.join(self.models_dir, 'nllb-200-distilled-600M'))
            if not os.path.isdir(path):
                raise FileNotFoundError(f'NLLB model not found: {path}')
            logger.info(f'Loading NLLB-200 on CPU (preserving VRAM for Whisper/Ollama)')
            tok = AutoTokenizer.from_pretrained(path)
            # Force CPU — VRAM is shared with Whisper and Ollama on RTX 4050
            mdl = AutoModelForSeq2SeqLM.from_pretrained(path, device_map='cpu')
            self._cache['nllb'] = (tok, mdl)
            logger.info('NLLB-200 loaded on CPU.')
        return self._cache['nllb']

    def translate(self, text: str, src: str, tgt: str,
              chunk_size: int = 1024, # Adjust chunk_size for performance/quality trade-off; 512 is a good starting point
              progress_callback=None) -> str:
        """
        Translate plain text using NLLB-200 with optional per-chunk progress.
        progress_callback(fraction: float, message: str)
        """
        if not text or not text.strip():
            return ''

        src_code = NLLB_LANG_CODES.get(src)
        tgt_code = NLLB_LANG_CODES.get(tgt)
        if not src_code or not tgt_code:
            raise ValueError(f'Unsupported pair: {src}→{tgt}')

        tok, mdl = self._load_nllb()
        tok.src_lang = src_code

        # Sentence-aware chunking
        sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if s.strip()]
        chunks, current, current_len = [], [], 0
        for sent in sentences:
            sp = sent + '.'
            if current_len + len(sp) > chunk_size and current:
                chunks.append(' '.join(current))
                current, current_len = [sp], len(sp)
            else:
                current.append(sp); current_len += len(sp)
        if current:
            chunks.append(' '.join(current))

        total_chunks = len(chunks)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            # Fire progress at start of each chunk
            if progress_callback:
                progress_callback(
                    i / total_chunks,
                    f'Translating chunk {i+1} of {total_chunks}...'
                )

            logger.debug(f'Translating chunk {i+1}/{total_chunks}')
            inputs = tok(chunk, return_tensors='pt', padding=True,
                         truncation=True, max_length=512)
            # inputs stay on CPU — model is CPU-bound
            tgt_id = tok.convert_tokens_to_ids(tgt_code)
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs, forced_bos_token_id=tgt_id,
                    max_length=512, num_beams=2, early_stopping=True # Adjust num_beams for quality/speed trade-off; 2 is a good starting point
                )
            translated_parts.append(
                tok.batch_decode(outputs, skip_special_tokens=True)[0]
            )

        if progress_callback:
            progress_callback(1.0, f'Translation complete — {total_chunks} chunks processed.')

        result = ' '.join(translated_parts)
        logger.info(f'Translation: {len(text)} chars → {len(result)} chars')
        return result
    
    def unload(self):
        """Release NLLB model from memory."""
        import gc
        if 'nllb' in self._cache:
            del self._cache['nllb']
            self._cache.clear()
        gc.collect()
        # Clear torch CUDA cache if model was on GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info('Translation model unloaded from memory.')