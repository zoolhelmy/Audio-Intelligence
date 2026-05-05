# app/summariser.py
import requests
import logging
import json
import time

logger = logging.getLogger('summariser')


class Summariser:
    def __init__(self, cfg: dict):
        self.base_url         = cfg.get('base_url', 'http://localhost:11434')
        self.model            = cfg.get('model', 'mistral:latest')
        self.timeout          = cfg.get('timeout', 300)
        self.max_tok          = cfg.get('max_tokens', 1024)
        self.max_prompt_chars = cfg.get('max_prompt_chars', 12000)

        # Persistent HTTP session — avoids TCP handshake overhead on repeated calls
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    # ── Health check ──────────────────────────────────────────────────────

    def _ping(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            r = self.session.get(f'{self.base_url}/api/tags', timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── Prompt utilities ──────────────────────────────────────────────────

    def load_prompt(self, prompt_path: str, transcript: str) -> str:
        """
        Load a .txt prompt template from prompt-summary/ and inject the transcript.
        The template must contain the placeholder: {transcript}
        """
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template.replace('{transcript}', transcript)

    # ── Summarisation ─────────────────────────────────────────────────────

    def summarise(self, prompt: str) -> str:
        """
        Send prompt to Ollama and return the complete summary text.

        - Truncates oversized prompts to max_prompt_chars to stay within
          the model's context window (configurable in config.yaml).
        - Streams the response token-by-token and assembles the full output.
        - Retries once on HTTP 500 — Ollama runner may need recovery time
          after prior out-of-memory events.
        - Raises RuntimeError with a descriptive message on unrecoverable failure.
        """
        if not self._ping():
            raise ConnectionError(
                'Ollama is not running.\n'
                'Start it with: ollama serve\n'
                'Then verify: curl http://localhost:11434/api/tags'
            )

        # Guard: truncate prompt if it exceeds the configured limit
        if len(prompt) > self.max_prompt_chars:
            logger.warning(
                f'Prompt truncated: {len(prompt)} → {self.max_prompt_chars} chars. '
                f'Increase max_prompt_chars in config.yaml for longer recordings.'
            )
            prompt = prompt[:self.max_prompt_chars] + '\n\n[Transcript truncated for length]'

        payload = {
            'model':   self.model,
            'prompt':  prompt,
            'stream':  True,
            'options': {
                'num_predict': self.max_tok,
                'num_ctx':     4096,
            },
        }

        # Retry once — Ollama runner may need a moment to recover after OOM
        for attempt in range(2):
            logger.info(
                f'Summarisation request → Ollama ({self.model}), attempt {attempt + 1}'
            )
            try:
                with self.session.post(
                    f'{self.base_url}/api/generate',
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = resp.text[:500]
                        logger.error(
                            f'Ollama API error {resp.status_code}: {error_body}'
                        )
                        if attempt == 0:
                            logger.warning('Retrying after 5 seconds...')
                            time.sleep(5)
                            continue
                        raise RuntimeError(
                            f'Ollama returned HTTP {resp.status_code}.\n'
                            f'Details: {error_body}\n'
                            f'Check: ollama list — does the model name in config.yaml '
                            f'match exactly (including :latest tag)?'
                        )

                    # Assemble streamed token chunks into full response
                    parts = []
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            parts.append(data.get('response', ''))
                            if data.get('done', False):
                                break

                    summary = ''.join(parts)
                    logger.info(f'Summarisation complete. Length: {len(summary)} chars')
                    return summary

            except requests.exceptions.Timeout:
                logger.error(f'Ollama timed out after {self.timeout}s')
                raise RuntimeError(
                    f'Ollama timed out after {self.timeout} seconds.\n'
                    f'Options: increase ollama.timeout in config.yaml, '
                    f'or switch to a smaller model (e.g. mistral:latest).'
                )

        return ''  # unreachable — satisfies type checker

    # ── VRAM release ──────────────────────────────────────────────────────

    def release(self):
        """
        Explicitly unload the Ollama model from VRAM immediately after use.

        By default Ollama keeps models loaded for OLLAMA_KEEP_ALIVE (5 minutes).
        Calling release() forces immediate eviction so VRAM is free when
        Whisper reloads to GPU for the next file or run.

        Set OLLAMA_KEEP_ALIVE=0 as a system env var for the same effect globally.
        """
        try:
            self.session.post(
                f'{self.base_url}/api/generate',
                json={'model': self.model, 'keep_alive': 0},
                timeout=10,
            )
            logger.info(f'Ollama model {self.model} unloaded from VRAM.')
        except Exception as e:
            logger.warning(f'Ollama release() failed (non-critical): {e}')
