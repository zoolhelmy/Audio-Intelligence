# app/summariser.py
import requests, logging, json
from pathlib import Path
import time

logger = logging.getLogger('summariser')
 
class Summariser:
    def __init__(self, cfg: dict):
        self.base_url = cfg.get('base_url', 'http://localhost:11434')
        self.model    = cfg.get('model', 'llama3.1')
        self.timeout  = cfg.get('timeout', 120)
        self.max_tok  = cfg.get('max_tokens', 1024)
        self.max_prompt_chars = cfg.get('max_prompt_chars', 12000)
    def _ping(self) -> bool:
        try:
            r = requests.get(f'{self.base_url}/api/tags', timeout=5)
            return r.status_code == 200
        except Exception:
            return False
 
    def load_prompt(self, prompt_path: str, transcript: str) -> str:
        """Load a .txt prompt template and inject the transcript."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template.replace('{transcript}', transcript)
 
    def summarise(self, prompt: str) -> str:
        """Send prompt to Ollama and return the summary text."""
        if not self._ping():
            raise ConnectionError('Ollama is not running. Start it with: ollama serve')

        MAX_PROMPT_CHARS = self.max_prompt_chars
        if len(prompt) > MAX_PROMPT_CHARS:
            logger.warning(f'Prompt truncated from {len(prompt)} to {MAX_PROMPT_CHARS} chars')
            prompt = prompt[:MAX_PROMPT_CHARS] + '\n\n[Transcript truncated for length]'

        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': True,
            'options': {
                'num_predict': self.max_tok,
                'num_ctx': 4096
            }
        }

        # Retry once on 500 — Ollama runner may need a moment to recover
        for attempt in range(2):
            logger.info(f'Summarisation request to Ollama ({self.model}), attempt {attempt + 1}')
            try:
                with requests.post(
                    f'{self.base_url}/api/generate',
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                ) as resp:
                    if resp.status_code != 200:
                        error_body = resp.text[:500]
                        logger.error(f'Ollama API error {resp.status_code}: {error_body}')
                        if attempt == 0:
                            logger.warning('Retrying after 5 seconds...')
                            time.sleep(5)
                            continue
                        raise RuntimeError(
                            f'Ollama returned HTTP {resp.status_code}.\n'
                            f'Details: {error_body}\n'
                            f'Check: ollama list — is model name correct in config.yaml?'
                        )
                    full_response = []
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            full_response.append(data.get('response', ''))
                            if data.get('done', False):
                                break
                    summary = ''.join(full_response)
                    logger.info(f'Summarisation complete. Length: {len(summary)} chars')
                    return summary
            except requests.exceptions.Timeout:
                logger.error(f'Ollama request timed out after {self.timeout}s')
                raise RuntimeError(
                    f'Ollama timed out after {self.timeout} seconds. '
                    f'Try increasing timeout in config.yaml or use a smaller model.'
                )
        return ''  # unreachable but satisfies type checker

    def release(self):
        """Explicitly unload the model from Ollama VRAM after use."""
        try:
            requests.post(
                f'{self.base_url}/api/generate',
                json={'model': self.model, 'keep_alive': 0},
                timeout=10
            )
            logger.info(f'Ollama model {self.model} unloaded from VRAM.')
        except Exception as e:
            logger.warning(f'Ollama release failed: {e}')