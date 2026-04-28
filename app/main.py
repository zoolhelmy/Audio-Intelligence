# app/main.py
import streamlit as st
import yaml, os, sys, datetime, gc
from pathlib import Path

# ── Suppress noisy library logs BEFORE any imports that trigger them ──────
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logger_config  import setup_logger
from app.file_manager   import discover_audio_files, ensure_output_path
from app.transcriber    import Transcriber
from app.translator     import Translator
from app.summariser     import Summariser
from app.file_picker    import render_file_picker

# ── Load config ───────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent.parent / 'config.yaml'
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

ROOT    = cfg['project_root']
FOLDERS = cfg['folders']
LOG_CFG = {
    'level':        cfg['logging']['level'],
    'log_dir':      os.path.join(ROOT, FOLDERS['logs']),
    'max_bytes':    cfg['logging']['max_bytes'],
    'backup_count': cfg['logging']['backup_count'],
}
logger = setup_logger('main', LOG_CFG)

# ── Pipeline state ────────────────────────────────────────────────────────
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Audio Intelligence System',
    page_icon='🎙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://img.icons8.com/fluency/96/microphone.png', width=80)
    st.markdown('---')
    st.subheader('⚙️ Setting Info')
    whisper_model = st.selectbox(
        'Whisper Model', ['tiny', 'base', 'small', 'medium'],
        index=3, disabled=True,
    )
    log_level = st.selectbox(
        'Log Level', ['INFO', 'DEBUG'],
        index=0 if cfg['logging']['level'] == 'INFO' else 1,
        disabled=True,
    )
    cfg['logging']['level'] = log_level
    ollama_model = st.text_input(
        'Ollama Model', value=cfg['ollama']['model'], disabled=True,
    )
    cfg['ollama']['model'] = ollama_model
    st.markdown('---')
    st.caption('Offline AI system')

# ── Main panel ────────────────────────────────────────────────────────────
st.title('🎙️ Audio Intelligence System')
st.markdown('Transcribe • Translate • Summarise')
st.markdown('---')

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('📁 Input Audio')
    input_mode = st.radio(
        # 'Select input mode', ['Single File', 'Entire Folder'], # TODO: re-enable folder mode after testing
        'Select input mode', ['Single File'],
        horizontal=True,
        help='Single File: pick one audio file. Entire Folder: process all audio files in a folder.',
    )
    audio_input = render_file_picker(
        default_dir=os.path.join(ROOT, FOLDERS['input_voice']),
        mode=input_mode,
    )
    st.subheader('🌐 Translation (Optional)')
    translate_on = st.toggle('Enable Translation')
    target_lang_label = src_lang = target_lang = None
    if translate_on:
        pairs  = cfg['translation']['available_pairs']
        labels = [p[2] for p in pairs]
        sel    = st.selectbox('Target Language', labels)
        target_lang_label = sel
        target_lang = next(p[1] for p in pairs if p[2] == sel)
        src_lang    = next(p[0] for p in pairs if p[2] == sel)

with col2:
    st.subheader('📝 Summarisation Prompt')
    prompt_dir = os.path.join(ROOT, FOLDERS['prompt_summary'])
    prompts    = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
    use_default = st.toggle('Use saved prompt', value=True)
    if use_default and prompts:
        chosen_prompt = st.selectbox('Select prompt file', prompts)
        prompt_path   = os.path.join(prompt_dir, chosen_prompt)  # type: ignore
        with open(prompt_path, 'r') as pf:
            prompt_preview = pf.read()
        st.text_area('Prompt preview', prompt_preview, height=180, disabled=True)
    else:
        custom_prompt = st.text_area(
            'Custom prompt (use {transcript} as placeholder)',
            value='Summarise the following transcript:\n\n{transcript}',
            height=180,
        )
        prompt_path = None

st.markdown('---')

# ── Run button ────────────────────────────────────────────────────────────
run_btn = st.button(
    '⏳  Processing...' if st.session_state.pipeline_running else '▶  Run Pipeline',
    type='primary',
    use_container_width=True,
    disabled=st.session_state.pipeline_running,
)

if run_btn:
    if not audio_input:
        st.warning('⚠️  Please select an audio file or folder first.')
        st.stop()
    if input_mode == 'Single File' and not os.path.isfile(audio_input):
        st.error(f'File not found: {audio_input}')
        st.stop()
    if input_mode == 'Entire Folder' and not os.path.isdir(audio_input):
        st.error(f'Folder not found: {audio_input}')
        st.stop()
    st.session_state.pipeline_running = True
    st.rerun()

# ── Pipeline ──────────────────────────────────────────────────────────────
if st.session_state.pipeline_running and audio_input:
    st.session_state.pipeline_results = []

    # ── Force CUDA cleanup from any previous run ──────────────────
    # Streamlit reruns keep previous script-scope objects alive in memory.
    # Explicitly clear all CUDA cache before loading new models.
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info('--------------------------------------------------------')
            logger.info(
                f'CUDA memory cleared. '
                f'Free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB / '
                f'{torch.cuda.mem_get_info()[1] / 1024**3:.1f} GB'
            )
    except Exception as e:
        logger.warning(f'CUDA pre-clear failed: {e}')

    try:
        # Discover files
        with st.spinner('Discovering audio files...'):
            try:
                files = discover_audio_files(audio_input)
                st.success(f'Found {len(files)} audio file(s) in {audio_input}')
                logger.info(f'Found {len(files)} file(s) in {audio_input}')
            except Exception as e:
                st.error(str(e))
                logger.error(f'Error discovering audio files: {e}')
                st.session_state.pipeline_running = False
                st.stop()

        # ── Initialise engines — tracked in session state ─────────
        # Storing in session_state ensures old instances are explicitly
        # replaced rather than orphaned in Streamlit's script scope.
        logger.info('Initialising engines...')
        st.session_state['_translator'] = Translator(
            os.path.join(ROOT, FOLDERS['models_nllb'])
        )
        st.session_state['_summariser'] = Summariser(cfg['ollama'])
        translator = st.session_state['_translator']
        summariser = st.session_state['_summariser']

        overall_bar = st.progress(0, text=f'Overall: 0 / {len(files)} file(s)')
        total = len(files)

        # ── Per-file loop ─────────────────────────────────────────
        for idx, audio_file in enumerate(files):

            st.markdown(
                f'---\n**📄 Processing file {idx + 1} of {total}:** `{audio_file.name}`'
            )
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            # ── Transcriber: fresh instance per file ──────────────
            # Required because unload() nullifies self.model.
            # Whisper reloads from local disk cache in ~2–3 seconds.
            # ── VRAM check before loading Whisper ─────────────────
            whisper_cfg = {
                **cfg['whisper'],
                'model_size': whisper_model,
                'model_dir':  os.path.join(ROOT, FOLDERS['models_whisper']),
            }
            # Verify sufficient VRAM before attempting GPU load
            # Whisper medium needs ~1.5 GB VRAM
            WHISPER_VRAM_REQUIRED_GB = 1.6
            if whisper_cfg.get('device') == 'cuda' and torch.cuda.is_available():
                free_vram_gb = torch.cuda.mem_get_info()[0] / 1024**3
                if free_vram_gb < WHISPER_VRAM_REQUIRED_GB:
                    logger.warning(
                        f'Insufficient VRAM ({free_vram_gb:.1f} GB free, '
                        f'{WHISPER_VRAM_REQUIRED_GB} GB required). '
                        f'Falling back to CPU.'
                    )
                    whisper_cfg['device'] = 'cpu'
                    whisper_cfg['fp16']   = False
                else:
                    logger.info(f'VRAM available: {free_vram_gb:.1f} GB — loading to GPU.')

            logger.info(f'Loading Whisper for file {idx + 1}/{total}...')
            transcriber = Transcriber(whisper_cfg)

            # ── 1. TRANSCRIBE ─────────────────────────────────────
            logger.info(f'Transcribing {audio_file.name}...')
            st.markdown('**🎙️ Transcribing...**')
            tx_bar  = st.progress(0.0, text='Starting transcription...')
            tx_stat = st.empty()

            def on_transcribe_progress(fraction: float, message: str):
                tx_bar.progress(min(fraction, 1.0), text=f'🎙️ {message}')
                tx_stat.caption(message)

            result     = transcriber.transcribe(
                audio_file, progress_callback=on_transcribe_progress
            )
            transcript = result['text'].strip()
            tx_bar.progress(1.0, text='✅ Transcription complete')
            tx_stat.empty()

            formatted = transcriber.format_transcript(result, audio_file.name)
            out_tx = ensure_output_path(
                os.path.join(ROOT, FOLDERS['output_transcription']),
                audio_file, f'_{ts}_transcription.txt',
            )
            out_tx.write_text(formatted, encoding='utf-8')
            logger.info(f'Transcription saved: {out_tx}')
            text_to_summarise = transcript

            # Free Whisper before translation/summarisation
            transcriber.unload()
            gc.collect()

            # ── 2. TRANSLATE (optional) ───────────────────────────
            translated = None
            if translate_on:
                # Estimate chunks for user awareness
                estimated_chunks = max(1, len(transcript) // 512)
                logger.info(
                    f'Translating {audio_file.name} to {target_lang_label}... '
                    f'(~{estimated_chunks} chunks estimated)'
                )
                st.markdown(
                    f'**🌐 Translating to {target_lang_label}...** '
                    f'*(~{estimated_chunks} chunks — may take several minutes for long audio)*'
                )
                tr_bar  = st.progress(0.0, text='Loading translation model...')
                tr_stat = st.empty()

                def on_translate_progress(fraction: float, message: str):
                    tr_bar.progress(min(fraction, 1.0), text=f'🌐 {message}')
                    tr_stat.caption(message)

                translated = translator.translate(
                    transcript, src_lang, target_lang,  # type: ignore
                    progress_callback=on_translate_progress,
                )
                text_to_summarise = translated
                tr_bar.progress(1.0, text='✅ Translation complete')
                tr_stat.empty()

                out_tr = ensure_output_path(
                    os.path.join(ROOT, FOLDERS['output_translated']),
                    audio_file, f'_{ts}_translated.txt',
                )
                out_tr.write_text(translated, encoding='utf-8')
                logger.info(f'Translation saved: {out_tr}')

                # Free NLLB before Ollama
                translator.unload()
                gc.collect()

            # ── 3. SUMMARISE ──────────────────────────────────────
            logger.info(f'Summarising {audio_file.name}...')
            with st.spinner('💬 Summarising with Ollama...'):
                if use_default and prompts:
                    full_prompt = summariser.load_prompt(
                        prompt_path, text_to_summarise  # type: ignore
                    )
                else:
                    full_prompt = custom_prompt.replace(
                        '{transcript}', text_to_summarise
                    )
                summary = summariser.summarise(full_prompt)
                summariser.release()   # ← free VRAM immediately after use
                out_sum = ensure_output_path(
                    os.path.join(ROOT, FOLDERS['output_summary']),
                    audio_file, f'_{ts}_summary.txt',
                )
                out_sum.write_text(summary, encoding='utf-8')
                logger.info(f'Summary saved: {out_sum}')

            # Collect result
            st.session_state.pipeline_results.append({
                'file':       audio_file.name,
                'transcript': transcript,
                'translated': translated,
                'summary':    summary,
            })

            overall_bar.progress(
                (idx + 1) / total,
                text=f'Overall: {idx + 1} / {total} file(s) complete',
            )

        st.success('✅ Pipeline complete!')
        logger.info('Pipeline complete.')

    except Exception as e:
        st.error(f'Pipeline error: {e}')
        logger.error(f'Pipeline error: {e}', exc_info=True)

    finally:
        # Explicitly release engine references from session state
        for key in ['_translator', '_summariser']:
            engine = st.session_state.pop(key, None)
            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass
                del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.session_state.pipeline_running = False
        st.rerun()

# ── Results panel ─────────────────────────────────────────────────────────
if st.session_state.pipeline_results:
    st.markdown('---')
    st.subheader('📊 Results')
    for r in st.session_state.pipeline_results:
        with st.expander(f"📄 {r['file']}"):
            st.markdown('**Transcript**')
            st.text_area(
                'Transcript text', r['transcript'], height=150,
                key=f't_{r["file"]}', label_visibility='collapsed',
            )
            if r['translated']:
                st.markdown('**Translation**')
                st.text_area(
                    'Translation text', r['translated'], height=150,
                    key=f'tr_{r["file"]}', label_visibility='collapsed',
                )
            st.markdown('**Summary**')
            st.text_area(
                'Summary text', r['summary'], height=200,
                key=f's_{r["file"]}', label_visibility='collapsed',
            )
