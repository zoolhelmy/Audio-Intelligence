# app/file_picker.py
import streamlit as st
import os
import threading
import logging

logger = logging.getLogger('file_picker')

AUDIO_EXTENSIONS = [
    ('Audio Files', '*.wav *.mp3 *.flac *.m4a *.ogg *.wma *.aac *.opus'),
    ('WAV Files', '*.wav'),
    ('MP3 Files', '*.mp3'),
    ('FLAC Files', '*.flac'),
    ('M4A Files', '*.m4a'),
    ('All Files', '*.*'),
]

# ── Fixed session state keys — one per mode ───────────────────────────────
KEY_FILE   = 'picker_selected_file'
KEY_FOLDER = 'picker_selected_folder'


def _run_file_dialog(result: list, initial_dir: str):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title='Select Audio File',
            initialdir=initial_dir,
            filetypes=AUDIO_EXTENSIONS,
        )
        root.destroy()
        result.append(path or '')
    except Exception as e:
        logger.error(f'File dialog error: {e}')
        result.append('')


def _run_folder_dialog(result: list, initial_dir: str):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        path = filedialog.askdirectory(
            title='Select Folder Containing Audio Files',
            initialdir=initial_dir,
        )
        root.destroy()
        result.append(path or '')
    except Exception as e:
        logger.error(f'Folder dialog error: {e}')
        result.append('')


def render_file_picker(default_dir: str, mode: str = 'Single File') -> str:
    """
    Render file/folder picker UI.
    State is stored ONLY in st.session_state — never in widget values directly.
    This ensures the selected path survives all widget reruns.
    Returns the currently selected path string.
    """
    # Initialise session state keys on first load
    if KEY_FILE   not in st.session_state: st.session_state[KEY_FILE]   = ''
    if KEY_FOLDER not in st.session_state: st.session_state[KEY_FOLDER] = ''

    is_file_mode = (mode == 'Single File')
    state_key    = KEY_FILE if is_file_mode else KEY_FOLDER
    current      = st.session_state[state_key]  # always read from session state

    if is_file_mode:
        # ── File picker ───────────────────────────────────────────
        col_browse, col_clear = st.columns([3, 1])

        with col_browse:
            browse_clicked = st.button(
                '📂  Browse for Audio File',
                key='btn_browse_file',
                use_container_width=True,
            )
        with col_clear:
            clear_clicked = st.button(
                '✕ Clear', key='btn_clear_file',
                use_container_width=True, type='secondary',
            )

        if browse_clicked:
            result = []
            t = threading.Thread(
                target=_run_file_dialog,
                args=(result, current or default_dir)
            )
            t.start(); t.join(timeout=60)
            if result and result[0]:
                st.session_state[KEY_FILE] = result[0]
                current = result[0]
                logger.info(f'File selected: {current}')

        if clear_clicked:
            st.session_state[KEY_FILE] = ''
            current = ''

        # Show selected path in a disabled text box (display only)
        # Using disabled=True prevents Streamlit widget state from overwriting session state
        st.text_input(
            'Selected file',
            value=current,
            key='display_file_path',
            disabled=True,
            label_visibility='collapsed',
            placeholder='No file selected — click Browse or type a path below',
        )

        # Optional manual override — separate widget with its own key
        manual = st.text_input(
            '✏️  Or enter path manually',
            value='',
            key='manual_file_path',
            placeholder=r'C:\AudioIntel\input-voice\meeting.mp3',
        )
        if manual and manual != current:
            st.session_state[KEY_FILE] = manual
            current = manual

        # File preview
        if current and os.path.isfile(current):
            size_mb = os.path.getsize(current) / (1024 * 1024)
            ext = os.path.splitext(current)[1].upper()
            st.success(
                f'✅  **{os.path.basename(current)}** — {ext}, {size_mb:.1f} MB'
            )
        elif current:
            st.error('⚠️  File not found. Check the path.')

    else:
        # ── Folder picker ─────────────────────────────────────────
        col_browse, col_clear = st.columns([3, 1])

        with col_browse:
            browse_clicked = st.button(
                '📁  Browse for Audio Folder',
                key='btn_browse_folder',
                use_container_width=True,
            )
        with col_clear:
            clear_clicked = st.button(
                '✕ Clear', key='btn_clear_folder',
                use_container_width=True, type='secondary',
            )

        if browse_clicked:
            result = []
            t = threading.Thread(
                target=_run_folder_dialog,
                args=(result, current or default_dir)
            )
            t.start(); t.join(timeout=60)
            if result and result[0]:
                st.session_state[KEY_FOLDER] = result[0]
                current = result[0]
                logger.info(f'Folder selected: {current}')

        if clear_clicked:
            st.session_state[KEY_FOLDER] = ''
            current = ''

        # Display only — disabled prevents widget overwriting session state
        st.text_input(
            'Selected folder',
            value=current,
            key='display_folder_path',
            disabled=True,
            label_visibility='collapsed',
            placeholder='No folder selected — click Browse or type a path below',
        )

        # Optional manual override
        manual = st.text_input(
            '✏️  Or enter folder path manually',
            value='',
            key='manual_folder_path',
            placeholder=r'C:\AudioIntel\input-voice',
        )
        if manual and manual != current:
            st.session_state[KEY_FOLDER] = manual
            current = manual

        # Folder preview
        if current and os.path.isdir(current):
            AUDIO_EXT = {'.wav','.mp3','.flac','.m4a','.ogg','.wma','.aac','.opus'}
            audio_files = sorted([
                f for f in os.listdir(current)
                if os.path.splitext(f)[1].lower() in AUDIO_EXT
            ])
            if audio_files:
                st.success(f'✅  **{len(audio_files)} audio file(s)** found in folder')
                with st.expander(f'📋 Preview ({len(audio_files)} files)', expanded=False):
                    for i, f in enumerate(audio_files):
                        fpath = os.path.join(current, f)
                        size_mb = os.path.getsize(fpath) / (1024 * 1024)
                        ext = os.path.splitext(f)[1].upper()
                        st.markdown(f'`{i+1:02d}.`  **{f}** — {ext}, {size_mb:.1f} MB')
            else:
                st.warning('⚠️  No supported audio files found in this folder.')
        elif current:
            st.error('⚠️  Folder not found. Check the path.')

    return st.session_state[state_key]