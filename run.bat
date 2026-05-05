@echo off

:: Kill any orphan streamlit
taskkill /f /im "streamlit.exe" /t

:: Kill any orphan ollama
taskkill /f /im "ollama.exe" /t

:: 1. Start Ollama in a minimized window with dated log
start /min powershell -Command "$dt = Get-Date -Format 'yyyyMMdd'; ollama serve 2>&1 | Tee-Object -FilePath \"C:\AudioIntel\logs\ollama_server_$dt.log\" -Append"

:: 2. Wait for Ollama
timeout /t 3

:: 3. Run Streamlit with dated log and environment activation
powershell -NoExit -ExecutionPolicy Bypass -Command "$dt = Get-Date -Format 'yyyyMMdd'; .\venv\Scripts\Activate.ps1; streamlit run app/main.py 2>&1 | Tee-Object -FilePath \"C:\AudioIntel\logs\streamlit_run_$dt.log\" -Append"

pause
