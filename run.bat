@echo off

:: 1. Create logs directory if it doesn't exist
if not exist "C:\AudioIntel\logs" mkdir "C:\AudioIntel\logs"

:: 2. Start Ollama in a minimized window with dated log
start /min powershell -Command "$dt = Get-Date -Format 'yyyyMMdd'; ollama serve 2>&1 | Tee-Object -FilePath \"C:\AudioIntel\logs\ollama_server_$dt.log\" -Append"

:: 3. Wait for Ollama
timeout /t 5

:: 4. Run Streamlit with dated log and environment activation
powershell -NoExit -ExecutionPolicy Bypass -Command "$dt = Get-Date -Format 'yyyyMMdd'; .\venv\Scripts\Activate.ps1; streamlit run app/main.py 2>&1 | Tee-Object -FilePath \"C:\AudioIntel\logs\streamlit_run_$dt.log\" -Append"

pause
