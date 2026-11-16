@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_ENDPOINT=https://hf-mirror.com
title RAG Web UI
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py
pause
