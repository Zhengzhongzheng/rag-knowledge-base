@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
title RAG Knowledge Base
cd /d "%~dp0"
.venv\Scripts\python.exe main.py
pause
