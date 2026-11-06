@echo off
title Quant Forex RL - 24/5 Execution Watchdog
echo ===================================================
echo   Quant Forex RL - 24/5 Production Watchdog
echo ===================================================
cd /d "%~dp0\.."

if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

echo Starting supervisor watchdog with %PYTHON_BIN%...
%PYTHON_BIN% scripts\run_daemon_watchdog.py
pause
