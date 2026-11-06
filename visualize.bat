@echo off
echo [Antigravity] Running Trade Visualizer with virtual environment...
.venv\Scripts\python.exe src\visualize_trades.py
if exist trade_executions.html (
    echo [Antigravity] Opening trade_executions.html in your default web browser...
    start trade_executions.html
)
