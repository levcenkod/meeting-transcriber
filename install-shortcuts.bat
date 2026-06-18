@echo off
cd /d "%~dp0"
title Install Desktop Shortcuts

echo ============================================
echo   Creating Desktop shortcuts:
echo     - "Meeting Transcriber - Start"
echo     - "Meeting Transcriber - Stop"
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-shortcuts.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create shortcuts.
    echo.
    pause
    exit /b 1
)

echo.
echo Done! Two shortcuts are now on the Desktop.
echo Just double-click them from now on.
echo.
pause
exit /b 0
