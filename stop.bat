@echo off
cd /d "%~dp0"
title Meeting Transcriber - Stop

echo ============================================
echo   Meeting Transcriber
echo   Stopping the app...
echo ============================================
echo.

docker compose down
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to stop. Docker may not be running.
    echo.
    pause
    exit /b 1
)

echo.
echo Удаление устаревших (dangling) образов...
docker image prune -f

echo.
echo App stopped. You can close this window.
timeout /t 4 >nul
exit /b 0
