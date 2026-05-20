@echo off
cd /d "%~dp0"
echo Meeting Transcriber - http://localhost:8080
echo.
docker image inspect meeting-transcriber >nul 2>&1
if errorlevel 1 (
    echo Первый запуск -- сборка образа. Займёт 10-20 минут, потом будет быстро.
    echo.
    docker compose up --build
) else (
    docker compose up
)