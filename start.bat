@echo off
cd /d "%~dp0"
echo Meeting Transcriber - http://localhost:8080
echo.

rem GPU-блок подключается только там, где есть NVIDIA (иначе compose не стартует)
set COMPOSE_FILES=-f docker-compose.yml
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml
    echo GPU: NVIDIA обнаружена, подключаю docker-compose.gpu.yml
) else (
    echo GPU: не найдена, работаем на CPU
)
echo.

docker image inspect meeting-transcriber >nul 2>&1
if errorlevel 1 (
    echo Первый запуск -- сборка образа. Займёт 10-20 минут, потом будет быстро.
    echo.
    docker compose %COMPOSE_FILES% up --build
) else (
    docker compose %COMPOSE_FILES% up
)
