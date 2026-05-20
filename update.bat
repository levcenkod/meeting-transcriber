@echo off
cd /d "%~dp0"
echo Обновление образа после git pull...
echo.
docker compose up --build