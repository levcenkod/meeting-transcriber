@echo off
cd /d "%~dp0"
echo Обновление образа после git pull...
echo.
docker compose up --build
echo.
echo Удаление устаревших (dangling) образов...
docker image prune -f
echo Готово.