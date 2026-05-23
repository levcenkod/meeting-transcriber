@echo off
chcp 65001 >nul
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Git не найден в PATH. Установи Git for Windows: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo === git fetch ===
git fetch --all --prune
if errorlevel 1 goto :err

echo.
echo === git pull ===
git pull --ff-only
if errorlevel 1 goto :err

echo.
echo === Текущая версия ===
git log -1 --oneline

echo.
echo Готово.
exit /b 0

:err
echo.
echo [ОШИБКА] git завершился с ошибкой. Проверь вывод выше.
pause
exit /b 1
