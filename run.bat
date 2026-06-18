@echo off
cd /d "%~dp0"
title Meeting Transcriber - Start

echo ============================================
echo   Meeting Transcriber
echo   Starting and checking for updates...
echo ============================================
echo.

rem -- 1. Make sure Docker is running (start Docker Desktop if not) --
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready

echo Docker is not running. Starting Docker Desktop...
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
) else (
    echo [WARNING] Docker Desktop not found. Please start it manually and retry.
)

echo Waiting for Docker to become ready (up to 3 minutes)...
set /a _tries=0
:wait_docker
timeout /t 3 >nul
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready
set /a _tries+=1
if %_tries% geq 60 (
    echo.
    echo [ERROR] Docker did not start. Open Docker Desktop manually and try again.
    echo.
    pause
    exit /b 1
)
goto wait_docker

:docker_ready
echo Docker is ready.
echo.

rem -- 2. Pull updates (if git and internet are available) --
where git >nul 2>nul
if errorlevel 1 (
    echo [info] Git not found -- skipping update check.
) else (
    if exist ".git" (
        echo Checking for updates...
        git pull --ff-only
        if errorlevel 1 echo [info] Update failed -- starting current version.
    )
)
echo.

rem -- 3. Build (if needed) and start in the background --
echo Starting the app...
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Check the output above.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Done! Opening http://localhost:8080
echo   To shut down, run "Stop".
echo ============================================
start "" "http://localhost:8080"

rem This window can be closed -- the app runs in the background.
timeout /t 6 >nul
exit /b 0
