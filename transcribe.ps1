# transcribe.ps1 — запуск транскрибации через WhisperX в Docker
# Использование:
#   .\transcribe.ps1 .\input\meeting.mp3       # один файл
#   .\transcribe.ps1 .\input                   # batch-режим (все аудио/видео)

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Путь к директории скрипта (корень проекта)
$ProjectRoot = $PSScriptRoot
$LogFile     = Join-Path $ProjectRoot "logs\transcriber.log"

# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

function Write-Info  { param([string]$msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Err   { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Warn  { param([string]$msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }

function Exit-WithError {
    param([string]$msg)
    Write-Err $msg
    exit 1
}

function Write-Log {
    param([string]$Level, [string]$FileName, [string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$timestamp | $($Level.ToUpper().PadRight(7)) | $FileName | $Message"
    try { Add-Content -Path $script:LogFile -Value $line -Encoding UTF8 } catch {}
}

function Get-Category {
    param([string]$FileName)
    $base = $FileName.ToLower()
    if ($base -match 'logistics') { return 'Логистика' }
    if ($base -match 'marketing') { return 'Маркетинг' }
    if ($base -match 'finance')   { return 'Финансы'   }
    return 'Общее'
}

function Move-FileToDir {
    param([string]$SourcePath, [string]$DestDir)
    if (-not (Test-Path $SourcePath)) { return $null }
    $name = [System.IO.Path]::GetFileName($SourcePath)
    $dest = Join-Path $DestDir $name
    if (Test-Path $dest) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $ext   = [System.IO.Path]::GetExtension($name)
        $stem  = [System.IO.Path]::GetFileNameWithoutExtension($name)
        $dest  = Join-Path $DestDir "${stem}_${stamp}${ext}"
    }
    Move-Item -Path $SourcePath -Destination $dest -Force
    return $dest
}

# ─────────────────────────────────────────────
# Проверка: Docker запущен?
# ─────────────────────────────────────────────

Write-Info "Проверка Docker..."
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Exit-WithError "Docker не запущен или не установлен. Запустите Docker Desktop и попробуйте снова."
}
Write-Ok "Docker доступен."

# ─────────────────────────────────────────────
# Проверка .env
# ─────────────────────────────────────────────

$EnvFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Exit-WithError "Файл .env не найден. Скопируйте .env.example в .env и заполните HF_TOKEN:`n  copy .env.example .env"
}

# Читаем .env вручную, чтобы проверить HF_TOKEN
$EnvVars = @{}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -match '^([^#=]+)=(.*)$') {
        $EnvVars[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}

if (-not $EnvVars.ContainsKey("HF_TOKEN") -or $EnvVars["HF_TOKEN"] -eq "" -or $EnvVars["HF_TOKEN"] -like "hf_xxx*") {
    Exit-WithError "HF_TOKEN не задан или содержит placeholder. Укажите реальный токен Hugging Face в .env."
}
Write-Ok ".env проверен."

# ─────────────────────────────────────────────
# Создание папок input/ и output/
# ─────────────────────────────────────────────

$InputDir     = Join-Path $ProjectRoot "input"
$OutputDir    = Join-Path $ProjectRoot "output"
$ProcessedDir = Join-Path $ProjectRoot "processed"
$FailedDir    = Join-Path $ProjectRoot "failed"
$LogsDir      = Join-Path $ProjectRoot "logs"

foreach ($dir in @($InputDir, $OutputDir, $ProcessedDir, $FailedDir, $LogsDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Info "Создана папка: $dir"
    }
}

# ─────────────────────────────────────────────
# Определение списка файлов для обработки
# ─────────────────────────────────────────────

$SupportedExtensions = @(".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".mkv", ".flac", ".aac", ".webm")

$FilesToProcess = @()

$ResolvedInput = Resolve-Path -Path $InputPath -ErrorAction SilentlyContinue
if (-not $ResolvedInput) {
    Exit-WithError "Путь не найден: $InputPath"
}
$ResolvedInput = $ResolvedInput.Path

if (Test-Path $ResolvedInput -PathType Container) {
    # Batch-режим: директория
    Write-Info "Batch-режим: обработка всех аудио/видео файлов в $ResolvedInput"
    $FilesToProcess = Get-ChildItem -Path $ResolvedInput -File |
        Where-Object { $SupportedExtensions -contains $_.Extension.ToLower() }

    if ($FilesToProcess.Count -eq 0) {
        Exit-WithError "В папке '$ResolvedInput' не найдено поддерживаемых файлов ($($SupportedExtensions -join ', '))."
    }
    Write-Info "Найдено файлов: $($FilesToProcess.Count)"
} else {
    # Одиночный файл
    if (-not (Test-Path $ResolvedInput -PathType Leaf)) {
        Exit-WithError "Файл не найден: $ResolvedInput"
    }
    $ext = [System.IO.Path]::GetExtension($ResolvedInput).ToLower()
    if ($SupportedExtensions -notcontains $ext) {
        Exit-WithError "Неподдерживаемый формат файла: $ext`nПоддерживаются: $($SupportedExtensions -join ', ')"
    }
    $FilesToProcess = @(Get-Item $ResolvedInput)
}

# ─────────────────────────────────────────────
# Сборка образа (если ещё не собран)
# ─────────────────────────────────────────────

Write-Info "Проверка Docker-образа meeting-transcriber..."
$ImageExists = docker images -q meeting-transcriber 2>&1
if (-not $ImageExists) {
    Write-Info "Образ не найден. Собираем (это займёт несколько минут при первом запуске)..."
    docker build -t meeting-transcriber "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Сборка Docker-образа завершилась с ошибкой."
    }
    Write-Ok "Образ собран."
} else {
    Write-Ok "Образ найден."
}

# ─────────────────────────────────────────────
# Обработка файлов
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Проверка GPU (один раз)
# ─────────────────────────────────────────────

$GpuAvailable = $false
Write-Info "Проверка GPU..."
try {
    $null = docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $GpuAvailable = $true
        Write-Ok "GPU найден, будет использован CUDA."
    } else {
        Write-Warn "GPU недоступен, будет использован CPU."
    }
} catch { Write-Warn "GPU недоступен, будет использован CPU." }

# ─────────────────────────────────────────────
# Обработка файлов
# ─────────────────────────────────────────────

$TotalFiles   = $FilesToProcess.Count
$CurrentFile  = 0
$SuccessCount = 0
$ErrorCount   = 0

Write-Log "RUN" "-" "Начало обработки. Файлов: $TotalFiles"

foreach ($File in $FilesToProcess) {
    $CurrentFile++
    $FileName  = $File.Name
    $Stem      = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $Category  = Get-Category $FileName
    $StartTime = Get-Date

    # Создать папку категории в output
    $CategoryOutputDir = Join-Path $OutputDir $Category
    if (-not (Test-Path $CategoryOutputDir)) {
        New-Item -ItemType Directory -Path $CategoryOutputDir | Out-Null
    }

    Write-Host ""
    Write-Info "[$CurrentFile/$TotalFiles] $FileName  →  $Category"
    Write-Log "START" $FileName "category=$Category"

    # Копируем файл в input/, если он уже не там
    $InputFilePath = Join-Path $InputDir $FileName
    if ($File.FullName -ne $InputFilePath) {
        Write-Info "Копирование '$FileName' в input/..."
        Copy-Item -Path $File.FullName -Destination $InputFilePath -Force
    }

    # Запуск контейнера
    $RunArgs = @(
        "run", "--rm",
        "--env-file", $EnvFile,
        "-e", "OUTPUT_SUBDIR=$Category",
        "-v", "${InputDir}:/input",
        "-v", "${OutputDir}:/output"
    )
    if ($GpuAvailable) { $RunArgs += "--gpus", "all" }
    $RunArgs += "meeting-transcriber"
    $RunArgs += "/input/$FileName"

    Write-Info "Запуск WhisperX..."
    & docker @RunArgs
    $DockerExit = $LASTEXITCODE

    if ($DockerExit -ne 0) {
        Write-Err "Ошибка Docker (exit $DockerExit): $FileName"
        Write-Log "ERROR" $FileName "Docker exit code: $DockerExit"
        Move-FileToDir $InputFilePath $FailedDir | Out-Null
        $ErrorCount++
        continue
    }

    # Постобработка
    Write-Info "Постобработка (transcript_speakers.txt)..."
    $PostprocessArgs = @(
        "run", "--rm",
        "-v", "${OutputDir}:/output",
        "--entrypoint", "python",
        "meeting-transcriber",
        "/scripts/postprocess.py",
        "--output-dir", "/output/$Category",
        "--filename", $Stem
    )
    & docker @PostprocessArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Постобработка завершилась с ошибкой. Основные файлы в $CategoryOutputDir."
    }

    # Summary через LLM (если настроен LLM_BASE_URL)
    $LlmEnabled = $EnvVars.ContainsKey("LLM_BASE_URL") -and $EnvVars["LLM_BASE_URL"] -ne ""
    if ($LlmEnabled) {
        Write-Info "Запуск LLM summary (summarize.py)..."
        $SummarizeArgs = @(
            "run", "--rm",
            "--env-file", $EnvFile,
            "-v", "${OutputDir}:/output",
            "--add-host", "host.docker.internal:host-gateway",
            "--entrypoint", "python",
            "meeting-transcriber",
            "/scripts/summarize.py",
            "--speakers-file", "/output/$Category/${Stem}_speakers.txt"
        )
        & docker @SummarizeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "LLM summary завершился с ошибкой (exit $($LASTEXITCODE)). Транскрипт сохранён."
            Write-Log "WARN" $FileName "summarize.py exit code: $LASTEXITCODE"
        } else {
            Write-Ok "Summary готов: ${Stem}_summary.md, ${Stem}_actions.json"
            Write-Log "SUMMARY" $FileName "summary+actions created"
        }
    } else {
        Write-Info "LLM_BASE_URL не задан — summary пропущен."
    }

    # Переместить исходный файл в processed/
    Move-FileToDir $InputFilePath $ProcessedDir | Out-Null

    $Duration = [int](((Get-Date) - $StartTime).TotalSeconds)
    Write-Log "SUCCESS" $FileName "category=$Category duration=${Duration}s output=$CategoryOutputDir"
    Write-Ok "[$CurrentFile/$TotalFiles] Готово: $FileName (${Duration}s)  →  $CategoryOutputDir"
    $SuccessCount++
}

Write-Host ""
Write-Log "DONE" "-" "Итого: успешно=$SuccessCount ошибок=$ErrorCount из $TotalFiles"
Write-Ok "Обработано: $SuccessCount из $TotalFiles. Лог: $LogFile"
if ($ErrorCount -gt 0) {
    Write-Warn "Файлы с ошибками перемещены в: $FailedDir"
}
