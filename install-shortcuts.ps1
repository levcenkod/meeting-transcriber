# install-shortcuts.ps1 - creates Desktop shortcuts for start/stop.
# Run via install-shortcuts.bat (handles execution policy).

$ErrorActionPreference = 'Stop'

$base    = $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$ws      = New-Object -ComObject WScript.Shell

$start = $ws.CreateShortcut((Join-Path $desktop 'Meeting Transcriber - Start.lnk'))
$start.TargetPath       = (Join-Path $base 'run.bat')
$start.WorkingDirectory = $base
$start.IconLocation     = 'shell32.dll,137'
$start.Description       = 'Start Meeting Transcriber (with auto-update)'
$start.Save()

$stop = $ws.CreateShortcut((Join-Path $desktop 'Meeting Transcriber - Stop.lnk'))
$stop.TargetPath       = (Join-Path $base 'stop.bat')
$stop.WorkingDirectory = $base
$stop.IconLocation     = 'shell32.dll,131'
$stop.Description       = 'Stop Meeting Transcriber'
$stop.Save()

Write-Host 'Done. Two shortcuts were created on the Desktop.'
