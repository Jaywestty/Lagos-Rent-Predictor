<#
.SYNOPSIS
  Weekly scrape-and-load, run by Windows Task Scheduler on this machine.

.DESCRIPTION
  Runs `entrypoints.cli weekly-scrape`, which applies migrations, scrapes rent
  listings from the last N days, loads them into the database named by
  DATABASE_URL in .env, and recomputes weekly price snapshots.
  Output goes to logs\weekly-scrape-<timestamp>.log; logs older than 60 days are removed.

  Register the schedule once with scripts\register-weekly-task.ps1.
#>
param(
    [int]$SinceDays = 7,
    [int]$MaxPages = 30,
    [string]$Site = "all",
    [string]$TriggeredBy = "laptop-schedule"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("weekly-scrape-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }

$env:PYTHONIOENCODING = "utf-8"
"[$(Get-Date -Format s)] Starting weekly scrape (since $SinceDays days, max $MaxPages pages, site $Site)" | Out-File $log -Encoding utf8

# Run through cmd so the app's stderr logging goes straight to the file. Windows PowerShell 5.1
# would otherwise turn every stderr line into an error record. The exit code decides success.
$cmdLine = "`"$uv`" run python -m entrypoints.cli weekly-scrape --since-days $SinceDays --max-pages $MaxPages --site $Site --triggered-by $TriggeredBy >> `"$log`" 2>&1"
cmd.exe /d /c $cmdLine
$code = $LASTEXITCODE

"[$(Get-Date -Format s)] Finished with exit code $code" | Out-File $log -Append -Encoding utf8
Get-ChildItem $logDir -Filter "weekly-scrape-*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-60) } |
    Remove-Item -Force -ErrorAction SilentlyContinue
exit $code
