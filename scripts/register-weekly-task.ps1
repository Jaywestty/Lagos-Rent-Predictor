<#
.SYNOPSIS
  Registers (or updates) the "Lagos Rent weekly scrape" task in Windows Task Scheduler.

.DESCRIPTION
  Runs scripts\weekly-scrape.ps1 every Monday at 09:00 for the current user.
  If the laptop was off or asleep at that time, Windows runs it as soon as
  possible afterwards. Because each run looks back 7 days, a late run loses nothing.
  Waits for a network connection, works on battery, and stops after 6 hours.

  Remove it with:  Unregister-ScheduledTask -TaskName "Lagos Rent weekly scrape"
#>
param(
    [string]$Day = "Monday",
    [string]$At = "09:00"
)

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "weekly-scrape.ps1"
$repo = Split-Path -Parent $PSScriptRoot

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName "Lagos Rent weekly scrape" -Description "Scrape last week's Lagos rent listings and load them into the database." `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName "Lagos Rent weekly scrape"
"Registered 'Lagos Rent weekly scrape'. Next run: $($info.NextRunTime)"
