# Downloads the newest LiKOVA IPA built by CI into Documents\LiKOVA\IPA,
# named by its build number, ready for Sideloadly.
#
#   powershell -ExecutionPolicy Bypass -File tools\get_ipa.ps1          (newest)
#   powershell -ExecutionPolicy Bypass -File tools\get_ipa.ps1 <run id> (a specific run)
#
# Needs the GitHub CLI (gh) signed in. CI keeps an IPA for 30 days.
param([string]$RunId = "")

$ErrorActionPreference = "Stop"
$repo = "bottlemaster-22/Nimbus3D"
$dest = Join-Path $env:USERPROFILE "Documents\LiKOVA\IPA"
New-Item -ItemType Directory -Force $dest | Out-Null

if ($RunId -eq "") {
    # Push runs on the branch skip the archive job; PR and manual runs build the IPA.
    $runs = gh run list --repo $repo --branch build/first-ci --status success --limit 20 `
        --json databaseId,number,event | ConvertFrom-Json
    $run = $runs | Where-Object { $_.event -ne "push" } | Select-Object -First 1
    if ($null -eq $run) { throw "No successful IPA build found on build/first-ci." }
    $RunId = "$($run.databaseId)"
    $build = $run.number
} else {
    $build = (gh run view $RunId --repo $repo --json number | ConvertFrom-Json).number
}

$tmp = Join-Path $env:TEMP "likova-ipa-$RunId"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
gh run download $RunId --repo $repo --pattern "*-unsigned-ipa" --dir $tmp
$ipa = Get-ChildItem -Path $tmp -Recurse -Filter *.ipa | Select-Object -First 1
if ($null -eq $ipa) { throw "Run $RunId has no IPA artifact (expired after 30 days, or the build failed)." }

$target = Join-Path $dest "LiKOVA-build$build.ipa"
Move-Item -Force $ipa.FullName $target
Remove-Item -Recurse -Force $tmp
Write-Output "Ready for Sideloadly: $target"
