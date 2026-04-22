$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$Config = Join-Path $PSScriptRoot "submit_config.json"

Write-Host "[run_submit.ps1] Root: $Root"
Write-Host "[run_submit.ps1] Config: $Config"

python (Join-Path $PSScriptRoot "make_submission.py") --config $Config
python (Join-Path $PSScriptRoot "preflight_check.py") --config $Config

Write-Host "[run_submit.ps1] Done."
