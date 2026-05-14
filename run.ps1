# PowerShell quick-start script for Windows.
#
# Usage:
#   .\run.ps1 setup    # create venv + install deps
#   .\run.ps1 dev      # run the API with reload
#   .\run.ps1 start    # run the API (no reload)
#   .\run.ps1 test     # run pytest
#   .\run.ps1 index    # POST /api/index to build the vector index
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "dev", "start", "test", "index")]
    [string]$Cmd = "start"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Venv = Join-Path $Root ".venv"
$Py   = Join-Path $Venv "Scripts\python.exe"

function Ensure-Venv {
    if (-not (Test-Path $Py)) {
        Write-Host ">> Creating virtualenv in $Venv"
        python -m venv $Venv
    }
}

switch ($Cmd) {
    "setup" {
        Ensure-Venv
        Write-Host ">> Installing dependencies"
        & $Py -m pip install --upgrade pip
        & $Py -m pip install -r requirements.txt
        if (-not (Test-Path ".env")) {
            Copy-Item .env.example .env
            Write-Host ">> Wrote .env from .env.example — review it before starting."
        }
    }
    "dev" {
        Ensure-Venv
        & $Py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "start" {
        Ensure-Venv
        & $Py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    }
    "test" {
        Ensure-Venv
        & $Py -m pytest -q
    }
    "index" {
        Invoke-RestMethod -Method Post http://localhost:8000/api/index
    }
}
