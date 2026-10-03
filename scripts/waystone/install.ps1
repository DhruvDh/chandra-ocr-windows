param(
    [Parameter(Mandatory=$true)][string]$Uv,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA 'chandra-ocr-windows\runtime')
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$SourceRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
New-Item $RuntimeRoot -ItemType Directory -Force | Out-Null
Copy-Item (Join-Path $SourceRoot 'runtime\waystone\pyproject.toml'), (Join-Path $SourceRoot 'runtime\waystone\uv.lock') $RuntimeRoot
$PythonRoot = Join-Path $RuntimeRoot 'python'
if (-not (Test-Path (Join-Path $PythonRoot 'python.exe'))) {
    Copy-Item (Split-Path $Python -Parent) $PythonRoot -Recurse
}
$Python = Join-Path $PythonRoot 'python.exe'
$env:UV_CACHE_DIR = Join-Path $RuntimeRoot 'uv-cache'
Push-Location $RuntimeRoot
try {
    & $Uv sync --frozen --link-mode copy --python $Python
    if ($LASTEXITCODE) { throw "uv sync failed: $LASTEXITCODE" }
} finally { Pop-Location }
