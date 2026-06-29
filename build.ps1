[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Create .venv or put Python on PATH."
    }
    $Python = $PythonCommand.Source
}

Push-Location $ProjectRoot
try {
    & $Python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is not installed. Run: $Python -m pip install -r requirements-build.txt"
    }

    if (-not $SkipTests) {
        & $Python -m unittest discover -s tests
        if ($LASTEXITCODE -ne 0) {
            throw "Tests failed; the package was not built."
        }
    }

    & $Python -m PyInstaller --noconfirm --clean StarAI.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }

    $Executable = Join-Path $ProjectRoot "dist\StarAI\StarAI.exe"
    $SmokeData = Join-Path $ProjectRoot "build\smoke-data"
    $PreviousVideoDriver = $env:SDL_VIDEODRIVER
    $PreviousAudioDriver = $env:SDL_AUDIODRIVER
    $PreviousDataDirectory = $env:STARAI_DATA_DIR
    try {
        $env:SDL_VIDEODRIVER = "dummy"
        $env:SDL_AUDIODRIVER = "dummy"
        $env:STARAI_DATA_DIR = $SmokeData
        $SmokeProcess = Start-Process `
            -FilePath $Executable `
            -ArgumentList "--smoke-test" `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($SmokeProcess.ExitCode -ne 0) {
            $SmokeLog = Join-Path $SmokeData "smoke-test-error.log"
            if (Test-Path $SmokeLog) {
                Get-Content $SmokeLog | Write-Error
            }
            throw "Packaged application smoke test failed."
        }
    } finally {
        $env:SDL_VIDEODRIVER = $PreviousVideoDriver
        $env:SDL_AUDIODRIVER = $PreviousAudioDriver
        $env:STARAI_DATA_DIR = $PreviousDataDirectory
    }

    $DistributionDirectory = Join-Path $ProjectRoot "dist\StarAI"
    $Archive = Join-Path $ProjectRoot "dist\StarAI-windows-x64.zip"
    Compress-Archive `
        -Path $DistributionDirectory `
        -DestinationPath $Archive `
        -CompressionLevel Optimal `
        -Force

    Write-Host "Build complete: $ProjectRoot\dist\StarAI\StarAI.exe"
    Write-Host "Shareable archive: $Archive"
} finally {
    Pop-Location
}
