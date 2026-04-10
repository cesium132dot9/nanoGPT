# =============================================================================
# setup_ml_env.ps1
# Windows setup script for a beginner PyTorch ML environment
# Usage: Right-click → "Run with PowerShell"  OR  powershell -ExecutionPolicy Bypass -File setup_ml_env.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

# --- Config ------------------------------------------------------------------
$EnvName       = "ml_env"
$PythonVersion = "3.11"
$VenvPath      = "$env:USERPROFILE\$EnvName"

# --- Helpers -----------------------------------------------------------------
function Write-Info    { param($Msg) Write-Host "[INFO]  $Msg" -ForegroundColor Cyan }
function Write-Ok      { param($Msg) Write-Host "[OK]    $Msg" -ForegroundColor Green }
function Write-Warn    { param($Msg) Write-Host "[WARN]  $Msg" -ForegroundColor Yellow }
function Write-Err     { param($Msg) Write-Host "[ERROR] $Msg" -ForegroundColor Red; exit 1 }

# =============================================================================
# 1. Check for Python
# =============================================================================
Write-Info "Checking for Python..."

$PythonCmd = $null

# Look for an existing Python 3.11+ installation
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python\s+3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 11) {
                $PythonCmd = $cmd
                Write-Ok "Found: $ver"
                break
            }
        }
    } catch { }
}

# If not found, try the Windows Python launcher
if (-not $PythonCmd) {
    try {
        $ver = & py "-$PythonVersion" --version 2>&1
        if ($ver -match "Python") {
            $PythonCmd = "py -$PythonVersion"
            Write-Ok "Found via py launcher: $ver"
        }
    } catch { }
}

# Still not found — offer to install via winget
if (-not $PythonCmd) {
    Write-Warn "Python $PythonVersion not found."

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Installing Python $PythonVersion via winget..."
        winget install --id Python.Python.$PythonVersion --accept-package-agreements --accept-source-agreements
        # Refresh PATH so the new install is visible
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                     [System.Environment]::GetEnvironmentVariable("Path", "User")
        $PythonCmd = "python"
        Write-Ok "Python installed. You may need to restart your terminal if issues arise."
    } else {
        Write-Err "Python $PythonVersion is required. Install it from https://www.python.org/downloads/ and re-run this script."
    }
}

# =============================================================================
# 2. Create virtual environment
# =============================================================================
Write-Info "Creating virtual environment at $VenvPath..."

if (Test-Path $VenvPath) {
    Write-Warn "Virtual environment already exists at $VenvPath. Skipping creation."
} else {
    if ($PythonCmd -eq "py -$PythonVersion") {
        & py "-$PythonVersion" -m venv $VenvPath
    } else {
        & $PythonCmd -m venv $VenvPath
    }
    Write-Ok "Virtual environment created."
}

# Activate
Write-Info "Activating virtual environment..."
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Err "Activation script not found at $ActivateScript"
}
& $ActivateScript
Write-Ok "Virtual environment active."

# =============================================================================
# 3. Install packages
# =============================================================================
Write-Info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

Write-Info "Installing core ML packages (this may take a few minutes)..."
pip install --quiet `
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 `
    numpy `
    pandas `
    matplotlib `
    scikit-learn `
    jupyter `
    notebook `
    ipykernel `
    tqdm

Write-Ok "Core packages installed."

# Register Jupyter kernel
Write-Info "Registering Jupyter kernel..."
python -m ipykernel install --user --name="$EnvName" --display-name "Python ($EnvName)"
Write-Ok "Jupyter kernel registered as 'Python ($EnvName)'."

# =============================================================================
# 4. Sanity check
# =============================================================================
Write-Info "Running sanity check..."

python -c @"
import torch, numpy as np, pandas as pd, matplotlib, sklearn

print(f'  torch       : {torch.__version__}')
print(f'  numpy       : {np.__version__}')
print(f'  pandas      : {pd.__version__}')
print(f'  matplotlib  : {matplotlib.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')

if torch.cuda.is_available():
    print('  GPU backend : CUDA available')
else:
    print('  GPU backend : CPU only (install CUDA toolkit for GPU support)')
"@

Write-Ok "Sanity check passed."

# =============================================================================
# 5. Add activate_ml function to PowerShell profile (works from any directory)
# =============================================================================
Write-Info "Setting up 'activate_ml' command..."

# Ensure the profile directory exists
$ProfileDir = Split-Path $PROFILE -Parent
if (-not (Test-Path $ProfileDir)) {
    New-Item -ItemType Directory -Path $ProfileDir -Force | Out-Null
}

# Ensure the profile file exists
if (-not (Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

# Add the function if not already present
if (-not (Select-String -Path $PROFILE -Pattern "function activate_ml" -Quiet)) {
    Add-Content -Path $PROFILE -Value @"

# Quick-activate ML environment
function activate_ml { & '$VenvPath\Scripts\Activate.ps1' }
"@
    Write-Ok "Added 'activate_ml' function to PowerShell profile ($PROFILE)"
    Write-Info "This will be available in every new PowerShell window."
} else {
    Write-Ok "'activate_ml' function already in PowerShell profile."
}

# Also create a .bat shortcut for CMD users
$BatPath = "$env:USERPROFILE\activate_ml.bat"
if (-not (Test-Path $BatPath)) {
    Set-Content -Path $BatPath -Value "@echo off`r`ncall `"$VenvPath\Scripts\activate.bat`""
    Write-Info "Created CMD shortcut: $BatPath"
}

# =============================================================================
# Done!
# =============================================================================
Write-Host ""
Write-Host "------------------------------------------------------" -ForegroundColor Green
Write-Host " Setup complete! Here's how to get started:" -ForegroundColor Green
Write-Host "------------------------------------------------------" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate (PowerShell):  activate_ml" -ForegroundColor Cyan
Write-Host "  Activate (CMD):         activate_ml.bat  (from %USERPROFILE%)" -ForegroundColor Cyan
Write-Host "  Or directly:            & '$VenvPath\Scripts\Activate.ps1'" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start Jupyter Notebook: jupyter notebook" -ForegroundColor Cyan
Write-Host "  Start JupyterLab:       jupyter lab" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Deactivate when done:   deactivate" -ForegroundColor Cyan
Write-Host ""