#!/usr/bin/env bash
# =============================================================================
# setup_ml_env.sh
# macOS / Linux setup script for a beginner PyTorch ML environment
# Usage: chmod +x setup_ml_env.sh && ./setup_ml_env.sh
# =============================================================================

set -e

# --- Config ------------------------------------------------------------------
ENV_NAME="ml_env"
PYTHON_VERSION="3.11"
VENV_PATH="$HOME/$ENV_NAME"

# --- Colors ------------------------------------------------------------------
info()    { printf "\033[0;36m[INFO]\033[0m  %s\n" "$1"; }
success() { printf "\033[0;32m[OK]\033[0m    %s\n" "$1"; }
warn()    { printf "\033[1;33m[WARN]\033[0m  %s\n" "$1"; }
error()   { printf "\033[0;31m[ERROR]\033[0m %s\n" "$1"; exit 1; }

# =============================================================================
# 1. Detect OS
# =============================================================================
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected OS: $OS ($ARCH)"

# =============================================================================
# 2. Install / verify Homebrew (macOS) or system Python (Linux)
# =============================================================================
if [[ "$OS" == "Darwin" ]]; then
    # --- macOS: use Homebrew + pyenv ---
    info "Checking for Homebrew..."
    if ! command -v brew &>/dev/null; then
        warn "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [[ "$ARCH" == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        success "Homebrew installed."
    else
        success "Homebrew found: $(brew --version | head -1)"
    fi

    info "Checking for pyenv..."
    if ! command -v pyenv &>/dev/null; then
        warn "pyenv not found. Installing via Homebrew..."
        brew install pyenv
        SHELL_RC="$HOME/.$(basename "$SHELL")rc"
        if ! grep -q 'pyenv init' "$SHELL_RC" 2>/dev/null; then
            cat >> "$SHELL_RC" <<'EOF'

# pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
            info "Added pyenv init to $SHELL_RC"
        fi
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        success "pyenv installed."
    else
        success "pyenv found: $(pyenv --version)"
    fi

    # Install the target Python version
    INSTALLED=$(pyenv versions --bare | grep "^${PYTHON_VERSION}" | tail -1)
    if [[ -z "$INSTALLED" ]]; then
        info "Installing Python $PYTHON_VERSION via pyenv (may take a few minutes)..."
        LATEST=$(pyenv install --list | tr -d ' ' | grep "^${PYTHON_VERSION}\." | grep -v '[a-zA-Z]' | tail -1)
        pyenv install "$LATEST"
        INSTALLED="$LATEST"
        success "Python $INSTALLED installed."
    else
        success "Python $INSTALLED already available."
    fi

    # Temporarily set pyenv local so `python3` resolves correctly
    pyenv local "$INSTALLED" 2>/dev/null || true
    PYTHON_CMD="$(pyenv root)/versions/$INSTALLED/bin/python3"

elif [[ "$OS" == "Linux" ]]; then
    # --- Linux: use system package manager ---
    info "Checking for Python 3..."
    if ! command -v python3 &>/dev/null; then
        warn "Python 3 not found. Attempting install via apt..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3 python3-venv python3-pip
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3 python3-pip
        elif command -v pacman &>/dev/null; then
            sudo pacman -Sy --noconfirm python python-pip
        else
            error "Could not detect a supported package manager. Install Python $PYTHON_VERSION manually."
        fi
        success "Python 3 installed."
    else
        success "Python 3 found: $(python3 --version)"
    fi

    # Make sure venv module is available
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        warn "python3-venv not found. Installing..."
        sudo apt-get install -y -qq python3-venv 2>/dev/null || true
    fi

    PYTHON_CMD="python3"
else
    error "Unsupported OS: $OS. Use setup_ml_env.ps1 for Windows."
fi

# =============================================================================
# 3. Create virtual environment
# =============================================================================
info "Creating virtual environment at $VENV_PATH..."
if [[ -d "$VENV_PATH" ]]; then
    warn "Virtual environment already exists at $VENV_PATH. Skipping creation."
else
    "$PYTHON_CMD" -m venv "$VENV_PATH"
    success "Virtual environment created."
fi

# Activate
info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
success "Virtual environment active: $VIRTUAL_ENV"

# =============================================================================
# 4. Install packages
# =============================================================================
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing core ML packages (this may take a few minutes)..."
pip install --quiet \
    torch torchvision torchaudio \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    jupyter \
    notebook \
    ipykernel \
    tqdm
success "Core packages installed."

# Register Jupyter kernel
info "Registering Jupyter kernel..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"
success "Jupyter kernel registered as 'Python ($ENV_NAME)'."

# =============================================================================
# 5. Sanity check
# =============================================================================
info "Running sanity check..."
python - <<'PYCHECK'
import torch, numpy as np, pandas as pd, matplotlib, sklearn

print(f"  torch       : {torch.__version__}")
print(f"  numpy       : {np.__version__}")
print(f"  pandas      : {pd.__version__}")
print(f"  matplotlib  : {matplotlib.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")

if torch.backends.mps.is_available():
    print("  GPU backend : Apple MPS ✓")
elif torch.cuda.is_available():
    print("  GPU backend : CUDA ✓")
else:
    print("  GPU backend : CPU only (normal for most setups)")
PYCHECK
success "Sanity check passed."

# =============================================================================
# 6. Shell alias
# =============================================================================
SHELL_RC="$HOME/.$(basename "$SHELL")rc"
ALIAS_LINE="alias activate_ml='source $VENV_PATH/bin/activate'"
if ! grep -q "activate_ml" "$SHELL_RC" 2>/dev/null; then
    printf "\n# Quick-activate ML environment\n%s\n" "$ALIAS_LINE" >> "$SHELL_RC"
    info "Added alias 'activate_ml' to $SHELL_RC"
fi

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "------------------------------------------------------"
echo " Setup complete! Here's how to get started:"
echo "------------------------------------------------------"
echo "" 
echo "  Activate your environment:  source ~/$ENV_NAME/bin/activate"
echo "  Or use the shortcut:        activate_ml  (after restarting your shell)"
echo ""
echo "  Start Jupyter Notebook:     jupyter notebook"
echo "  Start JupyterLab:           jupyter lab"
echo ""
echo "  Deactivate when done:       deactivate"
echo ""