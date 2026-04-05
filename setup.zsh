#!/usr/bin/env zsh
# =============================================================================
# setup_ml_env.zsh
# macOS setup script for a beginner PyTorch machine learning environment
# Usage: chmod +x setup_ml_env.zsh && ./setup_ml_env.zsh
# =============================================================================

set -e  # Exit immediately if any command fails

# --- Config ------------------------------------------------------------------
ENV_NAME="ml_env"
PYTHON_VERSION="3.11"  # Stable, well-supported version for PyTorch

# --- Colors for output -------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { print -P "%F{cyan}[INFO]%f  $1" }
success() { print -P "%F{green}[OK]%f    $1" }
warn()    { print -P "%F{yellow}[WARN]%f  $1" }
error()   { print -P "%F{red}[ERROR]%f $1"; exit 1 }

# =============================================================================
# 1. Check for Homebrew
# =============================================================================
info "Checking for Homebrew..."
if ! command -v brew &>/dev/null; then
  warn "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add Homebrew to PATH for Apple Silicon Macs
  if [[ $(uname -m) == "arm64" ]]; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
  success "Homebrew installed."
else
  success "Homebrew found: $(brew --version | head -1)"
fi

# =============================================================================
# 2. Install pyenv (Python version manager)
# =============================================================================
info "Checking for pyenv..."
if ! command -v pyenv &>/dev/null; then
  warn "pyenv not found. Installing via Homebrew..."
  brew install pyenv
  # Add pyenv init to ~/.zshrc if not already present
  if ! grep -q 'pyenv init' ~/.zshrc; then
    cat >> ~/.zshrc <<'EOF'

# pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
    info "Added pyenv init to ~/.zshrc"
  fi
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  success "pyenv installed."
else
  success "pyenv found: $(pyenv --version)"
fi

# =============================================================================
# 3. Install the target Python version via pyenv
# =============================================================================
info "Checking for Python $PYTHON_VERSION via pyenv..."
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
  info "Installing Python $PYTHON_VERSION (this may take a few minutes)..."
  pyenv install $PYTHON_VERSION
  success "Python $PYTHON_VERSION installed."
else
  success "Python $PYTHON_VERSION already available."
fi

# =============================================================================
# 4. Create a virtual environment
# =============================================================================
VENV_PATH="$HOME/$ENV_NAME"

info "Creating virtual environment at $VENV_PATH..."
if [[ -d "$VENV_PATH" ]]; then
  warn "Virtual environment already exists at $VENV_PATH. Skipping creation."
else
  PYTHON_BIN="$(pyenv root)/versions/$PYTHON_VERSION.$(pyenv versions | grep $PYTHON_VERSION | awk -F'.' '{print $NF}' | tr -d ' *\n' | head -c2)/bin/python3"
  # Simpler: use pyenv's shim after setting local version
  mkdir -p "$VENV_PATH"
  pyenv local $PYTHON_VERSION 2>/dev/null || true
  python3 -m venv "$VENV_PATH"
  success "Virtual environment created."
fi

# Activate the venv
info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
success "Virtual environment active: $VIRTUAL_ENV"

# =============================================================================
# 5. Upgrade pip & install core ML packages
# =============================================================================
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing core ML packages (this may take a few minutes)..."

# Detect architecture for PyTorch install hint
ARCH=$(uname -m)

# Core packages
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

# Register the venv as a Jupyter kernel
info "Registering Jupyter kernel..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"
success "Jupyter kernel registered as 'Python ($ENV_NAME)'."

# =============================================================================
# 6. Quick sanity check
# =============================================================================
info "Running sanity check..."
python - <<'PYCHECK'
import torch
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print(f"  torch      : {torch.__version__}")
print(f"  numpy      : {np.__version__}")
print(f"  pandas     : {pd.__version__}")
print(f"  matplotlib : {matplotlib.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")

# Check MPS (Apple Silicon GPU) availability
if torch.backends.mps.is_available():
    print("  Apple MPS (GPU) : AVAILABLE ✓")
elif torch.cuda.is_available():
    print("  CUDA (GPU) : AVAILABLE ✓")
else:
    print("  Running on CPU (normal for most Macs)")
PYCHECK

success "Sanity check passed."

# =============================================================================
# 7. Add a convenient activation alias to ~/.zshrc
# =============================================================================
ALIAS_LINE="alias activate_ml='source $VENV_PATH/bin/activate'"
if ! grep -q "activate_ml" ~/.zshrc; then
  echo "\n# Quick-activate ML environment\n$ALIAS_LINE" >> ~/.zshrc
  info "Added alias 'activate_ml' to ~/.zshrc"
fi

# =============================================================================
# Done!
# =============================================================================
print ""
print -P "%F{green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%f"
print -P "%F{green} Setup complete! Here's how to get started:%f"
print -P "%F{green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%f"
print ""
print -P "  Activate your environment:  %F{cyan}source ~/ml_env/bin/activate%f"
print -P "  Or use the shortcut:        %F{cyan}activate_ml%f  (after restarting your shell)"
print ""
print -P "  Start Jupyter Notebook:     %F{cyan}jupyter notebook%f"
print -P "  Start JupyterLab:           %F{cyan}jupyter lab%f"
print ""
print -P "  Deactivate when done:       %F{cyan}deactivate%f"
print ""
print -P "  To use in a Python file:"
print -P "    %F{yellow}import torch%f"
print -P "    %F{yellow}x = torch.rand(3, 3)%f"
print -P "    %F{yellow}print(x)%f"
print ""