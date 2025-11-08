# Environment Setup Guide - Patient Advocacy Agent

This guide explains how to set up and manage the Python environment using `uv` (the ultra-fast Python package installer).

---

## Prerequisites

Before setting up, ensure you have:

1. **uv installed**: [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
   ```bash
   # macOS
   brew install uv

   # Linux/WSL
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Python 3.12 installed** (for this project)
   ```bash
   python3.12 --version  # Should output Python 3.12.x
   ```

---

## Quick Setup (3 Steps)

### Step 1: Navigate to Project Directory
```bash
cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent
```

### Step 2: Create Virtual Environment with uv
```bash
uv sync --python 3.12
```

This command:
- ✓ Creates a virtual environment
- ✓ Installs all dependencies from `pyproject.toml`
- ✓ Creates `.venv` directory with isolated Python 3.12

### Step 3: Activate the Environment

**Option A: Using uv run (Recommended)**
```bash
# Run Python scripts directly with uv
uv run python --version

# Run the example
uv run python example_usage.py

# Or run any Python command
uv run pip list
```

**Option B: Traditional Activation (Manual)**

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```bash
.venv\Scripts\activate
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

---

## Understanding uv Environment

### Where uv Stores Files

```
project_root/
├── .venv/                    # Virtual environment
│   ├── bin/                  # Executables (python, pip, etc.)
│   ├── lib/                  # Python packages
│   ├── pyvenv.cfg           # Environment config
│   └── ...
│
├── uv.lock                   # Lock file (auto-generated)
└── pyproject.toml            # Project config
```

### uv.lock File

The `uv.lock` file is automatically created and contains:
- ✓ Exact versions of all dependencies
- ✓ Deterministic reproducible builds
- ✓ Should be committed to git

```bash
# DO commit this
git add uv.lock
git commit -m "Add dependency lock file"
```

---

## Common Commands

### Syncing Environment

```bash
# Sync with latest dependencies
uv sync

# Sync with specific Python version
uv sync --python 3.12

# Sync and include dev dependencies
uv sync --all-groups
```

### Running Python

```bash
# Run Python directly (auto-uses virtual env)
uv run python script.py

# Run Python interactively
uv run python

# Run with arguments
uv run python -m pip list

# Run the example
uv run python example_usage.py
```

### Adding Dependencies

```bash
# Add a new dependency
uv add torch

# Add development dependency
uv add --group dev pytest

# Add optional dependency
uv add --optional gpu faiss-gpu
```

### Removing Dependencies

```bash
# Remove a dependency
uv remove torch

# Remove from specific group
uv remove --group dev pytest
```

### Checking Environment

```bash
# Show installed packages
uv run pip list

# Show specific package info
uv run pip show torch

# Verify environment
uv run python -c "import torch; print(torch.__version__)"
```

---

## Troubleshooting

### Issue 1: "python3.12 not found"

**Problem**: uv can't find Python 3.12

**Solution**:
```bash
# Check available Python versions
uv python list

# Install Python 3.12 via uv
uv python install 3.12

# Or use system Python if available
which python3.12
```

### Issue 2: ".venv not created" after `uv sync`

**Problem**: Virtual environment folder not visible

**Solution**:
```bash
# uv now stores venv in .venv by default, might be hidden
ls -la .venv/

# If still not visible, create explicitly
uv venv .venv --python 3.12

# Then sync
uv sync
```

### Issue 3: "ModuleNotFoundError" when running scripts

**Problem**: Dependencies not installed in virtual environment

**Solution**:
```bash
# Use uv run (recommended - auto-activates)
uv run python example_usage.py

# Or manually activate
source .venv/bin/activate  # macOS/Linux
python example_usage.py
```

### Issue 4: CUDA/GPU not recognized

**Problem**: PyTorch CPU version installed instead of CUDA

**Solution**:
```bash
# Check current PyTorch
uv run python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 5: "Permission denied" on Linux/WSL

**Problem**: Cannot activate virtual environment

**Solution**:
```bash
# Make activation script executable
chmod +x .venv/bin/activate

# Try activation again
source .venv/bin/activate
```

---

## Development Workflow

### For Development Work

```bash
# 1. Create environment
uv sync

# 2. Add dev dependencies
uv add --group dev pytest pytest-cov black mypy

# 3. Run tests
uv run pytest tests/

# 4. Format code
uv run black src/

# 5. Type check
uv run mypy src/
```

### For Production Deployment

```bash
# 1. Sync without dev dependencies
uv sync --no-dev

# 2. Run the application
uv run python -m patient_advocacy_agent

# 3. Or use the executable directly
source .venv/bin/activate
python -m patient_advocacy_agent
```

---

## Advanced Configuration

### pyproject.toml Organization

The `pyproject.toml` file defines:

```toml
[project]
name = "patient-advocacy-agent"
version = "0.1.0"
description = "patient_advocacy_agent"
requires-python = ">=3.12"

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    # ... core dependencies
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "mypy"]
gpu = ["faiss-gpu"]
```

### Using Optional Dependencies

```bash
# Install with GPU support
uv sync --extra gpu

# Install with dev tools
uv sync --all-groups

# Run with specific extras
uv run --extra gpu python example_usage.py
```

---

## Git Integration

### Files to Commit

```bash
# Always commit these
git add pyproject.toml
git add uv.lock
git add .python-version

# Don't commit these
git add .gitignore  # Should already exclude .venv/
```

### .gitignore for uv

Make sure `.gitignore` contains:
```
# uv
.venv/
__pycache__/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.pyc

# OS
.DS_Store
.env
```

---

## Environment Variables

### Setting Up .env File

```bash
# Create .env from template
cat > .env << EOF
# Project paths
export PROJECT_ROOT_DIR="/Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent"
export PYTHONPATH="/Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent/src"

# API keys
export OPENAI_API_KEY=your-key-here

# Data paths
export SCIN_DATA_DIR=./data/scin
export MODEL_DIR=./models
export REPORT_DIR=./reports
EOF
```

### Loading .env with uv

```bash
# Option 1: Load before uv run
source .env
uv run python example_usage.py

# Option 2: Using python-dotenv (already in dependencies)
uv run python -c "from dotenv import load_dotenv; load_dotenv(); ..."
```

---

## Switching Between Python Versions

### Using Different Python Versions

```bash
# Check available versions
uv python list

# Install specific version
uv python install 3.13

# Use specific version for this project
uv sync --python 3.13

# Revert to 3.12
uv sync --python 3.12
```

### Managing Multiple Projects

```bash
# Project A uses Python 3.12
cd ~/projects/project_a
uv sync --python 3.12

# Project B uses Python 3.13
cd ~/projects/project_b
uv sync --python 3.13

# No conflicts - each has isolated environment
```

---

## IDE Integration

### VS Code Setup

1. **Select Python Interpreter**:
   - Open Command Palette (Cmd+Shift+P)
   - Search: "Python: Select Interpreter"
   - Choose: `./.venv/bin/python`

2. **Configure VS Code**:
   ```json
   // .vscode/settings.json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
     "python.formatting.provider": "black",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true
   }
   ```

### PyCharm Setup

1. **Open Project Settings** → **Project: patient_advocacy_agent** → **Python Interpreter**
2. **Add Interpreter** → **Add Local Interpreter**
3. **Existing Environment** → Select `.venv/bin/python`
4. Click **OK**

### Jupyter Notebook

```bash
# Install Jupyter
uv add jupyter ipykernel

# Install kernel for this environment
uv run python -m ipykernel install --user --name patient_advocacy --display-name "Patient Advocacy"

# Start Jupyter
uv run jupyter notebook
```

---

## Performance Tips

### Faster Package Installation

```bash
# uv is already fast! But you can optimize further

# 1. Use cache
uv sync --cache-dir /custom/cache/path

# 2. Parallel downloads (default)
uv sync

# 3. Skip venv creation if only updating
uv pip compile pyproject.toml
```

### Checking Disk Usage

```bash
# Check .venv size
du -sh .venv/

# Check if too large
ls -lah .venv/lib/python3.12/site-packages/ | sort -k5 -h | tail -20
```

---

## Uninstalling/Clean Setup

### Complete Fresh Start

```bash
# Remove everything
rm -rf .venv/
rm -f uv.lock

# Recreate from scratch
uv sync --python 3.12
```

### Keep .venv but Update Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv sync --upgrade-package torch
```

---

## Useful Aliases

Add to `.bashrc` or `.zshrc` for convenience:

```bash
# Activate uv environment
alias activate-paa="source /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent/.venv/bin/activate"

# Quick run
alias paa-run="cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent && uv run python"

# Go to project
alias goto-paa="cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent"

# Run tests
alias paa-test="cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent && uv run pytest tests/"

# Run example
alias paa-example="cd /Users/sankar/sankar/courses/agentic-ai/patient_advocacy_agent && uv run python example_usage.py"
```

Then use:
```bash
activate-paa          # Activate environment
paa-run script.py     # Run script
paa-test             # Run tests
paa-example          # Run example
```

---

## Documentation Links

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv FAQ](https://docs.astral.sh/uv/guides/projects/)
- [Python 3.12 Features](https://docs.python.org/3/whatsnew/3.12.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

---

## Summary

### Quick Reference Card

| Task | Command |
|------|---------|
| Setup environment | `uv sync --python 3.12` |
| Run script | `uv run python script.py` |
| Install package | `uv add package_name` |
| Run tests | `uv run pytest tests/` |
| Run example | `uv run python example_usage.py` |
| Activate shell | `source .venv/bin/activate` |
| Check Python | `uv run python --version` |
| List packages | `uv run pip list` |
| Update all | `uv sync --upgrade` |
| Fresh start | `rm -rf .venv/ && uv sync` |

### Next Steps

1. ✓ Run: `uv sync --python 3.12`
2. ✓ Test: `uv run python --version`
3. ✓ Run: `uv run python example_usage.py`
4. ✓ Verify: `uv run python -c "import patient_advocacy_agent; print(patient_advocacy_agent.__version__)"`

---

**Version**: 1.0
**Last Updated**: 2024
**Status**: Production Ready
