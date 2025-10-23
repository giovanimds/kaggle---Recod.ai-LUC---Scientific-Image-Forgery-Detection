# Setup Guide

This guide will help you set up the development environment for the Scientific Image Forgery Detection project.

## Quick Start with GitHub Codespaces (Recommended)

The easiest way to get started is using GitHub Codespaces:

1. Click the "Code" button on the GitHub repository
2. Select "Codespaces" tab
3. Click "Create codespace on main"

The devcontainer will automatically:
- Configure a 32-core machine with 64GB RAM (largest available)
- Install Python 3.11
- Install uv package manager
- Install all project dependencies
- Set up VS Code extensions for Python, Jupyter, and linting

Once the codespace is ready, you can immediately start working on the project!

## Local Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended for training)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/giovanimds/kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection.git
   cd kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install project dependencies:**
   ```bash
   uv pip install -e .
   ```

4. **Install development dependencies (optional):**
   ```bash
   uv pip install -e ".[dev]"
   ```

### Alternative: Using pip/venv

If you prefer traditional pip and virtual environments:

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Download Competition Data

1. **Install Kaggle CLI:**
   ```bash
   uv pip install kaggle
   # or: pip install kaggle
   ```

2. **Configure Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New Token" to download `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json`
   - On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download competition data:**
   ```bash
   kaggle competitions download -c recodai-luc-scientific-image-forgery-detection
   unzip recodai-luc-scientific-image-forgery-detection.zip -d data/raw/
   ```

## Verify Installation

Run the following commands to verify your setup:

```bash
# Check Python version
python --version  # Should be 3.11+

# Check uv installation
uv --version

# Verify project structure
make help

# Test imports
python -c "import sys; sys.path.insert(0, 'src'); import forgery_detection; print('✓ Package installed correctly')"
```

## IDE Setup

### VS Code (Recommended)

The project includes VS Code configuration for:
- Python IntelliSense
- Automatic code formatting with Black
- Linting with Ruff
- Jupyter notebook support

Install recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Jupyter (ms-toolsai.jupyter)
- Ruff (charliermarsh.ruff)

### PyCharm

1. Open the project folder
2. PyCharm should automatically detect the pyproject.toml
3. Configure the Python interpreter to use your virtual environment

## Next Steps

After setup:

1. **Explore the data:**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Run tests:**
   ```bash
   make test
   ```

3. **Train a model:**
   ```bash
   python scripts/train.py
   ```

4. **Monitor training with TensorBoard:**
   ```bash
   make tensorboard
   ```

## Common Issues

### Issue: `uv` command not found

**Solution:** Make sure you've sourced the environment:
```bash
source $HOME/.cargo/env
```

### Issue: CUDA not available

**Solution:** 
- Check if you have a CUDA-capable GPU
- Install appropriate CUDA drivers
- Install PyTorch with CUDA support: 
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Issue: Module not found errors

**Solution:** Make sure you've installed the package:
```bash
uv pip install -e .
```

## Getting Help

- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for bugs or questions
- Refer to the [Competition Page](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)

Happy coding! 🚀
