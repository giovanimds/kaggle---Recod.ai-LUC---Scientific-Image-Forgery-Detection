# Contributing to Scientific Image Forgery Detection

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/giovanimds/kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection.git
cd kaggle---Recod.ai-LUC---Scientific-Image-Forgery-Detection
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
make install-dev
# or
uv pip install -e ".[dev]"
```

## Development Workflow

### Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Format your code before committing:
```bash
make format
# or
black src/ tests/
ruff check --fix src/ tests/
```

### Testing

Run tests with:
```bash
make test
# or
pytest tests/ -v --cov=src
```

Write tests for new features in the `tests/` directory.

### Project Structure

```
├── src/forgery_detection/  # Main package
│   ├── data.py            # Data loading utilities
│   ├── model.py           # Model definitions
│   ├── train.py           # Training utilities
│   └── config.py          # Configuration
├── scripts/               # Training/inference scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── data/                  # Dataset (not committed)
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test them:
```bash
make test
make lint
```

3. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

4. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Code Guidelines

- Write clear, readable code
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions small and focused
- Write tests for new functionality

## Questions?

If you have questions or need help, feel free to:
- Open an issue
- Start a discussion
- Contact the maintainers

Happy coding! 🚀
