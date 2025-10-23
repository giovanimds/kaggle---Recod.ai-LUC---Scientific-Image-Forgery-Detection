.PHONY: help install install-dev test lint format clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install project dependencies with uv
	uv pip install -e .

install-dev:  ## Install project with development dependencies
	uv pip install -e ".[dev]"

test:  ## Run tests with pytest
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Lint code with ruff
	ruff check src/ tests/

format:  ## Format code with black
	black src/ tests/ notebooks/
	ruff check --fix src/ tests/

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

jupyter:  ## Start Jupyter notebook server
	jupyter notebook notebooks/

tensorboard:  ## Start TensorBoard for monitoring training
	tensorboard --logdir=runs/
