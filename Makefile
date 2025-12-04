.PHONY: help install lint format format-check test test-verbose clean all check

# Default target - show help
help:
	@echo "Rankers Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install        - Install package and dependencies"
	@echo "  make lint           - Run ruff linting (read-only check)"
	@echo "  make lint-fix       - Run ruff linting with auto-fix"
	@echo "  make format         - Format code with ruff"
	@echo "  make format-check   - Check code formatting without modifying"
	@echo "  make test           - Run pytest with short output"
	@echo "  make test-verbose   - Run pytest with verbose output"
	@echo "  make check          - Run all checks (lint + format-check)"
	@echo "  make all            - Run all checks and tests (CI simulation)"
	@echo "  make clean          - Clean up cache and temporary files"
	@echo ""
	@echo "GitHub Actions simulation:"
	@echo "  make all            - Simulates full CI/CD pipeline"

# Install dependencies
install:
	@echo "Installing package and dependencies..."
	python -m pip install --upgrade pip
	pip install -e ".[dev,docs]"
	@echo "✓ Installation complete"

# Linting (read-only check - matches python-package.yml)
lint:
	@echo "Running ruff lint check..."
	ruff check .
	@echo "✓ Lint check passed"

# Linting with auto-fix (matches lint-and-format.yml)
lint-fix:
	@echo "Running ruff lint with auto-fix..."
	ruff check . --fix
	@echo "✓ Lint auto-fix complete"

# Format code (matches lint-and-format.yml)
format:
	@echo "Formatting code with ruff..."
	ruff format .
	@echo "✓ Code formatting complete"

# Format check only (matches python-package.yml)
format-check:
	@echo "Checking code formatting..."
	ruff format . --check
	@echo "✓ Format check passed"

# Run tests with short output
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Tests passed"

# Run tests with verbose output
test-verbose:
	@echo "Running tests (verbose)..."
	pytest tests/ -vv --tb=long
	@echo "✓ Tests passed"

# Run both lint and format checks (no modifications)
check: lint format-check
	@echo "✓ All checks passed"

# Full CI/CD simulation - matches GitHub Actions exactly
all: check test
	@echo ""
	@echo "========================================="
	@echo "✓✓✓ ALL CHECKS PASSED ✓✓✓"
	@echo "========================================="
	@echo "Your code is ready to commit!"
	@echo ""
	@echo "Summary:"
	@echo "  - Linting: ✓"
	@echo "  - Formatting: ✓"
	@echo "  - Tests: ✓"
	@echo ""

# Clean up cache and temporary files
clean:
	@echo "Cleaning up cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete"
