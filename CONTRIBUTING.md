# Contributing to Rankers

Thank you for your interest in contributing to the Rankers project! This guide will help you get started.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- git
- pip

### Setting up your development environment

1. **Clone the repository**
```bash
git clone https://github.com/Parry-Parry/rankers.git
cd rankers
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package in editable mode with development dependencies**
```bash
pip install -e ".[docs]"
pip install ruff pytest pre-commit
```

4. **Install pre-commit hooks**
```bash
pre-commit install
```

## Code Style

We use **Ruff** for code linting and formatting. The configuration is defined in `ruff.toml`.

### Running Ruff locally

```bash
# Check for linting errors
ruff check .

# Fix linting errors automatically
ruff check . --fix

# Format code
ruff format .

# Check formatting without making changes
ruff format . --check
```

### Pre-commit hooks

When you have installed pre-commit hooks, they will automatically run on `git commit`:

```bash
git add <your-changes>
git commit -m "Your commit message"
# Pre-commit hooks will run automatically
```

To run pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Testing

We use **pytest** for testing. All tests should be in the `tests/` directory.

### Running tests

```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/unit/test_trainer_init.py

# Run tests matching a pattern
pytest tests/ -k "test_compute_metrics"

# Run with coverage
pytest tests/ --cov=rankers --cov-report=html
```

### Writing tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names that explain what is being tested
- Add docstrings to test functions
- Use fixtures from `tests/conftest.py` or create new ones in the test file

## Version Management

### Automatic version bumping and releases

When you update the version in `pyproject.toml`, a GitHub Actions workflow will:

1. Detect the version change
2. Run all tests to ensure they pass
3. Create a GitHub release
4. Publish the package to PyPI

**To trigger a release:**

1. Update the version in `pyproject.toml`:
```toml
[project]
version = "0.0.7"  # Increment from 0.0.6
```

2. Commit and push your changes:
```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.0.7"
git push origin main
```

3. The GitHub Actions workflow will automatically:
   - Create a tag `v0.0.7`
   - Create a GitHub release
   - Build the package
   - Publish to PyPI
   - Optionally publish to Test PyPI for verification

### Manual version bumping

If you prefer to bump the version manually using a tool:

```bash
# Using bump2version (install with: pip install bump2version)
bump2version patch  # 0.0.6 -> 0.0.7
bump2version minor  # 0.0.7 -> 0.1.0
bump2version major  # 0.1.0 -> 1.0.0
```

## Committing Code

### Commit message conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without feature changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, tools, configuration

**Examples:**

```bash
git commit -m "feat(trainer): add support for custom loss functions"
git commit -m "fix(corpus): handle DataFrame truth value correctly"
git commit -m "docs: add contributing guide"
git commit -m "test: add integration tests for evaluation dataset"
git commit -m "style: auto-format with Ruff"
```

## Pull Requests

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add <files>
git commit -m "feat: your feature description"
```

3. Push your branch:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request on GitHub with:
   - A clear title describing the changes
   - A detailed description of what changed and why
   - Reference to any related issues (e.g., "Fixes #123")

5. The GitHub Actions workflows will automatically:
   - Run Ruff linting and formatting checks
   - Run the full test suite
   - Comment on your PR if formatting changes are needed

## GitHub Actions Workflows

### `python-package.yml`
Runs on every push and PR to main:
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Linting with Ruff
- Formatting checks with Ruff
- Test execution with pytest

### `lint-and-format.yml`
Runs on every push and PR:
- Runs Ruff linting and formatting
- Auto-commits formatting changes on push
- Comments on PRs if formatting is needed

### `version-and-release.yml`
Runs when version in `pyproject.toml` changes:
- Detects version changes
- Runs tests to verify quality
- Creates a GitHub release
- Publishes to PyPI

## Questions?

If you have any questions about contributing, please:
- Open an issue on GitHub
- Check existing documentation in the `docs/` directory
- Look at existing tests for examples

Thank you for contributing to Rankers!
