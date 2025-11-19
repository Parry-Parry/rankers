# Quick Reference Guide

## Developer Cheat Sheet

### First Time Setup
```bash
git clone https://github.com/Parry-Parry/rankers.git
cd rankers
python -m venv venv && source venv/bin/activate
pip install -e ".[docs]" ruff pytest pre-commit
pre-commit install
```

### Before Each Commit
```bash
# Format and lint
ruff check . --fix && ruff format .

# Run tests
pytest tests/ -v

# Commit (pre-commit will run hooks)
git add .
git commit -m "type(scope): message"
```

### Testing
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/unit/test_trainer_init.py -v

# With coverage
pytest tests/ --cov=rankers
```

### Code Quality
```bash
# Check linting
ruff check .

# Fix linting issues
ruff check . --fix

# Check formatting
ruff format . --check

# Format code
ruff format .

# Pre-commit all files
pre-commit run --all-files
```

### Releasing (Automatic)
1. Edit `pyproject.toml`: Change `version = "0.0.6"` → `version = "0.0.7"`
2. Commit: `git commit -m "chore: bump version to 0.0.7"`
3. Push: `git push origin main`
4. GitHub Actions does the rest!

## Commit Message Format

```
type(scope): short summary (50 chars max)

Optional detailed explanation (wrap at 72 chars)
Can have multiple paragraphs.

Optional footer with issue references:
Fixes #123
```

### Type Reference
| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Code formatting |
| `refactor` | Code reorganization |
| `perf` | Performance improvement |
| `test` | Test additions/changes |
| `chore` | Maintenance, dependencies |

### Examples
```bash
git commit -m "feat(trainer): add support for custom loss functions"
git commit -m "fix(corpus): handle DataFrame truth value correctly"
git commit -m "test: add integration tests for evaluation"
git commit -m "docs: update API reference"
git commit -m "style: auto-format with Ruff"
```

## GitHub Actions Status

Check repository → Actions tab to see:
- ✅ **Python package**: Tests on Python 3.9-3.12, Ruff linting
- ✅ **Lint and format**: Auto-formatting on push, PR comments
- ✅ **Version and release**: Auto-publish on version bump

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ruff found issues" | Run `ruff check . --fix && ruff format .` |
| "Tests failed locally" | Run `pytest tests/ -v` to debug |
| "Pre-commit failed" | Run `pre-commit run --all-files` |
| "Can't push" | Usually due to pre-commit; fix issues and retry |
| "Release didn't publish" | Check `.github/GITHUB_SECRETS.md` for PYPI_API_TOKEN |

## Useful Links

- [CONTRIBUTING.md](CONTRIBUTING.md) - Full contributing guide
- [CI_CD_SETUP.md](CI_CD_SETUP.md) - Workflow details
- [GITHUB_SECRETS.md](GITHUB_SECRETS.md) - Secret setup
- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Pytest Docs](https://docs.pytest.org/)

## Common Commands

```bash
# Format everything
ruff check . --fix && ruff format .

# Run specific test type
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests

# Check what will be committed
git diff --staged

# Undo last commit (keep changes)
git reset --soft HEAD~1

# See commit history
git log --oneline -10
```

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, version |
| `ruff.toml` | Linting/formatting rules |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `.github/workflows/` | CI/CD automation |
| `tests/` | Test suite |
| `CONTRIBUTING.md` | Contributing guidelines |

---

**Need help?** See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information.
