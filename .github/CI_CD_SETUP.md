# CI/CD Setup Summary

This document summarizes the GitHub Actions workflows and local development tools configured for the Rankers project.

## Workflows Created

### 1. **python-package.yml** (Updated)
**Triggers:** Push to main, Pull requests to main

**Actions:**
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Linting with Ruff (`ruff check .`)
- Format checking with Ruff (`ruff format . --check`)
- Test execution with pytest
- Uses pip caching for faster builds

**Fails if:**
- Ruff lint errors detected
- Code formatting doesn't match Ruff standards
- Any pytest test fails

### 2. **lint-and-format.yml** (New)
**Triggers:** Push to main/develop, Pull requests to main/develop

**Actions:**
- Auto-fixes linting errors with Ruff
- Auto-formats code with Ruff
- Commits changes on push (auto-format branch)
- Comments on PRs if formatting needed
- Fails PR if formatting required

**Smart behavior:**
- On push: Auto-commits formatting changes
- On PR: Comments and fails, requiring user to fix locally

### 3. **version-and-release.yml** (New)
**Triggers:** Version change in `pyproject.toml` on main branch

**Multi-stage workflow:**
1. **check-version**: Detects if version changed
2. **test**: Runs full test suite on Python 3.9-3.11 (if version changed)
3. **create-release**: Builds package and creates GitHub release
4. **publish**: Pushes to PyPI and Test PyPI

**Publishes to:**
- GitHub Releases (with package files)
- PyPI (main registry)
- Test PyPI (optional, for verification)

## Local Development Setup

### Configuration Files Created

1. **ruff.toml**
   - Line length: 100 characters
   - Target Python: 3.9+
   - Rules: E, W, F, I, N, UP, B, C4, ARG, SIM, RUF
   - Excludes: tests, venv, build directories

2. **.pre-commit-config.yaml**
   - Trailing whitespace removal
   - File ending fixes
   - YAML validation
   - Large file checks
   - Ruff linting and formatting
   - pyupgrade for modern Python syntax
   - pycln for import cleanup

3. **CONTRIBUTING.md**
   - Development setup instructions
   - Code style guidelines
   - Testing procedures
   - Version management
   - Commit message conventions
   - Pull request guidelines

## Quick Start for Contributors

### Initial Setup
```bash
# Clone and enter directory
git clone https://github.com/Parry-Parry/rankers.git
cd rankers

# Create and activate venv
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[docs]"
pip install ruff pytest pre-commit

# Install pre-commit hooks
pre-commit install
```

### Before Committing
```bash
# Format code automatically
ruff check . --fix
ruff format .

# Run tests
pytest tests/ -v

# Or just commit (pre-commit hooks will run)
git add .
git commit -m "feat: your feature"
```

## Releasing a New Version

### Automatic Release (Recommended)
1. Update version in `pyproject.toml`:
```toml
version = "0.0.7"  # increment from 0.0.6
```

2. Commit and push:
```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.0.7"
git push origin main
```

3. GitHub Actions will automatically:
   - Test the code
   - Create a GitHub release
   - Publish to PyPI

### Manual Release (If needed)
```bash
python -m build
twine upload dist/*
```

## Environment Variables

### Required GitHub Secrets
- `PYPI_API_TOKEN`: PyPI API token for publishing
  - Required for auto-publishing to work
  - Get from: https://pypi.org/manage/account/

### Optional GitHub Secrets
- `TEST_PYPI_API_TOKEN`: Test PyPI token
  - Optional, for testing releases before production
  - Get from: https://test.pypi.org/manage/account/

See `.github/GITHUB_SECRETS.md` for detailed setup instructions.

## Monitoring Workflows

### View Workflow Runs
1. Go to your repository on GitHub
2. Click "Actions" tab
3. See all workflow runs and their status
4. Click any run to see detailed logs

### Debug Failed Workflows
1. Click the failed workflow
2. Click the failed job
3. Expand the failed step to see full error output
4. Make fixes and push to trigger re-run

## Customization

### Change Ruff Rules
Edit `ruff.toml`:
```toml
[lint]
select = ["E", "W", "F", "I"]  # Your rules here
ignore = ["E501", "E203"]      # Your ignores here
```

### Change Test Command
Edit `.github/workflows/python-package.yml`:
```yaml
- name: Test with pytest
  run: |
    pytest tests/ -v --cov=rankers
```

### Change Python Versions
Edit `.github/workflows/python-package.yml`:
```yaml
matrix:
  python-version: ["3.9", "3.10", "3.11"]
```

## Troubleshooting

### Workflows not running
- Check GitHub Actions is enabled in repository settings
- Verify workflow files are in `.github/workflows/`
- Check branch protection rules aren't blocking

### Format checks failing
```bash
# Fix all formatting issues locally
ruff check . --fix
ruff format .
git add .
git commit -m "style: auto-format with Ruff"
git push
```

### Tests failing in CI but passing locally
- Ensure you're using same Python version as CI
- Check for environmental differences
- Run full test suite: `pytest tests/ -v`

## Next Steps

1. **Set up PyPI token**
   - Follow `.github/GITHUB_SECRETS.md`
   - This enables auto-publishing

2. **Inform team members**
   - Share CONTRIBUTING.md with contributors
   - Explain version bumping process

3. **Monitor releases**
   - Watch GitHub Actions for first release
   - Verify package appears on PyPI

4. **Optional: Slack/Discord notifications**
   - Configure status checks webhook
   - Get notified when releases succeed

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Pre-commit Framework](https://pre-commit.com/)
