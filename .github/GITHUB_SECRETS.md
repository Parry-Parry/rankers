# GitHub Secrets Configuration

This document explains how to set up the required secrets for the CI/CD workflows.

## Required Secrets

### PYPI_API_TOKEN (Required)
Used to publish packages to PyPI.

**How to obtain:**
1. Go to [PyPI.org](https://pypi.org)
2. Log in to your account
3. Navigate to Account Settings → API tokens
4. Create a new API token (or use an existing one)
5. Copy the token (it starts with `pypi-`)

**How to add to GitHub:**
1. Go to your repository on GitHub
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste the token from PyPI
6. Click "Add secret"

### TEST_PYPI_API_TOKEN (Optional)
Used to publish packages to Test PyPI for verification before production release.

**How to obtain:**
1. Go to [Test PyPI](https://test.pypi.org)
2. Log in to your account
3. Navigate to Account Settings → API tokens
4. Create a new API token
5. Copy the token

**How to add to GitHub:**
1. Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `TEST_PYPI_API_TOKEN`
4. Value: Paste the token from Test PyPI
5. Click "Add secret"

## Workflow Behavior

### With PYPI_API_TOKEN set
The `version-and-release.yml` workflow will:
- Detect version changes in `pyproject.toml`
- Run all tests
- Create a GitHub release
- Publish to PyPI automatically

### Without PYPI_API_TOKEN set
The workflow will fail at the PyPI publishing step. You can:
1. Set the secret (recommended)
2. Or manually publish by running:
```bash
python -m build
twine upload dist/*
```

## Testing the Workflow

To test the release workflow without pushing to PyPI:

1. Comment out or remove the PyPI publishing step in `.github/workflows/version-and-release.yml`
2. Create a test commit with a version bump
3. Verify the GitHub release is created
4. Restore the PyPI publishing step

## PyPI Package Metadata

The following is automatically extracted from `pyproject.toml`:
- **Package name**: `rankers`
- **Version**: Updated from `pyproject.toml`
- **Description**: "A package for training and evaluating neural rankers."
- **Homepage**: From `project.urls`
- **License**: Apache License 2.0

Users can install your package with:
```bash
pip install rankers
```

## Troubleshooting

### PyPI Publish Fails with "Invalid distribution" error
- Ensure `setup.py` and `pyproject.toml` are valid
- Run `python -m build` locally to verify
- Check that no files are being included that shouldn't be

### Token Errors
- Verify the token hasn't expired
- Token should be copied exactly without extra spaces
- Ensure you're using the right PyPI instance (PyPI vs Test PyPI)

### Release not created on version bump
- Verify `pyproject.toml` version changed
- Check workflow logs in GitHub Actions
- Ensure you pushed to the `main` branch

## Best Practices

1. **Keep tokens secure**: Never commit tokens or share them
2. **Rotate tokens regularly**: Regenerate tokens every 6-12 months
3. **Use scoped tokens**: If PyPI supports it, create tokens for this repository only
4. **Test locally first**: Run `python -m build` before pushing version bumps
5. **Review releases**: Check the GitHub release was created correctly before users download
