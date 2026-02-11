# Contributing to Qanta

Thank you for your interest in contributing to Qanta!

## Development Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/QminersOrg/qanta.git
cd qanta
uv sync --all-extras
```

## Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Linting
uv run ruff check .        # Check for issues
uv run ruff check . --fix  # Auto-fix issues
uv run ruff format .       # Format code

# Type checking
uv run mypy src/

# Run all checks (mimics CI)
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest
```

## Code Style

- **Formatting**: Handled by `ruff format`
- **Quotes**: Single quotes for strings, triple double quotes for docstrings
- **Types**: Strict type hints required (enforced by mypy)
- **Line length**: 88 characters

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). This enables automatic changelog generation and clear history.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change (no new feature or fix) |
| `test` | Adding or fixing tests |
| `chore` | Maintenance (deps, CI, etc.) |
| `perf` | Performance improvement |
| `style` | Formatting, whitespace |

### Examples

```bash
git commit -m "feat: add portfolio optimizer"
git commit -m "fix: handle empty data arrays"
git commit -m "docs: add API examples to README"
git commit -m "chore: update dependencies"
git commit -m "feat(timeseries): add exponential smoothing"
```

### Breaking Changes

For breaking changes, add `!` after the type:

```bash
git commit -m "feat!: change return type of calculate()"
```

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with conventional commits
3. Ensure all checks pass (`ruff`, `mypy`, `pytest`)
4. Open a PR against `master`
5. Wait for review and CI to pass

## Releasing

Releases are handled by maintainers:

1. Merge PRs to `master`
2. Create and push a version tag: `git tag v1.2.3 && git push origin v1.2.3`
3. CI automatically publishes to PyPI and creates a GitHub Release

