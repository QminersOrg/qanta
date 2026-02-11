# Qanta

[![CI](https://github.com/QminersOrg/qanta/actions/workflows/ci.yml/badge.svg)](https://github.com/QminersOrg/qanta/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/qanta)](https://pypi.org/project/qanta/)
[![Python versions](https://img.shields.io/pypi/pyversions/qanta)](https://pypi.org/project/qanta/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library for analysts and quants.

## Installation

```bash
pip install qanta
```

## Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/QminersOrg/qanta.git
cd qanta
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy src/
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
