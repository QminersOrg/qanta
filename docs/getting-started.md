# Getting Started

This guide will help you install Qanta and start using it in your projects.

## Installation

### Using pip

```bash
pip install qanta
```

### Using uv

```bash
uv add qanta
```

### From Source

```bash
git clone https://github.com/QminersOrg/qanta.git
cd qanta
uv sync --all-extras
```

## Requirements

- Python 3.11 or higher

## Verifying Installation

After installation, verify that Qanta is correctly installed:

```python
import qanta

print(qanta.__version__)
```

## What's Next?

- Explore the [API Reference](reference.md) for detailed documentation
- Check out the [GitHub repository](https://github.com/QminersOrg/qanta) for examples and source code

