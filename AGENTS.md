# AI Agent Instructions for Qanta

This document provides guidelines for AI agents working on this codebase.

## Project Context

- **Type**: Python library for analysts and quants
- **Config**: All settings in `pyproject.toml`
- **Contributing**: See `CONTRIBUTING.md` for code style, commits, and dev setup

## Guidelines for AI Agents

1. **Don't commit** unless explicitly asked
2. **Run checks** before suggesting code is complete:
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest
   ```
3. **Use conventional commits** — see `CONTRIBUTING.md`
4. **Maintain strict typing** — all functions need type hints
5. **Keep it simple** — don't over-engineer
6. **Follow existing patterns** in the codebase
