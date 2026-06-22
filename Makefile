.PHONY: install test lint typecheck check

install:
	uv pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

typecheck:
	.venv/bin/python -m mypy

check: lint typecheck test

