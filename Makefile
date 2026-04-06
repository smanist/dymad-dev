.PHONY: lint lint-fix format format-check typecheck check

lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format . --check

typecheck:
	pyright

check: lint format-check typecheck
