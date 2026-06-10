.PHONY: lint lint-fix format format-check typecheck check install-dymad-skill check-dymad-skill-install

lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format . --check

typecheck:
	pyright --pythonpath "$(shell which python)"

check: lint format-check typecheck

install-dymad-skill:
	python skills/install_dymad_skills.py

check-dymad-skill-install:
	python skills/install_dymad_skills.py --check
