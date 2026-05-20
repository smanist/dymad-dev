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
	python scripts/install_dymad_train_eval_skill.py

check-dymad-skill-install:
	python scripts/install_dymad_train_eval_skill.py --check
