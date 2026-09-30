.PHONY: install test lint eval app
install:
	uv sync --extra dev
test:
	uv run pytest -q
lint:
	uv run ruff check . && uv run ruff format --check .
eval:
	uv run python evals/run_evals.py --help
app:
	uv run python app.py
