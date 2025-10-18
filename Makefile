.PHONY: check test unit integration golden determinism metrics lint type sec clean

check: lint type sec unit integration golden

lint:
	python -m ruff check src tests
	python -m ruff format --check src tests

type:
	mypy --strict --ignore-missing-imports src

sec:
	bandit -ll -q -r src

unit:
	pytest -q tests/unit

integration:
	pytest -q tests/integration

golden:
	pytest -q tests/golden

determinism:
	@if [ -f src/modeling/determinism_check.py ]; then \
		python -m src.modeling.determinism_check --seed 42; \
	else \
		echo "⚠️  Determinism check skipped (module not implemented yet)"; \
	fi

metrics:
	@if [ -f src/eval/validate_metrics.py ]; then \
		python -m src.eval.validate_metrics artifacts/portfolio/perf_metrics.json; \
	else \
		echo "⚠️  Metrics validation skipped (module not implemented yet)"; \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
