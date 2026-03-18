.PHONY: lint test coverage quality typecheck all clean

VENV := .venv/bin

# Run all quality checks
all: lint typecheck test quality

# Linting (fast — run first)
lint:
	$(VENV)/ruff check src/bl1/
	$(VENV)/ruff format --check src/bl1/

# Fix auto-fixable lint issues
lint-fix:
	$(VENV)/ruff check src/bl1/ --fix
	$(VENV)/ruff format src/bl1/

# Type checking
typecheck:
	$(VENV)/mypy src/bl1 --ignore-missing-imports --exclude "notebooks|benchmarks|scripts"

# Tests (skip slow bursting tests)
test:
	$(VENV)/python -m pytest tests/ --ignore=tests/test_bursting_calibration.py -x -q -m "not slow"

# Tests with coverage
coverage:
	$(VENV)/python -m pytest tests/ \
		--ignore=tests/test_bursting_calibration.py \
		--cov=src/bl1 --cov-branch \
		--cov-report=term-missing \
		-q -m "not slow"

# Code quality metrics (thresholds match CI)
quality:
	@echo "=== Docstring Coverage ==="
	$(VENV)/interrogate src/bl1/ --fail-under 60
	@echo "\n=== Dead Code ==="
	$(VENV)/vulture src/bl1/ --min-confidence 90
	@echo "\n=== Complexity ==="
	$(VENV)/radon cc src/bl1/ -a -nc
	@echo "\n=== Security ==="
	$(VENV)/bandit -r src/bl1/ -ll -q

# Full test suite including slow tests
test-full:
	$(VENV)/python -m pytest tests/ -v --tb=short

# Benchmark (local CPU)
benchmark:
	$(VENV)/python benchmarks/profile_scale.py --n-neurons 1000 5000 10000

# Benchmark (Modal A100)
benchmark-gpu:
	modal run benchmarks/modal_benchmark.py

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache coverage.xml htmlcov/
