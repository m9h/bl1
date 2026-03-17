.PHONY: lint test coverage quality typecheck all clean

# Run all quality checks
all: lint typecheck test quality

# Linting (fast — run first)
lint:
	ruff check src/bl1/
	ruff format --check src/bl1/

# Fix auto-fixable lint issues
lint-fix:
	ruff check src/bl1/ --fix
	ruff format src/bl1/

# Type checking
typecheck:
	mypy src/bl1 --ignore-missing-imports --exclude "notebooks|benchmarks|scripts"

# Tests (skip slow bursting tests)
test:
	python -m pytest tests/ --ignore=tests/test_bursting_calibration.py -x -q -m "not slow"

# Tests with coverage
coverage:
	python -m pytest tests/ \
		--ignore=tests/test_bursting_calibration.py \
		--cov=src/bl1 --cov-branch \
		--cov-report=term-missing \
		-q -m "not slow"

# Code quality metrics
quality:
	@echo "=== Docstring Coverage ==="
	interrogate src/bl1/
	@echo "\n=== Dead Code ==="
	vulture src/bl1/ --min-confidence 80
	@echo "\n=== Complexity ==="
	radon cc src/bl1/ -a -nc
	@echo "\n=== Security ==="
	bandit -r src/bl1/ -ll -q

# Full test suite including slow tests
test-full:
	python -m pytest tests/ -v --tb=short

# Benchmark (local CPU)
benchmark:
	python benchmarks/profile_scale.py --n-neurons 1000 5000 10000

# Benchmark (Modal A100)
benchmark-gpu:
	modal run benchmarks/modal_benchmark.py

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache coverage.xml htmlcov/
