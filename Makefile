.PHONY: test unit-test check-env clean help

help:
	@echo "Available targets:"
	@echo "  make test        - Run all offline unit tests (< 3s, zero external network)"
	@echo "  make check-env   - Run pre-flight environment and dependency diagnostics"
	@echo "  make clean       - Clean temporary build artifacts and __pycache__"

test: unit-test

unit-test:
	python3 -m unittest discover -s tests/unit -p "test_*.py" -v

check-env:
	python3 tools/infrastructure/env_checker.py --format=table

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
