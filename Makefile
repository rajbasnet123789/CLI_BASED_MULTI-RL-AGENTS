# Orion AI - developer tasks
#
#   make install   editable install with dev + server extras
#   make test      run the test suite
#   make lint      ruff check
#   make fmt       ruff format + autofix
#   make build     build wheel and sdist into dist/
#   make docker    build the Docker image
#   make clean     remove caches and build artifacts

PYTHON ?= python
IMAGE ?= orion-ai:1.0.0

.PHONY: help install install-dev test lint fmt build docker run clean

help:
	@echo "install   - pip install -e '.[dev,server]'"
	@echo "test      - pytest"
	@echo "lint      - ruff check"
	@echo "fmt       - ruff format + ruff check --fix"
	@echo "build     - python -m build"
	@echo "docker    - docker build -t $(IMAGE) ."
	@echo "run       - python -m orion"
	@echo "clean     - remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev,server]"

install-dev: install

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

fmt:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

build:
	$(PYTHON) -m build

docker:
	docker build -t $(IMAGE) .

run:
	$(PYTHON) -m orion

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
