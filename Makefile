# Copyright 2022- Autonomi AI, Inc. All rights reserved.
.PHONY: default clean clean-build clean-pyc clean-test test test-coverage develop install style

default: test

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

test: ## Basic testing using pytest (see pytest.ini)
	pytest -sv tests -k "not (skip)"

develop: ## Install wheel package in developer/editable-mode
	python -m pip install --upgrade pip
	pip install --upgrade pip setuptools
	pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
	pip install --editable '.[dev,docs]'
	python -c 'from nos._version import __version__; print(f"nos=={__version__}")'

install: ## Install wheel package
	pip install .

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -lh dist

style: ## Format source code automatically
	pre-commit run --all-files # Uses pyproject.toml
