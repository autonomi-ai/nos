.DEFAULT_GOAL := default
.PHONY: default clean clean-build clean-pyc clean-test test test-coverage develop install style
SHELL := /bin/bash

include makefiles/Makefile.base.mk
include makefiles/Makefile.mmdet.mk

default: test;

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

post-install-check: ## Post-install checks (check version, etc)
	python -c 'from nos.version import __version__; print(f"nos=={__version__}")'

develop: ## Install GPU dependencies and package in developer/editable-mode
	python -m pip install --upgrade pip
	pip install --upgrade pip setuptools
	pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
	pip install --editable '.[dev,test,docs]'
	make post-install-check

develop-cpu: ## Install CPU dependencies and package in developer/editable-mode
	python -m pip install --upgrade pip
	pip install --upgrade pip setuptools
	pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
	pip install --editable '.[dev,test,docs]'
	make post-install-check

install: ## Install wheel package
	pip install .

lint: ## Format source code automatically
	pre-commit run --all-files # Uses pyproject.toml

test: ## Basic testing using pytest (see pytest.ini)
	pytest -sv tests

test-cpu: ## Basic CPU testing using pytest (see pytest.ini)
	CUDA_VISIBLE_DEVICES="" \
	make test

test-benchmark: ## Testing with benchmarks
	pytest -sv tests -k "benchmark"

test-e2e:
	pytest -sv tests -k "e2e"

dist: clean ## builds source and wheel package
	python -m build --sdist --wheel
	ls -lh dist

update-conda: ## Update conda environment
	if [ ! -e "conda/envs/$(CONDA_DEFAULT_ENV)/env.yml" ]; then \
		## create the file if it doesn't exist \
		mkdir -p conda/envs/$(CONDA_DEFAULT_ENV); \
		touch conda/envs/$(CONDA_DEFAULT_ENV)/env.yml; \
		conda env export --file conda/envs/$(CONDA_DEFAULT_ENV)/env.yml; \
	else \
		conda env update --file conda/envs/$(CONDA_DEFAULT_ENV)/env.yml; \
	fi