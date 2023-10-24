export DOCKER_BUILDKIT ?= 1
export COMPOSE_DOCKER_CLI_BUILD ?= 1

.DEFAULT_GOAL := help
.PHONY: default clean clean-build clean-pyc clean-test test test-coverage develop install style
SHELL := /bin/bash

NOS_VERSION := $(shell python -c "exec(open('nos/version.py').read().strip()); print(__version__)")
NOS_VERSION_TAG = ${NOS_VERSION}
DOCKER_IMAGE_NAME := autonomi/nos
DOCKER_TARGET := server
DOCKER_ARGS :=
DOCKER_CMD :=

include makefiles/Makefile.base.mk
include makefiles/Makefile.docs.mk
include makefiles/Makefile.agibuild.mk

default: help;

help:
	@echo "nos 🔥: Nitrous Oxide System (NOS) for Computer Vision"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  clean               Remove all build, test, coverage and Python artifacts"
	@echo "  clean-build         Remove build artifacts"
	@echo "  clean-pyc           Remove Python file artifacts"
	@echo "  clean-test          Remove test and coverage artifacts"
	@echo "  post-install-check  Post-install checks (check version, etc)"
	@echo "  develop             Install GPU dependencies and package in developer/editable-mode"
	@echo "  develop-cpu         Install CPU dependencies and package in developer/editable-mode"
	@echo "  install             Install wheel package"
	@echo "  lint                Format source code automatically"
	@echo "  test                Basic GPU/CPU testing with a single GPU"
	@echo "  test-cpu            Basic CPU testing"
	@echo "  test-client         Basic client-side testing"
	@echo "  test-server         Server-side testing (all GPUs)"
	@echo "  test-cpu-benchmark  Testing with benchmarks (CPUs)"
	@echo "  test-gpu-benchmark  Testing with benchmarks (GPUs)"
	@echo "  test-all			 All tests including CPU, GPU, client, and server
	@echo "  dist                Builds source and wheel package"
	@echo ""

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr site/
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

develop-gpu: ## Install GPU dependencies and package in developer/editable-mode
	python -m pip install --upgrade pip
	pip install --editable '.[server,gpu,dev,test,docs]'
	make post-install-check

develop-cpu: ## Install CPU dependencies and package in developer/editable-mode
	python -m pip install --upgrade pip
	pip install --editable '.[server,cpu,dev,test,docs]'
	make post-install-check

install: ## Install wheel package
	pip install .

lint: ## Format source code automatically
	pre-commit run --all-files # Uses pyproject.toml

test: ## Basic CPU testing with mkdocs-build
	make test-cpu
	make mkdocs-build

test-cpu: ## Basic CPU testing
	CUDA_VISIBLE_DEVICES="" \
	pytest -sv tests

test-gpu: ## Basic GPU testing with single GPU
	CUDA_VISIBLE_DEVICES="0" \
	pytest -sv tests

test-client: agi-build-cpu agi-build-gpu ## Basic client-side testing
	CUDA_VISIBLE_DEVICES="" \
	pytest -sv tests/client -m "client"

test-cpu-benchmark: ## Testing with benchmarks (no GPUs)
	CUDA_VISIBLE_DEVICES="" \
	pytest -sv tests -m "benchmark"

test-gpu-benchmark: ## Testing with benchmarks (all GPUs)
	pytest -sv tests -m "benchmark"

test-server: ## Server-side testing (all GPUs)
	pytest -sv tests -m "server"

test-all:  ## All tests including CPU, GPU, client, and server
	make test-cpu test-gpu test-client test-server

dist: clean ## builds source and wheel package
	python -m build --sdist --wheel
	ls -lh dist

test-locust:
	locust -f tests/locustfile.py --config=tests/locust.conf
